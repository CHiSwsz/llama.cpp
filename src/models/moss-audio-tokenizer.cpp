#include "llama-moss-audio-tokenizer.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char MOSS_CODEC_ARCH[] = "moss-audio-tokenizer";
constexpr float MOSS_LAYER_NORM_EPS = 1e-5f;
constexpr size_t MOSS_CODEC_MAX_NODES_BASE = 256;
constexpr size_t MOSS_CODEC_MAX_NODES_PER_LAYER = 32;

enum class moss_codec_module_type {
    PATCHED_PRETRANSFORM,
    TRANSFORMER,
};

struct moss_codec_transformer_layer {
    ggml_tensor * attn_in   = nullptr;
    ggml_tensor * attn_out  = nullptr;
    ggml_tensor * linear1   = nullptr;
    ggml_tensor * linear2   = nullptr;
    ggml_tensor * norm1_w   = nullptr;
    ggml_tensor * norm1_b   = nullptr;
    ggml_tensor * norm2_w   = nullptr;
    ggml_tensor * norm2_b   = nullptr;
    ggml_tensor * scale1    = nullptr;
    ggml_tensor * scale2    = nullptr;
};

struct moss_codec_transformer_block {
    int input_dimension   = 0;
    int output_dimension  = 0;
    int d_model           = 0;
    int num_heads         = 0;
    int num_layers        = 0;
    int dim_feedforward   = 0;
    int context           = 0;
    float max_period      = 10000.0f;

    ggml_tensor * input_proj  = nullptr;
    ggml_tensor * output_proj = nullptr;

    std::vector<moss_codec_transformer_layer> layers;
};

struct moss_codec_module {
    moss_codec_module_type type = moss_codec_module_type::PATCHED_PRETRANSFORM;
    int patch_size = 1;
    moss_codec_transformer_block transformer;
};

struct moss_codec_quantizer_entry {
    ggml_tensor * in_proj_w   = nullptr;
    ggml_tensor * in_proj_b   = nullptr;
    ggml_tensor * codebook    = nullptr;
    ggml_tensor * out_proj_w  = nullptr;
    ggml_tensor * out_proj_b  = nullptr;
};

struct moss_codec_quantizer {
    int input_dim         = 0;
    int rvq_dim          = 0;
    int output_dim       = 0;
    int num_quantizers   = 0;
    int codebook_size    = 0;
    int codebook_dim     = 0;

    ggml_tensor * input_proj_w  = nullptr;
    ggml_tensor * input_proj_b  = nullptr;
    ggml_tensor * output_proj_w = nullptr;
    ggml_tensor * output_proj_b = nullptr;

    std::vector<moss_codec_quantizer_entry> quantizers;
};

static std::string moss_codec_module_type_to_string(const moss_codec_module_type type) {
    switch (type) {
        case moss_codec_module_type::PATCHED_PRETRANSFORM:
            return "PatchedPretransform";
        case moss_codec_module_type::TRANSFORMER:
            return "Transformer";
    }
    return "Unknown";
}

static moss_codec_module_type moss_codec_module_type_from_string(const std::string & value) {
    if (value == "PatchedPretransform") {
        return moss_codec_module_type::PATCHED_PRETRANSFORM;
    }
    if (value == "Transformer") {
        return moss_codec_module_type::TRANSFORMER;
    }
    throw std::runtime_error("unsupported codec module type: " + value);
}

static void moss_codec_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (backend == nullptr || n_threads <= 0) {
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return;
    }

    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn != nullptr) {
        fn(backend, n_threads);
    }
}

static std::vector<int32_t> moss_codec_make_positions(const size_t n_tokens) {
    std::vector<int32_t> positions(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        positions[i] = (int32_t) i;
    }
    return positions;
}

static std::vector<float> moss_codec_make_causal_mask(const size_t n_tokens, const int context) {
    std::vector<float> mask(n_tokens * n_tokens, -std::numeric_limits<float>::infinity());

    for (size_t iq = 0; iq < n_tokens; ++iq) {
        for (size_t ik = 0; ik < n_tokens; ++ik) {
            if (ik > iq) {
                continue;
            }
            if (context > 0 && (int) (iq - ik) >= context) {
                continue;
            }
            mask[iq * n_tokens + ik] = 0.0f;
        }
    }

    return mask;
}

static ggml_tensor * moss_codec_build_layer_norm(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * bias) {
    cur = ggml_norm(ctx0, cur, MOSS_LAYER_NORM_EPS);
    cur = ggml_mul(ctx0, cur, weight);
    cur = ggml_add(ctx0, cur, bias);
    return cur;
}

static ggml_tensor * moss_codec_build_attention(
        ggml_context * ctx0,
        ggml_tensor * wo,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale) {
    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
    v = ggml_cont(ctx0, v);

    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
    ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);

    if (wo != nullptr) {
        cur = ggml_mul_mat(ctx0, wo, cur);
    }

    return cur;
}

static std::vector<float> moss_codec_patch_decode(
        const std::vector<float> & input,
        const int channels,
        const size_t n_frames,
        const int patch_size) {
    if (patch_size <= 0) {
        throw std::runtime_error("invalid patch size");
    }
    if (channels % patch_size != 0) {
        throw std::runtime_error("patch decode channels not divisible by patch size");
    }
    if (input.size() != (size_t) channels * n_frames) {
        throw std::runtime_error("patch decode input size mismatch");
    }

    const int out_channels = channels / patch_size;
    const size_t out_frames = n_frames * (size_t) patch_size;
    std::vector<float> output((size_t) out_channels * out_frames);

    for (size_t t = 0; t < n_frames; ++t) {
        for (int d = 0; d < out_channels; ++d) {
            for (int i = 0; i < patch_size; ++i) {
                const float value = input[(size_t) (d * patch_size + i) + t * (size_t) channels];
                output[(size_t) d + (t * (size_t) patch_size + (size_t) i) * (size_t) out_channels] = value;
            }
        }
    }

    return output;
}

static std::vector<float> moss_codec_patch_encode(
        const std::vector<float> & input,
        const int channels,
        const size_t n_frames,
        const int patch_size) {
    if (patch_size <= 0) {
        throw std::runtime_error("invalid patch size");
    }
    if (n_frames % (size_t) patch_size != 0) {
        throw std::runtime_error("patch encode frame count not divisible by patch size");
    }
    if (input.size() != (size_t) channels * n_frames) {
        throw std::runtime_error("patch encode input size mismatch");
    }

    const int out_channels = channels * patch_size;
    const size_t out_frames = n_frames / (size_t) patch_size;
    std::vector<float> output((size_t) out_channels * out_frames);

    for (size_t t = 0; t < out_frames; ++t) {
        for (int d = 0; d < channels; ++d) {
            for (int i = 0; i < patch_size; ++i) {
                const float value = input[(size_t) d + (t * (size_t) patch_size + (size_t) i) * (size_t) channels];
                output[(size_t) (d * patch_size + i) + t * (size_t) out_channels] = value;
            }
        }
    }

    return output;
}

static std::vector<float> moss_codec_copy_f32_output(ggml_tensor * tensor) {
    std::vector<float> output((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));
    return output;
}

struct moss_codec_linear_f32 {
    int in_features = 0;
    int out_features = 0;
    std::vector<float> weight;
    std::vector<float> bias;

    bool empty() const {
        return weight.empty();
    }
};

struct moss_codec_quantizer_entry_f32 {
    moss_codec_linear_f32 in_proj;
    moss_codec_linear_f32 out_proj;
    int codebook_size = 0;
    int codebook_dim = 0;
    std::vector<float> codebook;
    std::vector<float> codebook_unit;
};

static std::vector<float> moss_codec_tensor_to_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return {};
    }

    const size_t n_elements = (size_t) ggml_nelements(tensor);

    switch (tensor->type) {
        case GGML_TYPE_F32: {
            std::vector<float> values(n_elements);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), values.data(), 0, ggml_nbytes(tensor));
            return values;
        }
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> values_f16(n_elements);
            std::vector<float> values(n_elements);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), values_f16.data(), 0, ggml_nbytes(tensor));
            for (size_t i = 0; i < n_elements; ++i) {
                values[i] = ggml_fp16_to_fp32(values_f16[i]);
            }
            return values;
        }
        default:
            throw std::runtime_error("unsupported tensor dtype for float conversion: " + std::string(ggml_type_name(tensor->type)));
    }
}

static moss_codec_linear_f32 moss_codec_linear_from_tensors(ggml_tensor * weight, ggml_tensor * bias) {
    moss_codec_linear_f32 result;
    if (weight == nullptr) {
        return result;
    }

    switch (ggml_n_dims(weight)) {
        case 2:
            result.in_features = (int) weight->ne[0];
            result.out_features = (int) weight->ne[1];
            break;
        case 3:
            if (weight->ne[0] != 1) {
                throw std::runtime_error("expected singleton leading dim for 3D linear weight tensor");
            }
            result.in_features = (int) weight->ne[1];
            result.out_features = (int) weight->ne[2];
            break;
        case 4:
            if (weight->ne[0] != 1 || weight->ne[1] != 1) {
                throw std::runtime_error("expected singleton leading dims for 4D linear weight tensor");
            }
            result.in_features = (int) weight->ne[2];
            result.out_features = (int) weight->ne[3];
            break;
        default:
            throw std::runtime_error("expected 2D/3D/4D linear weight tensor");
    }
    result.weight = moss_codec_tensor_to_f32(weight);
    result.bias = moss_codec_tensor_to_f32(bias);
    return result;
}

static std::vector<float> moss_codec_linear_apply(
        const moss_codec_linear_f32 & linear,
        const std::vector<float> & input,
        const size_t n_frames) {
    if (linear.empty()) {
        return input;
    }
    if (input.size() != (size_t) linear.in_features * n_frames) {
        throw std::runtime_error("linear input size mismatch");
    }

    std::vector<float> output((size_t) linear.out_features * n_frames, 0.0f);
    for (size_t t = 0; t < n_frames; ++t) {
        const float * x = input.data() + t * (size_t) linear.in_features;
        float * y = output.data() + t * (size_t) linear.out_features;

        if (!linear.bias.empty()) {
            std::copy(linear.bias.begin(), linear.bias.end(), y);
        }

        for (int o = 0; o < linear.out_features; ++o) {
            const float * w = linear.weight.data() + (size_t) o * (size_t) linear.in_features;
            float acc = y[o];
            for (int i = 0; i < linear.in_features; ++i) {
                acc += w[i] * x[i];
            }
            y[o] = acc;
        }
    }

    return output;
}

static std::vector<float> moss_codec_normalize_rows(
        const std::vector<float> & input,
        const int row_width) {
    if (row_width <= 0 || input.size() % (size_t) row_width != 0) {
        throw std::runtime_error("invalid row width for normalization");
    }

    std::vector<float> output = input;
    const size_t n_rows = input.size() / (size_t) row_width;
    for (size_t r = 0; r < n_rows; ++r) {
        float norm2 = 0.0f;
        for (int c = 0; c < row_width; ++c) {
            const float v = output[r * (size_t) row_width + (size_t) c];
            norm2 += v * v;
        }
        const float inv = 1.0f / std::sqrt(std::max(norm2, std::numeric_limits<float>::epsilon()));
        for (int c = 0; c < row_width; ++c) {
            output[r * (size_t) row_width + (size_t) c] *= inv;
        }
    }
    return output;
}

struct moss_codec_gguf_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;
    ggml_backend_ptr backend;
    ggml_backend_buffer_ptr buffer;

    std::string fname;
    std::map<std::string, size_t> tensor_offset;
    std::map<std::string, ggml_tensor *> loaded_tensors;
    std::vector<ggml_tensor *> tensors_to_load;

    explicit moss_codec_gguf_loader(const std::string & model_path)
        : fname(model_path),
          backend(ggml_backend_cpu_init()) {
        if (!backend) {
            throw std::runtime_error("failed to initialize CPU backend for codec");
        }

        ggml_context * meta = nullptr;
        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf.reset(gguf_init_from_file(fname.c_str(), params));
        if (!ctx_gguf) {
            throw std::runtime_error("failed to load codec GGUF metadata from: " + fname);
        }

        ctx_meta.reset(meta);

        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        ggml_init_params data_params = {
            /*.mem_size   =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_data.reset(ggml_init(data_params));
        if (!ctx_data) {
            throw std::runtime_error("failed to initialize codec tensor context");
        }
    }

    int find_key(const std::string & key, const bool required = true) const {
        const int idx = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (idx < 0 && required) {
            throw std::runtime_error("GGUF key not found: " + key);
        }
        return idx;
    }

    bool has_key(const std::string & key) const {
        return gguf_find_key(ctx_gguf.get(), key.c_str()) >= 0;
    }

    uint32_t get_u32(const std::string & key, const bool required = true, const uint32_t fallback = 0) const {
        const int idx = find_key(key, required);
        if (idx < 0) {
            return fallback;
        }
        return gguf_get_val_u32(ctx_gguf.get(), idx);
    }

    float get_f32(const std::string & key, const bool required = true, const float fallback = 0.0f) const {
        const int idx = find_key(key, required);
        if (idx < 0) {
            return fallback;
        }
        return gguf_get_val_f32(ctx_gguf.get(), idx);
    }

    std::string get_string(const std::string & key, const bool required = true, const std::string & fallback = {}) const {
        const int idx = find_key(key, required);
        if (idx < 0) {
            return fallback;
        }
        return std::string(gguf_get_val_str(ctx_gguf.get(), idx));
    }

    ggml_tensor * get_tensor(const std::string & name, const bool required = true) {
        const auto it = loaded_tensors.find(name);
        if (it != loaded_tensors.end()) {
            return it->second;
        }

        ggml_tensor * meta_tensor = ggml_get_tensor(ctx_meta.get(), name.c_str());
        if (!meta_tensor) {
            if (required) {
                throw std::runtime_error("codec tensor not found: " + name);
            }
            return nullptr;
        }

        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data.get(), meta_tensor);
        ggml_set_name(data_tensor, meta_tensor->name);
        loaded_tensors.emplace(name, data_tensor);
        tensors_to_load.push_back(data_tensor);
        return data_tensor;
    }

    void load_tensor_bytes() {
        if (!buffer) {
            buffer.reset(ggml_backend_alloc_ctx_tensors(ctx_data.get(), backend.get()));
            if (!buffer) {
                throw std::runtime_error("failed to allocate codec weight buffer");
            }
            ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        std::ifstream fin(fname, std::ios::binary);
        if (!fin) {
            throw std::runtime_error("failed to open codec GGUF for tensor loading: " + fname);
        }

        std::vector<uint8_t> read_buf;
        for (ggml_tensor * tensor : tensors_to_load) {
            const auto it = tensor_offset.find(tensor->name);
            if (it == tensor_offset.end()) {
                throw std::runtime_error("missing GGUF tensor offset for: " + std::string(tensor->name));
            }

            const size_t offset = it->second;
            const size_t num_bytes = ggml_nbytes(tensor);

            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                throw std::runtime_error("failed to seek codec tensor: " + std::string(tensor->name));
            }

            if (ggml_backend_buffer_is_host(buffer.get())) {
                fin.read(reinterpret_cast<char *>(tensor->data), (std::streamsize) num_bytes);
            } else {
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), (std::streamsize) num_bytes);
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, num_bytes);
            }

            if (!fin) {
                throw std::runtime_error("failed to read codec tensor: " + std::string(tensor->name));
            }
        }
    }
};

} // namespace

struct moss_audio_tokenizer::impl {
    int sample_rate = 0;
    uint32_t downsample_rate = 0;
    uint32_t num_quantizers = 0;
    int n_threads = -1;

    ggml_backend_ptr backend;
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;
    ggml_backend_buffer_ptr weights_buffer;

    moss_codec_quantizer quantizer;
    moss_codec_linear_f32 quantizer_input_proj_f32;
    std::vector<moss_codec_quantizer_entry_f32> quantizer_entries_f32;
    std::vector<moss_codec_module> encoder;
    std::vector<moss_codec_module> decoder;

    explicit impl(const std::string & model_path, const moss_audio_tokenizer_options & options) {
        moss_codec_gguf_loader loader(model_path);

        if (!loader.has_key(std::string(MOSS_CODEC_ARCH) + ".quantizer_type")) {
            throw std::runtime_error("model does not contain bundled MOSS audio tokenizer metadata");
        }

        sample_rate = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".sampling_rate");
        downsample_rate = loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".downsample_rate");
        num_quantizers = loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.num_quantizers");
        n_threads = options.n_threads;

        quantizer.input_dim = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.input_dim");
        quantizer.rvq_dim = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.rvq_dim");
        quantizer.output_dim = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.output_dim");
        quantizer.num_quantizers = (int) num_quantizers;
        quantizer.codebook_size = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.codebook_size");
        quantizer.codebook_dim = (int) loader.get_u32(std::string(MOSS_CODEC_ARCH) + ".quantizer.codebook_dim");
        quantizer.input_proj_w = loader.get_tensor("audio_tokenizer.quantizer.input_proj.weight", false);
        quantizer.input_proj_b = loader.get_tensor("audio_tokenizer.quantizer.input_proj.bias", false);
        quantizer.output_proj_w = loader.get_tensor("audio_tokenizer.quantizer.output_proj.weight", false);
        quantizer.output_proj_b = loader.get_tensor("audio_tokenizer.quantizer.output_proj.bias", false);
        quantizer.quantizers.resize(num_quantizers);
        for (uint32_t iq = 0; iq < num_quantizers; ++iq) {
            auto & entry = quantizer.quantizers[iq];
            const std::string prefix = "audio_tokenizer.quantizer.quantizers." + std::to_string(iq);
            entry.in_proj_w = loader.get_tensor(prefix + ".in_proj.weight", false);
            entry.in_proj_b = loader.get_tensor(prefix + ".in_proj.bias", false);
            entry.codebook = loader.get_tensor(prefix + ".codebook.weight");
            entry.out_proj_w = loader.get_tensor(prefix + ".out_proj.weight", false);
            entry.out_proj_b = loader.get_tensor(prefix + ".out_proj.bias", false);
        }

        const auto load_modules = [&](const std::string & section_name, std::vector<moss_codec_module> & modules) {
            const uint32_t block_count = loader.get_u32(std::string(MOSS_CODEC_ARCH) + "." + section_name + ".block_count");
            modules.resize(block_count);
            for (uint32_t ib = 0; ib < block_count; ++ib) {
                const std::string block_prefix = std::string(MOSS_CODEC_ARCH) + "." + section_name + "." + std::to_string(ib);
                moss_codec_module & block = modules[ib];
                block.type = moss_codec_module_type_from_string(loader.get_string(block_prefix + ".module_type"));

                if (block.type == moss_codec_module_type::PATCHED_PRETRANSFORM) {
                    block.patch_size = (int) loader.get_u32(block_prefix + ".patch_size");
                    continue;
                }

                auto & tr = block.transformer;
                tr.input_dimension = (int) loader.get_u32(block_prefix + ".input_dimension");
                tr.output_dimension = (int) loader.get_u32(block_prefix + ".output_dimension");
                tr.d_model = (int) loader.get_u32(block_prefix + ".d_model");
                tr.num_heads = (int) loader.get_u32(block_prefix + ".num_heads");
                tr.num_layers = (int) loader.get_u32(block_prefix + ".num_layers");
                tr.dim_feedforward = (int) loader.get_u32(block_prefix + ".dim_feedforward");
                tr.context = (int) loader.get_u32(block_prefix + ".context");
                tr.max_period = loader.get_f32(block_prefix + ".max_period", false, 10000.0f);
                tr.input_proj = loader.get_tensor("audio_tokenizer." + section_name + "." + std::to_string(ib) + ".input_proj.weight", false);
                tr.output_proj = loader.get_tensor("audio_tokenizer." + section_name + "." + std::to_string(ib) + ".output_proj.weight", false);

                tr.layers.resize(tr.num_layers);
                for (int il = 0; il < tr.num_layers; ++il) {
                    auto & layer = tr.layers[il];
                    const std::string layer_prefix =
                            "audio_tokenizer." + section_name + "." + std::to_string(ib) + ".transformer.layers." + std::to_string(il);
                    layer.attn_in  = loader.get_tensor(layer_prefix + ".self_attn.in_projs.0.weight");
                    layer.attn_out = loader.get_tensor(layer_prefix + ".self_attn.out_projs.0.weight");
                    layer.linear1  = loader.get_tensor(layer_prefix + ".linear1.weight");
                    layer.linear2  = loader.get_tensor(layer_prefix + ".linear2.weight");
                    layer.norm1_w  = loader.get_tensor(layer_prefix + ".norm1.weight");
                    layer.norm1_b  = loader.get_tensor(layer_prefix + ".norm1.bias");
                    layer.norm2_w  = loader.get_tensor(layer_prefix + ".norm2.weight");
                    layer.norm2_b  = loader.get_tensor(layer_prefix + ".norm2.bias");
                    layer.scale1   = loader.get_tensor(layer_prefix + ".layer_scale_1.scale", false);
                    layer.scale2   = loader.get_tensor(layer_prefix + ".layer_scale_2.scale", false);
                }
            }
        };

        load_modules("encoder", encoder);
        load_modules("decoder", decoder);

        loader.load_tensor_bytes();

        backend = std::move(loader.backend);
        ctx_meta = std::move(loader.ctx_meta);
        ctx_gguf = std::move(loader.ctx_gguf);
        ctx_data = std::move(loader.ctx_data);
        weights_buffer = std::move(loader.buffer);

        quantizer_input_proj_f32 = moss_codec_linear_from_tensors(quantizer.input_proj_w, quantizer.input_proj_b);
        quantizer_entries_f32.resize(num_quantizers);
        for (uint32_t iq = 0; iq < num_quantizers; ++iq) {
            auto & dst = quantizer_entries_f32[iq];
            const auto & src = quantizer.quantizers[iq];
            dst.in_proj = moss_codec_linear_from_tensors(src.in_proj_w, src.in_proj_b);
            dst.out_proj = moss_codec_linear_from_tensors(src.out_proj_w, src.out_proj_b);
            dst.codebook_dim = (int) src.codebook->ne[0];
            dst.codebook_size = (int) src.codebook->ne[1];
            dst.codebook = moss_codec_tensor_to_f32(src.codebook);
            dst.codebook_unit = moss_codec_normalize_rows(dst.codebook, dst.codebook_dim);
        }

        LLAMA_LOG_INFO("%s: sample_rate=%d downsample_rate=%u num_quantizers=%u encoder_blocks=%zu decoder_blocks=%zu\n",
                __func__, sample_rate, downsample_rate, num_quantizers, encoder.size(), decoder.size());
    }

    std::vector<float> run_quantizer_decode(
            const std::vector<llama_token> & codes,
            const size_t n_frames,
            uint32_t n_quantizers_req) const {
        const uint32_t nq = n_quantizers_req == 0 ? num_quantizers : n_quantizers_req;
        if (nq == 0 || nq > num_quantizers) {
            throw std::runtime_error("invalid quantizer count for decode");
        }
        if (codes.size() != n_frames * (size_t) nq) {
            throw std::runtime_error("raw code size does not match frame count");
        }

        const size_t max_nodes = MOSS_CODEC_MAX_NODES_BASE + (size_t) nq * 8;
        const size_t meta_size = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false);
        std::vector<uint8_t> meta_buf(meta_size);

        ggml_init_params params = {
            /*.mem_size   =*/ meta_size,
            /*.mem_buffer =*/ meta_buf.data(),
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx0 = ggml_init(params);
        if (!ctx0) {
            throw std::runtime_error("failed to init quantizer decode ggml context");
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx0, (int) max_nodes, false);
        std::vector<ggml_tensor *> code_inputs(nq);

        ggml_tensor * cur = nullptr;
        for (uint32_t iq = 0; iq < nq; ++iq) {
            const auto & entry = quantizer.quantizers[iq];
            ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) n_frames);
            ggml_set_input(inp);
            code_inputs[iq] = inp;

            ggml_tensor * emb = ggml_get_rows(ctx0, entry.codebook, inp);
            if (entry.out_proj_w) {
                emb = ggml_mul_mat(ctx0, entry.out_proj_w, emb);
            }
            if (entry.out_proj_b) {
                emb = ggml_add(ctx0, emb, entry.out_proj_b);
            }
            cur = cur ? ggml_add(ctx0, cur, emb) : emb;
        }

        if (quantizer.output_proj_w) {
            cur = ggml_mul_mat(ctx0, quantizer.output_proj_w, cur);
        }
        if (quantizer.output_proj_b) {
            cur = ggml_add(ctx0, cur, quantizer.output_proj_b);
        }

        ggml_build_forward_expand(gf, cur);

        ggml_gallocr_ptr allocr { ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())) };
        ggml_gallocr_alloc_graph(allocr.get(), gf);

        for (uint32_t iq = 0; iq < nq; ++iq) {
            std::vector<int32_t> gathered(n_frames);
            for (size_t t = 0; t < n_frames; ++t) {
                const llama_token code = codes[t * (size_t) nq + iq];
                if (code < 0 || code >= quantizer.codebook_size) {
                    ggml_free(ctx0);
                    throw std::runtime_error("audio code out of codec range during decode");
                }
                gathered[t] = (int32_t) code;
            }
            ggml_backend_tensor_set(code_inputs[iq], gathered.data(), 0, gathered.size() * sizeof(int32_t));
        }

        moss_codec_set_n_threads(backend.get(), n_threads);
        const ggml_status status = ggml_backend_graph_compute(backend.get(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            throw std::runtime_error("quantizer decode graph compute failed");
        }

        std::vector<float> output = moss_codec_copy_f32_output(cur);
        ggml_free(ctx0);
        return output;
    }

    std::vector<float> run_transformer_block(
            const moss_codec_transformer_block & block,
            const std::vector<float> & input,
            const size_t n_frames) const {
        if (input.size() != (size_t) block.input_dimension * n_frames) {
            throw std::runtime_error("transformer block input size mismatch");
        }

        const size_t max_nodes = MOSS_CODEC_MAX_NODES_BASE + (size_t) block.num_layers * MOSS_CODEC_MAX_NODES_PER_LAYER;
        const size_t meta_size = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false);
        std::vector<uint8_t> meta_buf(meta_size);

        ggml_init_params params = {
            /*.mem_size   =*/ meta_size,
            /*.mem_buffer =*/ meta_buf.data(),
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx0 = ggml_init(params);
        if (!ctx0) {
            throw std::runtime_error("failed to init transformer ggml context");
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx0, (int) max_nodes, false);

        ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, block.input_dimension, (int64_t) n_frames);
        ggml_set_input(inp);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) n_frames);
        ggml_set_input(positions);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (int64_t) n_frames, (int64_t) n_frames, 1, 1);
        ggml_set_input(mask);

        ggml_tensor * cur = inp;
        if (block.input_proj) {
            cur = ggml_mul_mat(ctx0, block.input_proj, cur);
        }

        const int d_head = block.d_model / block.num_heads;
        const float attn_scale = 1.0f / std::sqrt((float) d_head);

        for (int il = 0; il < block.num_layers; ++il) {
            const auto & layer = block.layers[il];

            ggml_tensor * inp_sa = cur;
            ggml_tensor * x = moss_codec_build_layer_norm(ctx0, cur, layer.norm1_w, layer.norm1_b);
            ggml_tensor * qkv = ggml_mul_mat(ctx0, layer.attn_in, x);

            ggml_tensor * q = ggml_view_3d(ctx0, qkv, d_head, block.num_heads, (int64_t) n_frames,
                    ggml_row_size(qkv->type, d_head), qkv->nb[1], 0);
            ggml_tensor * k = ggml_view_3d(ctx0, qkv, d_head, block.num_heads, (int64_t) n_frames,
                    ggml_row_size(qkv->type, d_head), qkv->nb[1], ggml_row_size(qkv->type, block.d_model));
            ggml_tensor * v = ggml_view_3d(ctx0, qkv, d_head, block.num_heads, (int64_t) n_frames,
                    ggml_row_size(qkv->type, d_head), qkv->nb[1], ggml_row_size(qkv->type, 2 * block.d_model));

            q = ggml_rope_ext(ctx0, q, positions, nullptr, d_head, 0, 0,
                    block.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx0, k, positions, nullptr, d_head, 0, 0,
                    block.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * attn = moss_codec_build_attention(ctx0, layer.attn_out, q, k, v, mask, attn_scale);
            if (layer.scale1) {
                attn = ggml_mul(ctx0, attn, layer.scale1);
            }
            cur = ggml_add(ctx0, inp_sa, attn);

            ggml_tensor * inp_ff = cur;
            x = moss_codec_build_layer_norm(ctx0, cur, layer.norm2_w, layer.norm2_b);
            x = ggml_mul_mat(ctx0, layer.linear1, x);
            x = ggml_gelu(ctx0, x);
            x = ggml_mul_mat(ctx0, layer.linear2, x);
            if (layer.scale2) {
                x = ggml_mul(ctx0, x, layer.scale2);
            }
            cur = ggml_add(ctx0, inp_ff, x);
        }

        if (block.output_proj) {
            cur = ggml_mul_mat(ctx0, block.output_proj, cur);
        }

        ggml_build_forward_expand(gf, cur);

        ggml_gallocr_ptr allocr { ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())) };
        ggml_gallocr_alloc_graph(allocr.get(), gf);

        const std::vector<int32_t> positions_data = moss_codec_make_positions(n_frames);
        const std::vector<float> mask_data = moss_codec_make_causal_mask(n_frames, block.context);

        ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(positions, positions_data.data(), 0, positions_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

        moss_codec_set_n_threads(backend.get(), n_threads);
        const ggml_status status = ggml_backend_graph_compute(backend.get(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_free(ctx0);
            throw std::runtime_error("transformer graph compute failed");
        }

        std::vector<float> output = moss_codec_copy_f32_output(cur);
        ggml_free(ctx0);
        return output;
    }

    std::vector<float> decode(
            const std::vector<llama_token> & codes,
            const size_t n_frames,
            const uint32_t n_quantizers_req) const {
        uint32_t nq = n_quantizers_req == 0 ? num_quantizers : n_quantizers_req;
        if (nq == 0 || nq > num_quantizers) {
            throw std::runtime_error("invalid quantizer count");
        }

        std::vector<float> cur = run_quantizer_decode(codes, n_frames, nq);
        int channels = quantizer.output_dim;
        size_t frames = n_frames;

        for (const auto & module : decoder) {
            switch (module.type) {
                case moss_codec_module_type::TRANSFORMER:
                    cur = run_transformer_block(module.transformer, cur, frames);
                    channels = module.transformer.output_dimension;
                    break;
                case moss_codec_module_type::PATCHED_PRETRANSFORM:
                    cur = moss_codec_patch_decode(cur, channels, frames, module.patch_size);
                    channels /= module.patch_size;
                    frames *= (size_t) module.patch_size;
                    break;
            }
        }

        if (channels != 1) {
            throw std::runtime_error("codec decoder did not end with a mono waveform channel");
        }

        return cur;
    }

    std::vector<llama_token> run_quantizer_encode(
            const std::vector<float> & input,
            const size_t n_frames,
            const uint32_t n_quantizers_req) const {
        const uint32_t nq = n_quantizers_req == 0 ? num_quantizers : n_quantizers_req;
        if (nq == 0 || nq > num_quantizers) {
            throw std::runtime_error("invalid quantizer count for encode");
        }

        std::vector<float> residual = moss_codec_linear_apply(quantizer_input_proj_f32, input, n_frames);
        if (residual.size() != (size_t) quantizer.rvq_dim * n_frames) {
            throw std::runtime_error("quantizer input projection size mismatch");
        }

        std::vector<llama_token> codes(n_frames * (size_t) nq, 0);
        std::vector<float> latents;
        std::vector<float> latents_unit;
        std::vector<float> decoded;

        for (uint32_t iq = 0; iq < nq; ++iq) {
            const auto & entry = quantizer_entries_f32[iq];
            latents = moss_codec_linear_apply(entry.in_proj, residual, n_frames);
            if (latents.size() != (size_t) entry.codebook_dim * n_frames) {
                throw std::runtime_error("quantizer latent projection size mismatch");
            }

            latents_unit.resize(latents.size());
            for (size_t t = 0; t < n_frames; ++t) {
                const float * in_ptr = latents.data() + t * (size_t) entry.codebook_dim;
                float * out_ptr = latents_unit.data() + t * (size_t) entry.codebook_dim;

                float norm2 = 0.0f;
                for (int d = 0; d < entry.codebook_dim; ++d) {
                    norm2 += in_ptr[d] * in_ptr[d];
                }
                const float inv = 1.0f / std::sqrt(std::max(norm2, std::numeric_limits<float>::epsilon()));
                for (int d = 0; d < entry.codebook_dim; ++d) {
                    out_ptr[d] = in_ptr[d] * inv;
                }
            }

            std::vector<float> codebook_emb((size_t) entry.codebook_dim * n_frames, 0.0f);
            for (size_t t = 0; t < n_frames; ++t) {
                const float * latent = latents_unit.data() + t * (size_t) entry.codebook_dim;

                float best_score = -std::numeric_limits<float>::infinity();
                int best_index = 0;
                for (int code = 0; code < entry.codebook_size; ++code) {
                    const float * row = entry.codebook_unit.data() + (size_t) code * (size_t) entry.codebook_dim;
                    float score = 0.0f;
                    for (int d = 0; d < entry.codebook_dim; ++d) {
                        score += row[d] * latent[d];
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_index = code;
                    }
                }

                codes[t * (size_t) nq + iq] = best_index;
                const float * row = entry.codebook.data() + (size_t) best_index * (size_t) entry.codebook_dim;
                std::copy(row, row + entry.codebook_dim, codebook_emb.begin() + (ptrdiff_t) (t * (size_t) entry.codebook_dim));
            }

            decoded = moss_codec_linear_apply(entry.out_proj, codebook_emb, n_frames);
            if (decoded.size() != residual.size()) {
                throw std::runtime_error("quantizer decoded embedding size mismatch");
            }

            for (size_t i = 0; i < residual.size(); ++i) {
                residual[i] -= decoded[i];
            }
        }

        return codes;
    }

    std::vector<llama_token> encode(
            const std::vector<float> & audio,
            size_t * out_frames,
            const uint32_t n_quantizers_req) const {
        const uint32_t nq = n_quantizers_req == 0 ? num_quantizers : n_quantizers_req;
        if (nq == 0 || nq > num_quantizers) {
            throw std::runtime_error("invalid quantizer count");
        }

        const size_t padded_samples =
                ((audio.size() + (size_t) downsample_rate - 1) / (size_t) downsample_rate) * (size_t) downsample_rate;
        const size_t valid_frames = audio.size() / (size_t) downsample_rate;

        std::vector<float> cur(padded_samples, 0.0f);
        std::copy(audio.begin(), audio.end(), cur.begin());

        int channels = 1;
        size_t frames = padded_samples;

        for (const auto & module : encoder) {
            switch (module.type) {
                case moss_codec_module_type::PATCHED_PRETRANSFORM:
                    cur = moss_codec_patch_encode(cur, channels, frames, module.patch_size);
                    channels *= module.patch_size;
                    frames /= (size_t) module.patch_size;
                    break;
                case moss_codec_module_type::TRANSFORMER:
                    cur = run_transformer_block(module.transformer, cur, frames);
                    channels = module.transformer.output_dimension;
                    break;
            }
        }

        if (channels != quantizer.input_dim) {
            throw std::runtime_error("codec encoder output dimension does not match quantizer input dimension");
        }

        std::vector<llama_token> codes = run_quantizer_encode(cur, frames, nq);
        if (out_frames) {
            *out_frames = valid_frames;
        }

        if (valid_frames >= frames) {
            return codes;
        }

        std::vector<llama_token> trimmed(valid_frames * (size_t) nq);
        for (size_t t = 0; t < valid_frames; ++t) {
            std::copy_n(codes.data() + t * (size_t) nq, nq, trimmed.data() + t * (size_t) nq);
        }
        return trimmed;
    }
};

moss_audio_tokenizer::moss_audio_tokenizer(
        const std::string & model_path,
        const moss_audio_tokenizer_options & options)
    : impl_(std::make_unique<impl>(model_path, options)) {
}

moss_audio_tokenizer::~moss_audio_tokenizer() = default;

moss_audio_tokenizer::moss_audio_tokenizer(moss_audio_tokenizer &&) noexcept = default;

moss_audio_tokenizer & moss_audio_tokenizer::operator=(moss_audio_tokenizer &&) noexcept = default;

int moss_audio_tokenizer::sample_rate() const {
    return impl_->sample_rate;
}

uint32_t moss_audio_tokenizer::downsample_rate() const {
    return impl_->downsample_rate;
}

uint32_t moss_audio_tokenizer::num_quantizers() const {
    return impl_->num_quantizers;
}

std::vector<float> moss_audio_tokenizer::decode(
        const std::vector<llama_token> & codes,
        const size_t n_frames,
        const uint32_t n_quantizers) const {
    return impl_->decode(codes, n_frames, n_quantizers);
}

std::vector<llama_token> moss_audio_tokenizer::encode(
        const std::vector<float> & audio,
        size_t * out_frames,
        const uint32_t n_quantizers) const {
    return impl_->encode(audio, out_frames, n_quantizers);
}

static std::string moss_codec_model_meta_str(const llama_model * model, const std::string & key) {
    const auto it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        throw std::runtime_error("missing GGUF key: " + key);
    }

    std::string value = it->second;
    if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') || (value.front() == '"' && value.back() == '"'))) {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

static uint32_t moss_codec_model_meta_u32(const llama_model * model, const std::string & key) {
    return (uint32_t) std::stoul(moss_codec_model_meta_str(model, key));
}

static const ggml_tensor * moss_codec_model_require_tensor(const llama_model * model, const std::string & name) {
    const ggml_tensor * tensor = model->get_tensor(name.c_str());
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

static const ggml_tensor * moss_codec_model_optional_tensor(const llama_model * model, const std::string & name) {
    return model->get_tensor(name.c_str());
}

int moss_audio_model_sample_rate(const llama_model * model) {
    const std::string arch_name = llm_arch_name(model->arch);
    return (int) moss_codec_model_meta_u32(model, arch_name + ".sampling_rate");
}

uint32_t moss_audio_model_downsample_rate(const llama_model * model) {
    const std::string arch_name = llm_arch_name(model->arch);
    return moss_codec_model_meta_u32(model, arch_name + ".downsample_rate");
}

uint32_t moss_audio_model_num_quantizers(const llama_model * model) {
    const std::string arch_name = llm_arch_name(model->arch);
    return moss_codec_model_meta_u32(model, arch_name + ".quantizer.num_quantizers");
}

std::vector<llama_token> moss_audio_model_quantizer_encode(
        const llama_model * model,
        const std::vector<float> & input,
        size_t n_frames,
        uint32_t n_quantizers_req) {
    if (model->arch != LLM_ARCH_MOSS_TTS_AUDIO_ENCODER) {
        throw std::runtime_error("quantizer encode expects a moss-tts-audio-encoder model");
    }

    const std::string arch_name = llm_arch_name(model->arch);
    const uint32_t num_quantizers = moss_codec_model_meta_u32(model, arch_name + ".quantizer.num_quantizers");
    const uint32_t nq = n_quantizers_req == 0 ? num_quantizers : n_quantizers_req;
    if (nq == 0 || nq > num_quantizers) {
        throw std::runtime_error("invalid quantizer count");
    }

    moss_codec_linear_f32 quantizer_input_proj = moss_codec_linear_from_tensors(
            const_cast<ggml_tensor *>(moss_codec_model_require_tensor(model, "quantizer.input_proj.weight")),
            const_cast<ggml_tensor *>(moss_codec_model_optional_tensor(model, "quantizer.input_proj.bias")));

    std::vector<moss_codec_quantizer_entry_f32> quantizers(nq);
    for (uint32_t iq = 0; iq < nq; ++iq) {
        auto & entry = quantizers[iq];
        const std::string prefix = "quantizer.quantizers." + std::to_string(iq);
        entry.in_proj = moss_codec_linear_from_tensors(
                const_cast<ggml_tensor *>(moss_codec_model_require_tensor(model, prefix + ".in_proj.weight")),
                const_cast<ggml_tensor *>(moss_codec_model_optional_tensor(model, prefix + ".in_proj.bias")));
        entry.out_proj = moss_codec_linear_from_tensors(
                const_cast<ggml_tensor *>(moss_codec_model_require_tensor(model, prefix + ".out_proj.weight")),
                const_cast<ggml_tensor *>(moss_codec_model_optional_tensor(model, prefix + ".out_proj.bias")));
        const ggml_tensor * codebook = moss_codec_model_require_tensor(model, prefix + ".codebook.weight");
        entry.codebook_dim = (int) codebook->ne[0];
        entry.codebook_size = (int) codebook->ne[1];
        entry.codebook = moss_codec_tensor_to_f32(codebook);
        entry.codebook_unit = moss_codec_normalize_rows(entry.codebook, entry.codebook_dim);
    }

    std::vector<float> residual = moss_codec_linear_apply(quantizer_input_proj, input, n_frames);
    std::vector<llama_token> codes(n_frames * (size_t) nq, 0);
    std::vector<float> latents;
    std::vector<float> latents_unit;
    std::vector<float> decoded;

    for (uint32_t iq = 0; iq < nq; ++iq) {
        const auto & entry = quantizers[iq];
        latents = moss_codec_linear_apply(entry.in_proj, residual, n_frames);
        if (latents.size() != (size_t) entry.codebook_dim * n_frames) {
            throw std::runtime_error("quantizer latent projection size mismatch");
        }

        latents_unit.resize(latents.size());
        for (size_t t = 0; t < n_frames; ++t) {
            const float * in_ptr = latents.data() + t * (size_t) entry.codebook_dim;
            float * out_ptr = latents_unit.data() + t * (size_t) entry.codebook_dim;

            float norm2 = 0.0f;
            for (int d = 0; d < entry.codebook_dim; ++d) {
                norm2 += in_ptr[d] * in_ptr[d];
            }
            const float inv = 1.0f / std::sqrt(std::max(norm2, std::numeric_limits<float>::epsilon()));
            for (int d = 0; d < entry.codebook_dim; ++d) {
                out_ptr[d] = in_ptr[d] * inv;
            }
        }

        std::vector<float> codebook_emb((size_t) entry.codebook_dim * n_frames, 0.0f);
        for (size_t t = 0; t < n_frames; ++t) {
            const float * latent = latents_unit.data() + t * (size_t) entry.codebook_dim;

            float best_score = -std::numeric_limits<float>::infinity();
            int best_index = 0;
            for (int code = 0; code < entry.codebook_size; ++code) {
                const float * row = entry.codebook_unit.data() + (size_t) code * (size_t) entry.codebook_dim;
                float score = 0.0f;
                for (int d = 0; d < entry.codebook_dim; ++d) {
                    score += row[d] * latent[d];
                }
                if (score > best_score) {
                    best_score = score;
                    best_index = code;
                }
            }

            codes[t * (size_t) nq + iq] = best_index;
            const float * row = entry.codebook.data() + (size_t) best_index * (size_t) entry.codebook_dim;
            std::copy(row, row + entry.codebook_dim, codebook_emb.begin() + (ptrdiff_t) (t * (size_t) entry.codebook_dim));
        }

        decoded = moss_codec_linear_apply(entry.out_proj, codebook_emb, n_frames);
        if (decoded.size() != residual.size()) {
            throw std::runtime_error("quantizer decoded embedding size mismatch");
        }

        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] -= decoded[i];
        }
    }

    return codes;
}
