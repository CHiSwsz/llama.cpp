#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct moss_audio_tokenizer_options {
    int n_threads = -1;
};

class LLAMA_API moss_audio_tokenizer {
public:
    explicit moss_audio_tokenizer(
            const std::string & model_path,
            const moss_audio_tokenizer_options & options = {});
    ~moss_audio_tokenizer();

    moss_audio_tokenizer(const moss_audio_tokenizer &) = delete;
    moss_audio_tokenizer & operator=(const moss_audio_tokenizer &) = delete;

    moss_audio_tokenizer(moss_audio_tokenizer &&) noexcept;
    moss_audio_tokenizer & operator=(moss_audio_tokenizer &&) noexcept;

    int sample_rate() const;
    uint32_t downsample_rate() const;
    uint32_t num_quantizers() const;

    std::vector<float> decode(
            const std::vector<llama_token> & codes,
            size_t n_frames,
            uint32_t n_quantizers = 0) const;

    std::vector<llama_token> encode(
            const std::vector<float> & audio,
            size_t * out_frames = nullptr,
            uint32_t n_quantizers = 0) const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

LLAMA_API int moss_audio_model_sample_rate(const struct llama_model * model);

LLAMA_API uint32_t moss_audio_model_downsample_rate(const struct llama_model * model);

LLAMA_API uint32_t moss_audio_model_num_quantizers(const struct llama_model * model);

LLAMA_API std::vector<llama_token> moss_audio_model_quantizer_encode(
        const struct llama_model * model,
        const std::vector<float> & input,
        size_t n_frames,
        uint32_t n_quantizers = 0);
