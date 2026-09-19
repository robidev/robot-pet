#include "common.h"
#include "whisper.h"
#include <array>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int SAMPLE_RATE = 16000;
constexpr int CHANNELS = 1;
constexpr int BYTES_PER_SAMPLE = 2;

constexpr char MAGIC[4] = {'L', 'G', 'A', '1'};
constexpr uint8_t PROTOCOL_VERSION = 1;

constexpr int UDP_HEADER_SIZE = 20;
constexpr int VAD_WINDOW_SAMPLES = 512; // 32 ms @ 16 kHz

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

// -----------------------------------------------------------------------------
// UDP packet format
//
// Little endian:
//
//   4s  magic
//   B   version
//   B   channels
//   H   sample_count
//   I   sequence
//   I   timestamp
//   I   sample_rate
//
// followed by sample_count int16 PCM samples.
// -----------------------------------------------------------------------------

struct AudioPacket {
    uint8_t channels;
    uint16_t sample_count;
    uint32_t sequence;
    uint32_t timestamp;
    uint32_t sample_rate;
    std::vector<int16_t> samples;
};

static uint16_t read_u16_le(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t read_u32_le(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static int16_t read_i16_le(const uint8_t * p) {
    return static_cast<int16_t>(
        static_cast<uint16_t>(p[0]) |
        (static_cast<uint16_t>(p[1]) << 8)
    );
}

static bool parse_packet(
        const uint8_t * data,
        size_t size,
        AudioPacket & packet) {

    if (size < UDP_HEADER_SIZE) {
        return false;
    }

    if (std::memcmp(data, MAGIC, 4) != 0) {
        return false;
    }

    const uint8_t version = data[4];
    const uint8_t channels = data[5];
    const uint16_t sample_count = read_u16_le(data + 6);
    const uint32_t sequence = read_u32_le(data + 8);
    const uint32_t timestamp = read_u32_le(data + 12);
    const uint32_t sample_rate = read_u32_le(data + 16);

    if (version != PROTOCOL_VERSION) {
        return false;
    }

    if (channels != CHANNELS) {
        return false;
    }

    if (sample_rate != SAMPLE_RATE) {
        return false;
    }

    const size_t payload_size =
        static_cast<size_t>(sample_count) * BYTES_PER_SAMPLE;

    if (size < UDP_HEADER_SIZE + payload_size) {
        return false;
    }

    packet.channels = channels;
    packet.sample_count = sample_count;
    packet.sequence = sequence;
    packet.timestamp = timestamp;
    packet.sample_rate = sample_rate;

    packet.samples.resize(sample_count);

    const uint8_t * pcm = data + UDP_HEADER_SIZE;

    for (size_t i = 0; i < sample_count; ++i) {
        packet.samples[i] = read_i16_le(pcm + i * 2);
    }

    return true;
}

// -----------------------------------------------------------------------------
// Audio level stats (for --log-levels)
//
// Only computed when explicitly enabled, so it has no cost on the default
// path. Useful for figuring out whether microphone gain is too low
// (quiet speech sitting below the VAD threshold) or too high (clipping,
// which hurts both VAD and transcription accuracy).
// -----------------------------------------------------------------------------

struct LevelStats {
    float rms_dbfs = -std::numeric_limits<float>::infinity();
    float peak_dbfs = -std::numeric_limits<float>::infinity();
    float clip_pct = 0.0f;
};

// Samples at or above this magnitude are considered clipped/near full-scale.
constexpr float CLIP_THRESHOLD = 0.99f;

// Iterator-based so it works for both a contiguous std::vector<float>
// (utterance_) and a non-contiguous std::deque<float> (pre_roll_).
template <typename It>
static LevelStats compute_level_stats(It first, It last) {
    LevelStats stats;

    const size_t n = static_cast<size_t>(std::distance(first, last));

    if (n == 0) {
        return stats;
    }

    double sum_sq = 0.0;
    float peak = 0.0f;
    size_t clipped = 0;

    for (It it = first; it != last; ++it) {
        const float a = std::fabs(*it);

        sum_sq += static_cast<double>(*it) * (*it);
        peak = std::max(peak, a);

        if (a >= CLIP_THRESHOLD) {
            ++clipped;
        }
    }

    const float rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(n)));

    if (rms > 0.0f) {
        stats.rms_dbfs = 20.0f * std::log10(rms);
    }

    if (peak > 0.0f) {
        stats.peak_dbfs = 20.0f * std::log10(peak);
    }

    stats.clip_pct =
        100.0f * static_cast<float>(clipped) / static_cast<float>(n);

    return stats;
}

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct Params {
    int port = 5000;
    int threads = 8;

    std::string whisper_model = "models/ggml-base.en.bin";
    std::string vad_model = "models/ggml-silero-v6.2.0.bin";

    float vad_threshold = 0.3f;

    int min_speech_ms = 150;
    int min_silence_ms = 500;
    int speech_pad_ms = 300;

    int max_speech_s = 30;

    bool use_gpu = false;
    bool verbose = false;
    bool log_levels = false;
    bool json_output = false;
};

// -----------------------------------------------------------------------------
// Output
//
// Human mode (default) prints "[Speech detected]" / "[Speech ended]" /
// "[YOU] text" to stdout, as before. JSON mode (--json) prints exactly one
// JSON object per line to stdout and sends all human-readable banners to
// stderr, so a supervising process can parse stdout without heuristics.
// stdout is written from both the receive thread and the transcription
// worker, hence the mutex.
// -----------------------------------------------------------------------------

bool g_json_output = false;
std::mutex g_stdout_mutex;

// Wall-clock UTC as Unix epoch seconds.
static double wall_now() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string json_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 2);

    for (const char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += c;
                }
        }
    }

    return out;
}

static void emit_json_line(const std::string & line) {
    std::lock_guard<std::mutex> lock(g_stdout_mutex);
    std::fputs(line.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

// Human-readable status banners: stdout in human mode, stderr in JSON mode.
static FILE * banner_stream() {
    return g_json_output ? stderr : stdout;
}

static void print_usage(const char * argv0) {
    std::fprintf(stderr, R"(
Usage:
  %s [options]

Options:
  --port N                 UDP port [5000]
  --threads N              Whisper/VAD threads [8]

  --model FILE             Whisper model
                           [models/ggml-base.en.bin]

  --vad-model FILE         Silero VAD model
                           [models/ggml-silero-v6.2.0.bin]

  --vad-threshold N        VAD speech threshold [0.30]

  --min-speech-ms N        Minimum speech duration [150]
  --min-silence-ms N       Silence required to end utterance [500]
  --speech-pad-ms N        Audio padding before/after speech [300]

  --max-speech-s N         Maximum utterance length [30]

  --gpu                    Enable GPU
  --verbose                Show whisper.cpp/VAD internal logs
                           (very noisy: several lines per 32ms window)
  --log-levels             Log RMS/peak/clipping stats for captured
                           audio, to help tune microphone gain [off]
  --json                   Machine-readable output: one JSON object per
                           line on stdout (ready, speech_start,
                           speech_end, text, stopped); banners go to
                           stderr. Times are Unix epoch seconds (UTC).
  --help                   Show this help

Example:

  %s \
    --port 5000 \
    --threads 8 \
    --model models/ggml-base.en.bin \
    --vad-model models/ggml-silero-v6.2.0.bin

)",
                 argv0,
                 argv0);
}

static bool parse_args(int argc, char ** argv, Params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(1);
            }

            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--port") {
            params.port = std::stoi(require_value("--port"));
        } else if (arg == "--threads") {
            params.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--model") {
            params.whisper_model = require_value("--model");
        } else if (arg == "--vad-model") {
            params.vad_model = require_value("--vad-model");
        } else if (arg == "--vad-threshold") {
            params.vad_threshold =
                std::stof(require_value("--vad-threshold"));
        } else if (arg == "--min-speech-ms") {
            params.min_speech_ms =
                std::stoi(require_value("--min-speech-ms"));
        } else if (arg == "--min-silence-ms") {
            params.min_silence_ms =
                std::stoi(require_value("--min-silence-ms"));
        } else if (arg == "--speech-pad-ms") {
            params.speech_pad_ms =
                std::stoi(require_value("--speech-pad-ms"));
        } else if (arg == "--max-speech-s") {
            params.max_speech_s =
                std::stoi(require_value("--max-speech-s"));
        } else if (arg == "--gpu") {
            params.use_gpu = true;
        } else if (arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--log-levels") {
            params.log_levels = true;
        } else if (arg == "--json") {
            params.json_output = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    return true;
}

// -----------------------------------------------------------------------------
// Whisper transcription
// -----------------------------------------------------------------------------

// One finished utterance, with the wall-clock span of the speech it holds
// (padding included), so consumers can line transcripts up with other
// timestamped sensor data.
struct Utterance {
    std::vector<float> audio;
    double t_start_utc = 0.0;
    double t_end_utc = 0.0;
};

class Transcriber {
public:
    bool init(const Params & params) {
        struct whisper_context_params cparams =
            whisper_context_default_params();

        cparams.use_gpu = params.use_gpu;
        cparams.flash_attn = true;

        ctx_ = whisper_init_from_file_with_params(
            params.whisper_model.c_str(),
            cparams);

        if (ctx_ == nullptr) {
            std::fprintf(
                stderr,
                "Failed to initialize Whisper model: %s\n",
                params.whisper_model.c_str());

            return false;
        }

        params_ = &params;

        return true;
    }

    ~Transcriber() {
        if (ctx_) {
            whisper_free(ctx_);
        }
    }

    void transcribe(const Utterance & utterance) {
        const std::vector<float> & audio = utterance.audio;

        if (audio.empty()) {
            return;
        }

        const auto started = std::chrono::steady_clock::now();

        std::fprintf(
            stderr,
            "\n[STT] Transcribing %.2f seconds...\n",
            static_cast<double>(audio.size()) / SAMPLE_RATE);

        whisper_full_params wparams =
            whisper_full_default_params(
                WHISPER_SAMPLING_GREEDY);

        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;

        wparams.translate = false;
        wparams.language = "en";

        wparams.n_threads = params_->threads;

        wparams.no_context = true;
        wparams.single_segment = false;

        wparams.no_timestamps = true;

        wparams.temperature = 0.0f;

        wparams.audio_ctx = 0;

        const int result = whisper_full(
            ctx_,
            wparams,
            audio.data(),
            static_cast<int>(audio.size()));

        if (result != 0) {
            std::fprintf(
                stderr,
                "[STT] whisper_full() failed: %d\n",
                result);

            return;
        }

        const int n_segments = whisper_full_n_segments(ctx_);

        std::string text;

        // Highest per-segment no-speech probability: a cheap signal for
        // consumers to reject transcripts hallucinated from noise.
        float no_speech_prob = 0.0f;

        for (int i = 0; i < n_segments; ++i) {
            const char * segment =
                whisper_full_get_segment_text(ctx_, i);

            if (segment) {
                text += segment;
            }

            no_speech_prob = std::max(
                no_speech_prob,
                whisper_full_get_segment_no_speech_prob(ctx_, i));
        }

        // Trim whitespace.
        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.front()))) {
            text.erase(text.begin());
        }

        while (!text.empty() &&
               std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }

        if (g_json_output) {
            // Emitted even when empty, so the consumer knows the utterance
            // it saw start/end has been fully resolved.
            const double transcribe_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();

            char fields[160];
            std::snprintf(
                fields, sizeof(fields),
                "\"t_start_utc\":%.3f,\"t_end_utc\":%.3f,"
                "\"no_speech_prob\":%.3f,\"transcribe_ms\":%.0f",
                utterance.t_start_utc, utterance.t_end_utc,
                no_speech_prob, transcribe_ms);

            emit_json_line(
                std::string("{\"type\":\"text\",\"text\":\"") +
                json_escape(text) + "\"," + fields + "}");
        } else if (!text.empty()) {
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            std::printf("\n[YOU] %s\n", text.c_str());
            std::fflush(stdout);
        }
    }

private:
    whisper_context * ctx_ = nullptr;
    const Params * params_ = nullptr;
};

// -----------------------------------------------------------------------------
// Background transcription worker
//
// whisper_full() is not guaranteed to run faster than real time, so it must
// not block the UDP receive loop. Finished utterances are handed off to a
// dedicated worker thread that transcribes them one at a time while the
// main thread keeps draining the socket.
// -----------------------------------------------------------------------------

class TranscriptionWorker {
public:
    void start(Transcriber * transcriber) {
        transcriber_ = transcriber;
        thread_ = std::thread(&TranscriptionWorker::run, this);
    }

    void enqueue(Utterance utterance) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(utterance));
        }

        cv_.notify_one();
    }

    // Signals the worker to exit once the queue has been drained, and
    // waits for it to do so.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }

        cv_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        for (;;) {
            Utterance utterance;

            {
                std::unique_lock<std::mutex> lock(mutex_);

                cv_.wait(lock, [this] {
                    return !queue_.empty() || stopping_;
                });

                if (queue_.empty()) {
                    if (stopping_) {
                        return;
                    }

                    continue;
                }

                utterance = std::move(queue_.front());
                queue_.pop();
            }

            transcriber_->transcribe(utterance);
        }
    }

    Transcriber * transcriber_ = nullptr;
    std::thread thread_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Utterance> queue_;
    bool stopping_ = false;
};

// -----------------------------------------------------------------------------
// Streaming VAD
// -----------------------------------------------------------------------------

class StreamingVad {
public:
    bool init(const Params & params) {
        struct whisper_vad_context_params ctx_params =
            whisper_vad_default_context_params();

        ctx_params.n_threads = params.threads;
        ctx_params.use_gpu = params.use_gpu;

        ctx_ = whisper_vad_init_from_file_with_params(
            params.vad_model.c_str(),
            ctx_params);

        if (!ctx_) {
            std::fprintf(
                stderr,
                "Failed to initialize VAD model: %s\n",
                params.vad_model.c_str());

            return false;
        }

        threshold_ = params.vad_threshold;

        return true;
    }

    ~StreamingVad() {
        if (ctx_) {
            whisper_vad_free(ctx_);
        }
    }

    float process(const float * samples, int n_samples) {
        if (!whisper_vad_detect_speech_no_reset(
                ctx_,
                samples,
                n_samples)) {

            return 0.0f;
        }

        const int n_probs = whisper_vad_n_probs(ctx_);

        if (n_probs <= 0) {
            return 0.0f;
        }

        float * probs = whisper_vad_probs(ctx_);

        if (!probs) {
            return 0.0f;
        }

        return probs[n_probs - 1];
    }

    bool is_speech(float probability) const {
        return probability >= threshold_;
    }

    void reset() {
        whisper_vad_reset_state(ctx_);
    }

private:
    whisper_vad_context * ctx_ = nullptr;
    float threshold_ = 0.5f;
};

// -----------------------------------------------------------------------------
// Main streaming speech state machine
// -----------------------------------------------------------------------------

class SpeechRecognizer {
public:
    bool init(const Params & params) {
        params_ = &params;

        if (!vad_.init(params)) {
            return false;
        }

        if (!transcriber_.init(params)) {
            return false;
        }

        speech_start_padding_samples_ =
            params.speech_pad_ms * SAMPLE_RATE / 1000;

        min_speech_samples_ =
            params.min_speech_ms * SAMPLE_RATE / 1000;

        min_silence_samples_ =
            params.min_silence_ms * SAMPLE_RATE / 1000;

        max_speech_samples_ =
            params.max_speech_s * SAMPLE_RATE;

        // The pre-roll buffer must be able to hold the padding plus the
        // full run of speech windows that can accumulate before
        // start_utterance() fires. Since brief inter-word silence no
        // longer resets that run (it's tolerated up to
        // min_silence_samples_), the run can span a gap of nearly that
        // long, plus one extra window of slack for the window that
        // crosses the threshold.
        pre_roll_capacity_ =
            static_cast<size_t>(speech_start_padding_samples_) +
            static_cast<size_t>(min_speech_samples_) +
            static_cast<size_t>(min_silence_samples_) +
            VAD_WINDOW_SAMPLES;

        transcription_worker_.start(&transcriber_);

        return true;
    }

    ~SpeechRecognizer() {
        transcription_worker_.shutdown();
    }

    void process(const float * samples, size_t n_samples) {
        input_buffer_.insert(
            input_buffer_.end(),
            samples,
            samples + n_samples);

        while (input_buffer_.size() >= VAD_WINDOW_SAMPLES) {
            std::array<float, VAD_WINDOW_SAMPLES> window;

            std::copy_n(
                input_buffer_.begin(),
                VAD_WINDOW_SAMPLES,
                window.begin());

            input_buffer_.erase(
                input_buffer_.begin(),
                input_buffer_.begin() + VAD_WINDOW_SAMPLES);

            process_window(window.data());
        }
    }

    void finish() {
        if (input_buffer_.empty()) {
            return;
        }

        std::array<float, VAD_WINDOW_SAMPLES> window{};

        std::copy(
            input_buffer_.begin(),
            input_buffer_.end(),
            window.begin());

        process_window(window.data());

        input_buffer_.clear();

        if (in_speech_) {
            finish_utterance();
        }
    }

private:
    void process_window(const float * window) {
        // Keep a rolling pre-roll buffer of raw audio so that once speech
        // is confirmed we can recover the audio that came before the VAD
        // actually crossed its detection threshold.
        pre_roll_.insert(
            pre_roll_.end(),
            window,
            window + VAD_WINDOW_SAMPLES);

        while (pre_roll_.size() > pre_roll_capacity_) {
            pre_roll_.pop_front();
        }

        if (params_->log_levels && !in_speech_) {
            ambient_log_samples_ += VAD_WINDOW_SAMPLES;

            if (ambient_log_samples_ >= SAMPLE_RATE) {
                ambient_log_samples_ = 0;

                const LevelStats stats =
                    compute_level_stats(pre_roll_.begin(), pre_roll_.end());

                std::fprintf(
                    stderr,
                    "[LEVEL] ambient: rms=%.1f dBFS peak=%.1f dBFS "
                    "clip=%.2f%%\n",
                    stats.rms_dbfs,
                    stats.peak_dbfs,
                    stats.clip_pct);
            }
        }

        const float probability =
            vad_.process(window, VAD_WINDOW_SAMPLES);

        const bool speech =
            vad_.is_speech(probability);

        const size_t window_samples = VAD_WINDOW_SAMPLES;

        if (!in_speech_) {
            if (speech) {
                speech_run_samples_ += window_samples;
                pre_speech_silence_samples_ = 0;

                if (speech_run_samples_ >= min_speech_samples_) {
                    start_utterance();
                }
            } else if (speech_run_samples_ > 0) {
                // Tolerate brief gaps (e.g. between words) without losing
                // the speech run accumulated so far, mirroring the
                // min_silence_ms hangover used to *end* an utterance.
                // Only a sustained silence resets progress.
                pre_speech_silence_samples_ += window_samples;

                if (pre_speech_silence_samples_ >= min_silence_samples_) {
                    speech_run_samples_ = 0;
                    pre_speech_silence_samples_ = 0;
                }
            }

            return;
        }

        // Already inside an utterance.
        utterance_.insert(
            utterance_.end(),
            window,
            window + VAD_WINDOW_SAMPLES);

        if (speech) {
            silence_samples_ = 0;
        } else {
            silence_samples_ += window_samples;

            if (silence_samples_ >= min_silence_samples_) {
                finish_utterance();
                return;
            }
        }

        if (utterance_.size() >= max_speech_samples_) {
            finish_utterance();
        }
    }

    void start_utterance() {
        in_speech_ = true;

        silence_samples_ = 0;
        pre_speech_silence_samples_ = 0;

        utterance_.clear();

        // Recover the speech_run_samples_ of audio that already elapsed
        // before we crossed min_speech_samples_, plus speech_pad_ms of
        // additional lead-in, from the pre-roll buffer.
        const size_t needed =
            static_cast<size_t>(speech_start_padding_samples_) +
            static_cast<size_t>(speech_run_samples_);

        const size_t take = std::min(needed, pre_roll_.size());

        utterance_.insert(
            utterance_.end(),
            pre_roll_.end() - static_cast<std::ptrdiff_t>(take),
            pre_roll_.end());

        // Packets arrive in real time, so "now" is the wall-clock end of
        // the audio received so far; the recovered pre-roll started `take`
        // samples earlier.
        utterance_t_start_utc_ =
            wall_now() - static_cast<double>(take) / SAMPLE_RATE;

        if (g_json_output) {
            char line[96];
            std::snprintf(line, sizeof(line),
                          "{\"type\":\"speech_start\",\"t_utc\":%.3f}",
                          utterance_t_start_utc_);
            emit_json_line(line);
        } else {
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            std::printf("\n[Speech detected]\n");
            std::fflush(stdout);
        }
    }

    void emit_speech_end(double t_end_utc, bool discarded) {
        if (!g_json_output) {
            return;
        }

        char line[160];
        std::snprintf(line, sizeof(line),
                      "{\"type\":\"speech_end\",\"t_utc\":%.3f,"
                      "\"duration_s\":%.3f,\"discarded\":%s}",
                      t_end_utc, t_end_utc - utterance_t_start_utc_,
                      discarded ? "true" : "false");
        emit_json_line(line);
    }

    void finish_utterance() {
        // The endpoint detector only fires after min_silence_ms of
        // silence, so the speech itself ended that long ago.
        const double t_end_utc =
            wall_now() - static_cast<double>(silence_samples_) / SAMPLE_RATE;

        // A speech_start was already announced for this utterance, so a
        // discarded one still gets a speech_end: consumers need to know
        // no text event will follow.
        if (utterance_.empty() ||
            utterance_.size() < static_cast<size_t>(min_speech_samples_)) {
            emit_speech_end(t_end_utc, true);
            reset_state();
            return;
        }

        if (g_json_output) {
            emit_speech_end(t_end_utc, false);
        } else {
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            std::printf("[Speech ended]\n");
            std::fflush(stdout);
        }

        // Remove trailing silence introduced by the endpoint detector.
        const size_t trailing =
            std::min(
                static_cast<size_t>(silence_samples_),
                utterance_.size());

        if (trailing < utterance_.size()) {
            utterance_.resize(utterance_.size() - trailing);
        }

        if (params_->log_levels) {
            const LevelStats stats =
                compute_level_stats(utterance_.begin(), utterance_.end());

            std::fprintf(
                stderr,
                "[LEVEL] utterance: rms=%.1f dBFS peak=%.1f dBFS "
                "clip=%.2f%% (%.2fs)\n",
                stats.rms_dbfs,
                stats.peak_dbfs,
                stats.clip_pct,
                static_cast<double>(utterance_.size()) / SAMPLE_RATE);
        }

        Utterance finished;
        finished.audio = std::move(utterance_);
        finished.t_start_utc = utterance_t_start_utc_;
        finished.t_end_utc = t_end_utc;

        transcription_worker_.enqueue(std::move(finished));

        reset_state();
    }

    void reset_state() {
        utterance_.clear();

        input_buffer_.clear();

        in_speech_ = false;

        speech_run_samples_ = 0;
        silence_samples_ = 0;
        pre_speech_silence_samples_ = 0;

        vad_.reset();
    }

private:
    const Params * params_ = nullptr;

    StreamingVad vad_;
    Transcriber transcriber_;
    TranscriptionWorker transcription_worker_;

    std::vector<float> input_buffer_;
    std::vector<float> utterance_;
    std::deque<float> pre_roll_;
    size_t pre_roll_capacity_ = 0;

    bool in_speech_ = false;
    double utterance_t_start_utc_ = 0.0;

    int speech_run_samples_ = 0;
    int silence_samples_ = 0;
    int pre_speech_silence_samples_ = 0;
    int ambient_log_samples_ = 0;

    int speech_start_padding_samples_ = 0;
    int min_speech_samples_ = 0;
    int min_silence_samples_ = 0;
    int max_speech_samples_ = 0;
};

static void cb_log_disable(enum ggml_log_level, const char *, void *) {}

} // namespace

int main(int argc, char ** argv) {
    Params params;

    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    g_json_output = params.json_output;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (!params.verbose) {
        whisper_log_set(cb_log_disable, nullptr);
    }

    ggml_backend_load_all();

    SpeechRecognizer recognizer;

    if (!recognizer.init(params)) {
        return 1;
    }

    const int sock =
        socket(AF_INET, SOCK_DGRAM, 0);

    if (sock < 0) {
        std::perror("socket");
        return 1;
    }

    int receive_buffer = 1024 * 1024;

    setsockopt(
        sock,
        SOL_SOCKET,
        SO_RCVBUF,
        &receive_buffer,
        sizeof(receive_buffer));

    // std::signal() installs handlers with SA_RESTART, so a blocked
    // recvfrom() is transparently restarted on SIGINT/SIGTERM and the loop
    // never sees g_running go false while no audio is arriving. A short
    // receive timeout makes the loop re-check it.
    timeval receive_timeout{};
    receive_timeout.tv_usec = 250 * 1000;

    setsockopt(
        sock,
        SOL_SOCKET,
        SO_RCVTIMEO,
        &receive_timeout,
        sizeof(receive_timeout));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(params.port);

    if (bind(
            sock,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) < 0) {

        std::perror("bind");
        close(sock);
        return 1;
    }

    FILE * banner = banner_stream();

    std::fprintf(
        banner,
        "Listening for UDP audio on 0.0.0.0:%d\n",
        params.port);

    std::fprintf(
        banner,
        "VAD threshold: %.2f\n",
        params.vad_threshold);

    std::fprintf(
        banner,
        "Models:\n"
        "  Whisper: %s\n"
        "  VAD:     %s\n\n",
        params.whisper_model.c_str(),
        params.vad_model.c_str());

    if (g_json_output) {
        char line[64];
        std::snprintf(line, sizeof(line),
                      "{\"type\":\"ready\",\"port\":%d}", params.port);
        emit_json_line(line);
    } else {
        std::printf("[Waiting for speech]\n");
        std::fflush(stdout);
    }

    std::vector<uint8_t> packet_buffer(65536);

    uint32_t expected_sequence = 0;
    bool have_sequence = false;

    while (g_running) {
        const ssize_t received =
            recvfrom(
                sock,
                packet_buffer.data(),
                packet_buffer.size(),
                0,
                nullptr,
                nullptr);

        if (received < 0) {
            if (!g_running) {
                break;
            }

            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }

            std::perror("recvfrom");
            break;
        }

        AudioPacket packet;

        if (!parse_packet(
                packet_buffer.data(),
                static_cast<size_t>(received),
                packet)) {

            continue;
        }

        if (have_sequence &&
            packet.sequence != expected_sequence) {

            const uint32_t gap =
                packet.sequence - expected_sequence;

            std::fprintf(
                stderr,
                "\n[UDP] packet gap: expected %u, got %u "
                "(gap=%u)\n",
                expected_sequence,
                packet.sequence,
                gap);
        }

        expected_sequence =
            packet.sequence + 1;

        have_sequence = true;

        // Convert int16 PCM to [-1, +1] float PCM.
        std::vector<float> pcm(
            packet.samples.size());

        for (size_t i = 0; i < packet.samples.size(); ++i) {
            pcm[i] =
                static_cast<float>(packet.samples[i]) /
                32768.0f;
        }

        recognizer.process(
            pcm.data(),
            pcm.size());
    }

    recognizer.finish();

    close(sock);

    if (g_json_output) {
        emit_json_line("{\"type\":\"stopped\"}");
    } else {
        std::printf("\nStopped.\n");
    }

    return 0;
}

