# safemlx-codec

Neural audio codec components for `safemlx`.

This crate keeps audio codec implementations separate from `safemlx-lm`, while
still using the same MLX runtime and tensor types. Realtime speech language
models can stay codec-token based, and applications can opt into codec support
only when they need PCM encode/decode.

## Mimi

`safemlx_codec::mimi` implements the safemlx-native Mimi encoder, residual
vector quantizer, and decoder used by Moshi-family models:

- released Mimi v0.1 config metadata
- safetensors checkpoint loading
- active codebook selection, for example 8 or 16 codebooks from a 32-codebook checkpoint
- `AudioTokenizer::encode`: `[batch, 1, samples] -> [batch, codebooks, frames]`
- `encode_latent`: `[batch, 512, frames] -> [batch, codebooks, frames]`
- `decode_latent`: `[batch, codebooks, frames] -> [batch, 512, frames]`
- `AudioTokenizer::decode`: `[batch, codebooks, frames] -> [batch, 1, samples]`
- `Mimi::decode_step`: stateful one-frame token decode for realtime playback

Audio device I/O is intentionally out of scope for this crate. Applications can
pair these tensor APIs with their own capture/playback stack.

```rust,ignore
use safemlx_codec::AudioTokenizer;
use safemlx_codec::mimi::Mimi;

let mut mimi = Mimi::load("/path/to/tokenizer.safetensors", Some(8), stream)?;
let tokens = mimi.encode(&pcm, stream)?;
let pcm = mimi.decode(&codec_tokens, stream)?;
let latents = mimi.decode_latent(&codec_tokens, stream)?;
let recoded_tokens = mimi.encode_latent(&latents, stream)?;
```

## PersonaPlex quantization evaluation

The `personaplex_quantization_eval` example compares a dense and quantized
PersonaPlex checkpoint on the same real audio. It reports realtime deadline
performance, teacher-forced text/audio distribution drift, free-running token
divergence, and writes a blinded pair of decoded WAV responses.

Prepare mono 24 kHz raw `f32le` PCM with a tool such as FFmpeg, then run:

```sh
ffmpeg -i input.wav -f f32le -ac 1 -ar 24000 /tmp/input.f32le
ffmpeg -i voice-prompt.wav -f f32le -ac 1 -ar 24000 /tmp/voice-prompt.f32le

cargo run --release -p safemlx-codec \
  --example personaplex_quantization_eval -- \
  /path/to/personaplex-dense \
  /path/to/personaplex-quantized \
  /path/to/tokenizer.safetensors \
  /path/to/tokenizer_spm_32k_3.model \
  /tmp/voice-prompt.f32le \
  /tmp/input.f32le \
  /tmp/personaplex-eval \
  128
```

An optional argument after the frame limit overrides the default assistant text
prompt, and a final optional integer sets the sampling seed. PersonaPlex
requires both voice and text conditioning; omitting them can produce non-speech
codec noise, so the evaluator makes both explicit. The released 7B v1
checkpoint is intended for English conversational input.

The optional frame argument limits the number of 80 ms frames. The output
directory contains `metrics.json`, `input.wav`, blinded `sample_a.wav` and
`sample_b.wav`, a reusable `listening_manifest.json`, and a separate
`answer_key.json`. It also writes streaming and whole-clip codec sanity checks
as `input_codec_roundtrip.wav` and `input_codec_roundtrip_offline.wav`, plus
`token_diagnostics.json` for debugging invalid audio.
Generated and input-conditioned audio heads are summarized separately. The
quantized model is forced onto the dense model's token history only for
distribution metrics; the WAV pair comes from independent free-running
generations. Listening samples use PersonaPlex's production defaults (text
temperature 0.7/top-k 25 and audio temperature 0.8/top-k 250) with a fixed
explicit PRNG state. The evaluator warns when either the input or generated
speech is still active at the selected frame boundary.

### Multi-case regression suite

The suite runner expands each input across one or more sampling seeds, invokes
the evaluator in a fresh process per trial, aggregates performance and
teacher-forced drift, and creates one blind listening manifest and ratings file:

```json
{
  "format_version": 1,
  "dense_model": "/path/to/personaplex-dense",
  "quantized_model": "/path/to/personaplex-q4",
  "mimi": "/path/to/tokenizer.safetensors",
  "text_tokenizer": "/path/to/tokenizer_spm_32k_3.model",
  "voice_prompt": "/path/to/voice-prompt.f32le",
  "sampling_seeds": [20260713, 20260714],
  "cases": [
    {
      "id": "procedural_question",
      "category": "procedural",
      "input": "/path/to/input-mono-24khz.f32le"
    }
  ]
}
```

```sh
python safemlx-codec/scripts/personaplex_quantization_suite.py run \
  suite.json /tmp/personaplex-quantization-suite
```

The runner rejects silent inputs and byte-identical cases by default, catching
failed corpus generation before an expensive Metal run. Each case may override
`sampling_seeds`, `frames`, and `text_prompt`. Set `allow_silent_input` only for
an intentional silence diagnostic and `allow_duplicate_input` only when the
duplication is deliberate.

Listen using `listening_manifest.json`, then fill the generated
`human_ratings.json` without opening per-case answer keys. Unblind and aggregate
the completed ratings with:

```sh
python safemlx-codec/scripts/personaplex_quantization_suite.py summarize \
  /tmp/personaplex-quantization-suite \
  /tmp/personaplex-quantization-suite/human_ratings.json
```

### Dense safemlx versus upstream PyTorch

`personaplex_quantization_eval` also exports the exact voice-prompt, text, and
user-audio token streams plus dense greedy and sampled traces in
`token_diagnostics.json`. Feed that file to the upstream reference runner to
remove tokenizer and encoder differences from a backend comparison:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHONPATH=/path/to/upstream/moshi:/path/to/python/dependencies \
python safemlx-codec/scripts/personaplex_pytorch_backend_reference.py \
  --moshi-source /path/to/upstream/moshi \
  --model /path/to/personaplex/model.safetensors \
  --mimi /path/to/tokenizer.safetensors \
  --tokenizer /path/to/tokenizer_spm_32k_3.model \
  --safemlx-eval-dir /tmp/personaplex-eval \
  --output-dir /tmp/personaplex-backend-comparison \
  --device mps
```

The runner performs a short deterministic greedy token-parity trace and a
full production-sampled listening run. It writes a newly randomized blind WAV
pair, `metrics.json`, and `answer_key.json`. Sampling parameters and the seed
match, but PyTorch and MLX use their native RNG algorithms, so stochastic token
draws are not expected to be identical. PyTorch MPS may fall back to CPU for
KV-cache updates; its timing is reported for diagnostics rather than as a fair
backend performance benchmark.
