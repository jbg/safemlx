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
