use std::time::{Duration, Instant};

use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp},
    transforms::eval,
    Array, Stream,
};

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::{
        gemma4::{sample, Model as Gemma4Model, ModelInput},
        gemma4_assistant::Gemma4AssistantDraftModel,
    },
};

#[derive(Debug, Clone, Default)]
pub struct MtpStats {
    pub target_tokens: usize,
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
    pub rounds: usize,
    pub accept_lens: Vec<usize>,
    pub elapsed: Duration,
}

impl MtpStats {
    pub fn accept_rate(&self) -> f64 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.draft_tokens as f64
        }
    }
}

pub fn generate_gemma4_mtp(
    target: &mut Gemma4Model,
    assistant: &mut Gemma4AssistantDraftModel,
    prompt_tokens: &Array,
    eos_token_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    stream: &Stream,
) -> Result<(Vec<u32>, MtpStats), Error> {
    let start = Instant::now();
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let mut generated = Vec::new();
    let mut stats = MtpStats::default();

    let prompt_len = prompt_tokens.shape()[1];
    if prompt_len > 1 {
        let prefix = prompt_tokens.try_index_device((.., ..prompt_len - 1), stream)?;
        target.forward_with_state(
            ModelInput {
                inputs: &prefix,
                mask: None,
                cache: &mut cache,
            },
            stream,
        )?;
    }

    let last = prompt_tokens.try_index_device((.., prompt_len - 1..), stream)?;
    let first_out = target.forward_with_state(
        ModelInput {
            inputs: &last,
            mask: None,
            cache: &mut cache,
        },
        stream,
    )?;
    let first_logits = first_out.logits.try_index_device((.., -1, ..), stream)?;
    let first_token = sample(&first_logits, temp, stream)?;
    eval([&first_token])?;
    let mut bonus = first_token.item::<u32>(&stream);
    stats.target_tokens += 1;
    if eos_token_ids.contains(&bonus) || max_tokens == 0 {
        stats.elapsed = start.elapsed();
        return Ok((generated, stats));
    }

    generated.push(bonus);
    let mut hidden = first_out.hidden.try_index_device((.., -1.., ..), stream)?;
    let mut shared_kv = first_out.shared_kv_states;
    assistant.reset();
    let mut emitted = 1usize;

    while emitted < max_tokens {
        let block = assistant.block_size().min(max_tokens - emitted + 1);
        if block <= 1 {
            break;
        }

        let kv_offset = cache
            .iter()
            .flatten()
            .next()
            .map(KeyValueCache::offset)
            .unwrap_or(0);
        assistant.set_shared_kv(shared_kv.clone(), kv_offset);
        let draft = assistant.draft_block(target, bonus, &hidden, block, temp, stream)?;
        eval([&draft])?;

        let verify_input = concatenate_axis(
            &[Array::from_slice(&[bonus], &[1, 1]), draft.clone()],
            1,
            stream,
        )?;
        let verify_out = target.forward_with_state(
            ModelInput {
                inputs: &verify_input,
                mask: None,
                cache: &mut cache,
            },
            stream,
        )?;
        let target_tokens = sample(&verify_out.logits, temp, stream)?;
        eval([&target_tokens])?;
        stats.target_tokens += verify_input.shape()[1] as usize;

        let draft_ids = array_to_ids(&draft, stream);
        let target_ids = array_to_ids(&target_tokens, stream);
        stats.draft_tokens += draft_ids.len();

        let mut accepted = 0usize;
        while accepted < draft_ids.len()
            && accepted < target_ids.len()
            && draft_ids[accepted] == target_ids[accepted]
            && emitted + accepted < max_tokens
        {
            accepted += 1;
        }
        stats.accepted_tokens += accepted;
        stats.accept_lens.push(accepted);
        stats.rounds += 1;

        let mut new_tokens = draft_ids[..accepted].to_vec();
        if emitted + new_tokens.len() < max_tokens {
            if let Some(id) = target_ids.get(accepted).copied() {
                new_tokens.push(id);
            }
        }

        for id in new_tokens.iter().copied() {
            if eos_token_ids.contains(&id) {
                emitted = max_tokens;
                break;
            }
            generated.push(id);
            emitted += 1;
        }
        if emitted >= max_tokens {
            break;
        }

        let block_size = draft_ids.len() + 1;
        if accepted < draft_ids.len() {
            target.rollback_speculative_cache(&mut cache, accepted, block_size, stream)?;
        }

        hidden = verify_out
            .hidden
            .try_index_device((.., accepted as i32..accepted as i32 + 1, ..), stream)?;
        shared_kv = verify_out.shared_kv_states;
        if let Some(last) = generated.last().copied() {
            bonus = last;
        }
    }

    stats.elapsed = start.elapsed();
    Ok((generated, stats))
}

fn array_to_ids(array: &Array, stream: &Stream) -> Vec<u32> {
    array
        .flatten(None, None, stream)
        .expect("flatten token array")
        .into_evaluated()
        .expect("evaluate token array")
        .as_slice::<u32>()
        .to_vec()
}
