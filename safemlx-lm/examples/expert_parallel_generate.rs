//! Minimal sparse-cache Ring expert-parallel generation probe for supported MoE models.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    expert_cache::ExpertCacheLoadOptions, expert_parallel::load_expert_parallel_model_with_options,
    models::ModelLoadOptions, sampler::DefaultSampler, DeviceAssignment, ParallelTopology,
    WeightResidency,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: expert_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &group,
        1,
        1,
        group.size(),
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let options = ModelLoadOptions::with_parallel(topology).with_weight_residency(
        WeightResidency::SparseExpertCache(ExpertCacheLoadOptions::default()),
    );
    let mut model =
        load_expert_parallel_model_with_options(&model_dir, options, &stream, &weights_stream)?;
    if group.rank() == 0 {
        eprintln!(
            "EP={} assignment={:?} local experts by rank 0={:?}",
            model.info().expert_parallel_size,
            model.info().assignment.policy(),
            model.info().assignment.local_global_expert_ids(),
        );
    }

    let mut cache = model.new_cache();
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let mut logits = model.prefill(&prompt, &mut cache, &group, &stream)?;
    let mut sampler = DefaultSampler;
    for _ in 0..8 {
        let synchronized = model.sample_and_synchronize(
            &logits,
            &mut sampler,
            0.0,
            None,
            false,
            0,
            &group,
            &stream,
        )?;
        if group.rank() == 0 {
            eprintln!(
                "sampled token {:?}; routing {:?}",
                synchronized.token.evaluated()?.as_slice::<u32>(),
                model.latest_routing_statistics(),
            );
        }
        if synchronized.finished {
            break;
        }
        logits = model.decode(&synchronized.token, &mut cache, &group, &stream)?;
    }
    Ok(())
}
