//! Minimal two-or-more-process tensor-parallel generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    sampler::DefaultSampler, tensor_parallel::load_tensor_parallel_model, DeviceAssignment,
    ParallelTopology,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: tensor_parallel_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &group,
        group.size(),
        1,
        1,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let mut model = load_tensor_parallel_model(&model_dir, topology, &stream, &weights_stream)?;
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
                "sampled token {:?}",
                synchronized.token.evaluated()?.as_slice::<u32>()
            );
        }
        if synchronized.finished {
            break;
        }
        logits = model.decode(&synchronized.token, &mut cache, &group, &stream)?;
    }
    Ok(())
}
