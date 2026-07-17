//! Minimal two-or-more-process serial pipeline generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    pipeline::{load_pipeline_model, PipelineStep},
    sampler::DefaultSampler,
    DeviceAssignment, ParallelTopology,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: pipeline_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &group,
        1,
        group.size(),
        1,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let mut model = load_pipeline_model(&model_dir, topology, &stream, &weights_stream)?;
    let mut cache = model.new_cache();
    let prompt = safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let mut logits = model.forward_pipeline(
        model.stage_info().is_first.then_some(&prompt),
        PipelineStep::new(1, 3)?,
        None,
        &mut cache,
        &group,
        &stream,
    )?;
    let mut sampler = DefaultSampler;
    for _ in 0..8 {
        let synchronized = model.sample_and_synchronize(
            logits.as_ref(),
            PipelineStep::new(1, 1)?,
            &mut sampler,
            0.0,
            None,
            false,
            &group,
            &stream,
        )?;
        if model.stage_info().is_last {
            eprintln!(
                "sampled token {:?}",
                synchronized.token.evaluated()?.as_slice::<u32>()
            );
        }
        if synchronized.finished {
            break;
        }
        logits = model.forward_pipeline(
            model.stage_info().is_first.then_some(&synchronized.token),
            PipelineStep::new(1, 1)?,
            None,
            &mut cache,
            &group,
            &stream,
        )?;
    }
    Ok(())
}
