use std::{error::Error, path::PathBuf};

use safemlx::{
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    Device, DeviceType, Stream,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or(
            "usage: cargo run -p safemlx --example gguf_info -- <model-or-first-shard.gguf> [max-tensors] [sample-tensor]",
        )?;
    let max_tensors = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(20);
    let sample_tensor = args
        .next()
        .map(|value| value.to_string_lossy().into_owned());
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let checkpoint = GgufCheckpoint::open(&path)?;
    let metadata = checkpoint.metadata();

    println!("file: {}", path.display());
    if let Some(GgufMetadataValue::String(architecture)) = metadata.get("general.architecture") {
        println!("architecture: {architecture}");
    }
    println!("metadata entries: {}", metadata.len());
    println!(
        "physical tensors: {}",
        checkpoint.catalog().physical_tensor_count()
    );

    let mut layouts = checkpoint.catalog().logical_outputs().collect::<Vec<_>>();
    layouts.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    for layout in layouts.into_iter().take(max_tensors) {
        println!("{}: {:?} {:?}", layout.name, layout.shape, layout.dtype);
    }

    if let Some(name) = sample_tensor {
        let mut tensor = None;
        for group in checkpoint.converted_tensors() {
            if let Some((_, array)) = group?
                .into_arrays()
                .into_iter()
                .find(|(logical_name, _)| logical_name == &name)
            {
                tensor = Some(array);
                break;
            }
        }
        let tensor = tensor.ok_or_else(|| format!("tensor {name:?} was not found"))?;
        let sample_source = if tensor.ndim() > 1 {
            tensor.try_index_device(0, &stream)?
        } else {
            tensor
        };
        let flat = sample_source.flatten(None, None, &stream)?;
        let mut sample = Vec::new();
        for index in 0..flat.size().min(8) {
            sample.push(
                flat.try_index_device(index as i32, &stream)?
                    .try_item::<f32>(&stream)?,
            );
        }
        println!("{name} sample: {sample:?}");
    }

    Ok(())
}
