use std::{error::Error, path::PathBuf};

use safemlx::{
    ops::{indexing::TryIndexOp, GgufMetadataValue},
    Array, Device, DeviceType, Stream,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or(
            "usage: cargo run -p safemlx --example gguf_info -- <model.gguf> [max-tensors] [sample-tensor]",
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
    let (tensors, metadata) = Array::load_gguf_with_metadata(&path, &stream)?;

    println!("file: {}", path.display());
    if let Some(GgufMetadataValue::String(architecture)) = metadata.get("general.architecture") {
        println!("architecture: {architecture}");
    }
    println!("metadata entries: {}", metadata.len());
    println!("tensors: {}", tensors.len());

    let mut tensor_names = tensors.keys().collect::<Vec<_>>();
    tensor_names.sort_unstable();
    for name in tensor_names.into_iter().take(max_tensors) {
        let tensor = &tensors[name];
        println!("{name}: {:?} {:?}", tensor.shape(), tensor.dtype());
    }

    if let Some(name) = sample_tensor {
        let tensor = tensors
            .get(&name)
            .ok_or_else(|| format!("tensor {name:?} was not found"))?;
        let sample_source = if tensor.ndim() > 1 {
            tensor.try_index_device(0, &stream)?
        } else {
            tensor.clone()
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
