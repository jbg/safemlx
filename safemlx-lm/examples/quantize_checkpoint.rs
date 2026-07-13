use std::path::PathBuf;

use clap::Parser;
use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::quantization::{AffineQuantization, CheckpointQuantizationOptions};

#[derive(Debug, Parser)]
#[command(about = "Quantize a safetensors model directory using MLX affine packing")]
struct Args {
    /// Source Hugging Face/MLX model directory.
    source: PathBuf,
    /// New output directory. It must not already exist.
    output: PathBuf,
    /// Number of values sharing each scale and bias.
    #[arg(long, default_value_t = 64)]
    group_size: i32,
    /// Packed bits per weight.
    #[arg(long, default_value_t = 4)]
    bits: i32,
    /// Approximate maximum output shard size in MiB.
    #[arg(long, default_value_t = 512)]
    shard_size_mib: usize,
    /// Only quantize tensor names containing this string (repeatable).
    #[arg(long)]
    include: Vec<String>,
    /// Do not quantize tensor names containing this string (repeatable).
    #[arg(long)]
    exclude: Vec<String>,
    /// Skip matrices with fewer than this many elements.
    #[arg(long, default_value_t = 0)]
    minimum_elements: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let options = CheckpointQuantizationOptions {
        quantization: AffineQuantization::new(args.group_size, args.bits)?,
        shard_size_bytes: args.shard_size_mib * 1024 * 1024,
        include: args.include,
        exclude: args.exclude,
        minimum_elements: args.minimum_elements,
    };
    let report =
        safemlx_lm::quantization::quantize_checkpoint(args.source, args.output, &options, stream)?;
    stream.synchronize()?;
    println!("quantized_tensors={}", report.quantized_tensors);
    println!("copied_tensors={}", report.copied_tensors);
    println!("shards={}", report.shards);
    println!("total_size_bytes={}", report.total_size);
    Ok(())
}
