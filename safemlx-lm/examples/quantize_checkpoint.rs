use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::quantization::{
    AffineQuantization, CheckpointQuantizationOptions, WeightQuantization,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Affine,
    Mxfp4,
}

#[derive(Debug, Parser)]
#[command(about = "Quantize a safetensors model directory using MLX affine or MXFP4 packing")]
struct Args {
    /// Source Hugging Face/MLX model directory.
    source: PathBuf,
    /// New output directory. It must not already exist.
    output: PathBuf,
    /// Quantized weight encoding.
    #[arg(long, value_enum, default_value_t = Mode::Affine)]
    mode: Mode,
    /// Number of values sharing each scale (and affine bias).
    #[arg(long)]
    group_size: Option<i32>,
    /// Packed bits per weight.
    #[arg(long)]
    bits: Option<i32>,
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
    let quantization = match args.mode {
        Mode::Affine => WeightQuantization::Affine(AffineQuantization::new(
            args.group_size.unwrap_or(64),
            args.bits.unwrap_or(4),
        )?),
        Mode::Mxfp4 => {
            let group_size = args.group_size.unwrap_or(32);
            let bits = args.bits.unwrap_or(4);
            if group_size != 32 || bits != 4 {
                return Err(format!(
                    "MXFP4 requires --group-size 32 and --bits 4, got {group_size}/{bits}"
                )
                .into());
            }
            WeightQuantization::MxFp4
        }
    };
    let options = CheckpointQuantizationOptions {
        quantization,
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
