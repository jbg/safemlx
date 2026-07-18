//! Opt-in expert-parallel performance and parity probe.
//!
//! This compares a complete single-rank model, replicated-input expert
//! parallelism, and the synthetic sharded-token all-to-all fallback. Phase
//! profiling inserts evaluation boundaries and is intentionally not enabled by
//! normal inference.

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum};
use safemlx::{
    distributed::{self, Backend, Group},
    ops::indexing::TryIndexOp,
    transforms::eval,
    Array, Device, DeviceType, Stream,
};
use safemlx_lm::{
    expert_parallel::{
        dispatch_sharded, load_expert_parallel_model, profile_expert_parallel_timings,
        ExpertAssignment, LocalExpertBank, RoutingStatistics, ShardedRouteBlocks,
    },
    models::{
        self,
        input::{InputPart, ModelInput},
        Model, ModelCache,
    },
    DeviceAssignment, ParallelTopology,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchDevice {
    Cpu,
    Gpu,
}

impl BenchDevice {
    fn device_type(self) -> DeviceType {
        match self {
            Self::Cpu => DeviceType::Cpu,
            Self::Gpu => DeviceType::Gpu,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Compare complete, replicated-EP, and synthetic sharded-EP inference")]
struct Args {
    /// Hugging Face model directory containing config and safetensors.
    model_dir: PathBuf,
    /// Distributed backend: any, ring, mpi, jaccl, or nccl.
    #[arg(long, default_value = "any")]
    backend: Backend,
    /// Execution device. Ring collectives currently require CPU.
    #[arg(long, value_enum, default_value_t = BenchDevice::Gpu)]
    device: BenchDevice,
    /// Number of untimed repetitions for each model path.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Number of measured repetitions.
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    /// Synthetic prompt length in tokens.
    #[arg(long, default_value_t = 32)]
    prompt_tokens: usize,
    /// Fixed-token cache decode steps per repetition.
    #[arg(long, default_value_t = 8)]
    decode_tokens: usize,
    /// Hidden width for the synthetic sharded-token exchange.
    #[arg(long, default_value_t = 1024)]
    synthetic_hidden: usize,
}

#[derive(Default)]
struct PhaseResult {
    seconds: f64,
    tokens: usize,
    statistics: RoutingStatistics,
    output: Option<Array>,
}

struct ModelResults {
    prefill: PhaseResult,
    decode: PhaseResult,
    peak_memory: usize,
}

struct IdentityBank;

impl LocalExpertBank for IdentityBank {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        _local_expert_ids: &Array,
        _stream: &Stream,
    ) -> Result<Array, safemlx_lm::error::Error> {
        Ok(hidden.clone())
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.iterations > 0, "--iterations must be positive");
    anyhow::ensure!(args.prompt_tokens > 0, "--prompt-tokens must be positive");
    let group = distributed::init(true, args.backend)?;
    anyhow::ensure!(
        group.size() > 1,
        "expert-parallel benchmark requires at least two ranks"
    );

    let local_rank = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(group.rank());
    let device_index = match args.device {
        BenchDevice::Cpu => 0,
        BenchDevice::Gpu => local_rank,
    };
    let device = Device::new(args.device.device_type(), device_index.try_into()?);
    let stream = Stream::new_with_device(&device);
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let prompt_values = vec![1u32; args.prompt_tokens];
    let prompt =
        Array::from_slice(&prompt_values, &[1, args.prompt_tokens as i32]).copy(&stream)?;
    let decode_token = Array::from_slice(&[1u32], &[1, 1]).copy(&stream)?;

    // The complete baseline runs only on rank zero. Every rank meets at the
    // following scalar collective before loading its local EP shard.
    let baseline = if group.rank() == 0 {
        safemlx::memory::reset_peak_memory()?;
        let mut model = models::load_model(&args.model_dir, &stream, &weights_stream)?;
        let results = benchmark_complete_model(
            &mut model,
            &prompt,
            &decode_token,
            args.warmup,
            args.iterations,
            args.decode_tokens,
            &stream,
        )?;
        Some(results)
    } else {
        None
    };
    synchronize_ranks(&group, &stream)?;

    safemlx::memory::reset_peak_memory()?;
    let topology = ParallelTopology::from_group(
        &group,
        1,
        1,
        group.size(),
        DeviceAssignment::new(args.device.device_type(), device_index),
    )?;
    let mut ep_model =
        load_expert_parallel_model(&args.model_dir, topology, &stream, &weights_stream)?;
    let _profiling = profile_expert_parallel_timings();
    let ep = benchmark_replicated_ep(
        &mut ep_model,
        &prompt,
        &decode_token,
        args.warmup,
        args.iterations,
        args.decode_tokens,
        &group,
        &stream,
    )?;
    let ep_peak_memory = safemlx::memory::peak_memory()?;

    let synthetic = benchmark_synthetic_sharded(
        args.warmup,
        args.iterations,
        args.synthetic_hidden,
        &group,
        &stream,
    )?;

    let prefill_routes = route_range(ep.prefill.statistics.local_routes, &group, &stream)?;
    let decode_routes = route_range(ep.decode.statistics.local_routes, &group, &stream)?;
    let sharded_routes = route_range(synthetic.statistics.local_routes, &group, &stream)?;

    if group.rank() == 0 {
        println!("strategy,phase,iterations,tokens,wall_ms,tokens_per_s,router_ms,compaction_ms,exchange_ms,expert_ms,reduction_ms,shared_ms,total_moe_ms,model_ms,local_routes_min_total,local_routes_max_total,imbalance,padding_routes,exchanged_bytes,synchronizations,sync_ms");
        let baseline = baseline.expect("rank zero created the complete-model baseline");
        print_phase(
            "complete",
            "prefill",
            args.iterations,
            &baseline.prefill,
            (0, 0),
        );
        print_phase(
            "complete",
            "decode",
            args.iterations,
            &baseline.decode,
            (0, 0),
        );
        print_phase(
            "replicated_ep",
            "prefill",
            args.iterations,
            &ep.prefill,
            prefill_routes,
        );
        print_phase(
            "replicated_ep",
            "decode",
            args.iterations,
            &ep.decode,
            decode_routes,
        );
        print_phase(
            "sharded_ep_synthetic",
            "exchange",
            args.iterations,
            &synthetic,
            sharded_routes,
        );

        let prefill_error = max_abs_difference(
            baseline
                .prefill
                .output
                .as_ref()
                .expect("baseline prefill output"),
            ep.prefill.output.as_ref().expect("EP prefill output"),
            &stream,
        )?;
        let decode_error = max_abs_difference(
            baseline
                .decode
                .output
                .as_ref()
                .expect("baseline decode output"),
            ep.decode.output.as_ref().expect("EP decode output"),
            &stream,
        )?;
        println!("parity,prefill_max_abs={prefill_error:.8},decode_max_abs={decode_error:.8}");
        println!(
            "memory,complete_peak_bytes={},ep_peak_bytes={},ep_local_parameter_bytes={},ep_routed_expert_bytes={},ep_replicated_parameter_bytes={}",
            baseline.peak_memory,
            ep_peak_memory,
            ep_model.info().local_parameter_bytes,
            ep_model.info().routed_expert_bytes,
            ep_model.info().replicated_parameter_bytes,
        );
        println!("note,profiling inserts device evaluation boundaries; compare trends and phase costs, not unprofiled production throughput");
    }
    Ok(())
}

fn forward_complete(
    model: &mut Model,
    tokens: &Array,
    cache: &mut ModelCache,
    stream: &Stream,
) -> anyhow::Result<Array> {
    let parts = [InputPart::text_token_ids(tokens)];
    Ok(model.prefill_input_with_cache(ModelInput::new(&parts), cache, stream)?)
}

fn benchmark_complete_model(
    model: &mut Model,
    prompt: &Array,
    decode_token: &Array,
    warmup: usize,
    iterations: usize,
    decode_steps: usize,
    stream: &Stream,
) -> anyhow::Result<ModelResults> {
    for _ in 0..warmup {
        let mut cache = model.new_cache();
        eval([&forward_complete(model, prompt, &mut cache, stream)?])?;
    }
    let mut prefill = PhaseResult::default();
    for _ in 0..iterations {
        let mut cache = model.new_cache();
        let started = Instant::now();
        let output = forward_complete(model, prompt, &mut cache, stream)?;
        eval([&output])?;
        prefill.seconds += started.elapsed().as_secs_f64();
        prefill.output = Some(last_logits(&output, stream)?);
    }
    prefill.tokens = prompt.dim(1) as usize * iterations;

    let mut decode = PhaseResult::default();
    for _ in 0..iterations {
        let mut cache = model.new_cache();
        eval([&forward_complete(model, prompt, &mut cache, stream)?])?;
        let started = Instant::now();
        for _ in 0..decode_steps {
            let output = forward_complete(model, decode_token, &mut cache, stream)?;
            eval([&output])?;
            decode.output = Some(last_logits(&output, stream)?);
        }
        decode.seconds += started.elapsed().as_secs_f64();
    }
    decode.tokens = decode_steps * iterations;
    Ok(ModelResults {
        prefill,
        decode,
        peak_memory: safemlx::memory::peak_memory()?,
    })
}

#[allow(clippy::too_many_arguments)]
fn benchmark_replicated_ep(
    model: &mut safemlx_lm::expert_parallel::ExpertParallelModel,
    prompt: &Array,
    decode_token: &Array,
    warmup: usize,
    iterations: usize,
    decode_steps: usize,
    group: &Group,
    stream: &Stream,
) -> anyhow::Result<ModelResults> {
    for _ in 0..warmup {
        let mut cache = model.new_cache();
        eval([&model.prefill(prompt, &mut cache, group, stream)?])?;
    }
    let mut prefill = PhaseResult::default();
    for _ in 0..iterations {
        let mut cache = model.new_cache();
        let started = Instant::now();
        let output = model.prefill(prompt, &mut cache, group, stream)?;
        eval([&output])?;
        prefill.seconds += started.elapsed().as_secs_f64();
        prefill
            .statistics
            .accumulate(model.latest_routing_statistics());
        prefill.output = Some(last_logits(&output, stream)?);
    }
    prefill.tokens = prompt.dim(1) as usize * iterations;

    let mut decode = PhaseResult::default();
    for _ in 0..iterations {
        let mut cache = model.new_cache();
        eval([&model.prefill(prompt, &mut cache, group, stream)?])?;
        let started = Instant::now();
        for _ in 0..decode_steps {
            let output = model.decode(decode_token, &mut cache, group, stream)?;
            eval([&output])?;
            decode
                .statistics
                .accumulate(model.latest_routing_statistics());
            decode.output = Some(last_logits(&output, stream)?);
        }
        decode.seconds += started.elapsed().as_secs_f64();
    }
    decode.tokens = decode_steps * iterations;
    Ok(ModelResults {
        prefill,
        decode,
        peak_memory: 0,
    })
}

fn benchmark_synthetic_sharded(
    warmup: usize,
    iterations: usize,
    hidden_size: usize,
    group: &Group,
    stream: &Stream,
) -> anyhow::Result<PhaseResult> {
    let assignment = ExpertAssignment::balanced(group.size(), group.size(), group.rank())?;
    let mut bank = IdentityBank;
    for _ in 0..warmup {
        let returned = dispatch_sharded(
            synthetic_blocks(hidden_size, group, stream)?,
            &assignment,
            &mut bank,
            group,
            stream,
        )?;
        eval([&returned.output])?;
    }
    let mut result = PhaseResult::default();
    for _ in 0..iterations {
        let started = Instant::now();
        let returned = dispatch_sharded(
            synthetic_blocks(hidden_size, group, stream)?,
            &assignment,
            &mut bank,
            group,
            stream,
        )?;
        eval([&returned.output])?;
        result.seconds += started.elapsed().as_secs_f64();
        result.statistics.accumulate(&returned.statistics);
        result.output = Some(returned.output);
    }
    result.tokens = iterations;
    let expected_value = ((group.rank() + 1) * (group.rank() + 1)) as f32 / group.size() as f32;
    let expected = Array::from_slice(&vec![expected_value; hidden_size], &[1, hidden_size as i32])
        .copy(stream)?;
    let error = max_abs_difference(
        result
            .output
            .as_ref()
            .expect("positive iterations produced a synthetic output"),
        &expected,
        stream,
    )?;
    anyhow::ensure!(
        error <= 1e-5,
        "synthetic sharded dispatch identity check failed on rank {}: max abs error {error}",
        group.rank()
    );
    Ok(result)
}

fn synthetic_blocks(
    hidden_size: usize,
    group: &Group,
    stream: &Stream,
) -> anyhow::Result<ShardedRouteBlocks> {
    let mut hidden = Vec::with_capacity(group.size());
    let mut global_expert_ids = Vec::with_capacity(group.size());
    let mut original_route_indices = Vec::with_capacity(group.size());
    let mut weights = Vec::with_capacity(group.size());
    for destination in 0..group.size() {
        let rows = usize::from(destination <= group.rank());
        hidden.push(
            Array::from_slice(
                &vec![group.rank() as f32 + 1.0; rows * hidden_size],
                &[rows as i32, hidden_size as i32],
            )
            .copy(stream)?,
        );
        let expert_ids = if rows == 0 {
            Vec::new()
        } else {
            vec![destination as i32]
        };
        global_expert_ids.push(Array::from_slice(&expert_ids, &[rows as i32]).copy(stream)?);
        original_route_indices.push(Array::from_slice(&expert_ids, &[rows as i32]).copy(stream)?);
        let route_weights = if rows == 0 {
            Vec::new()
        } else {
            vec![1.0f32 / group.size() as f32]
        };
        weights.push(Array::from_slice(&route_weights, &[rows as i32]).copy(stream)?);
    }
    Ok(ShardedRouteBlocks {
        hidden,
        global_expert_ids,
        original_route_indices,
        weights,
        top_k: group.size() as i32,
        source_tokens: 1,
    })
}

fn synchronize_ranks(group: &Group, stream: &Stream) -> anyhow::Result<()> {
    let local = Array::from_slice(&[1i32], &[1]).copy(stream)?;
    let synchronized = distributed::all_sum(&local, group, stream)?;
    eval([&synchronized])?;
    Ok(())
}

fn route_range(local: usize, group: &Group, stream: &Stream) -> anyhow::Result<(usize, usize)> {
    let local = Array::from_slice(&[i32::try_from(local)?], &[1]).copy(stream)?;
    let gathered = distributed::all_gather(&local, group, stream)?;
    eval([&gathered])?;
    let evaluated = gathered.evaluated()?;
    let routes = evaluated.as_slice::<i32>();
    Ok((
        routes.iter().copied().min().unwrap_or(0) as usize,
        routes.iter().copied().max().unwrap_or(0) as usize,
    ))
}

fn last_logits(logits: &Array, stream: &Stream) -> anyhow::Result<Array> {
    if logits.ndim() == 3 {
        Ok(logits.try_index_device((.., -1, ..), stream)?)
    } else {
        Ok(logits.clone())
    }
}

fn max_abs_difference(left: &Array, right: &Array, stream: &Stream) -> anyhow::Result<f32> {
    anyhow::ensure!(
        left.shape() == right.shape(),
        "logit shape mismatch: {:?} versus {:?}",
        left.shape(),
        right.shape()
    );
    Ok(left
        .subtract(right, stream)?
        .abs(stream)?
        .max(None, stream)?
        .item::<f32>(stream))
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn print_phase(
    strategy: &str,
    phase: &str,
    iterations: usize,
    result: &PhaseResult,
    routes: (usize, usize),
) {
    let stats = &result.statistics;
    let imbalance = if routes.0 == 0 {
        f64::INFINITY
    } else {
        routes.1 as f64 / routes.0 as f64
    };
    let throughput = if result.seconds == 0.0 {
        0.0
    } else {
        result.tokens as f64 / result.seconds
    };
    println!(
        "{strategy},{phase},{iterations},{},{:.3},{throughput:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{imbalance:.3},{},{},{},{:.3}",
        result.tokens,
        result.seconds * 1_000.0,
        milliseconds(stats.router_time),
        milliseconds(stats.compaction_time),
        milliseconds(stats.exchange_time),
        milliseconds(stats.expert_time),
        milliseconds(stats.reduction_time),
        milliseconds(stats.shared_expert_time),
        milliseconds(stats.total_time),
        milliseconds(stats.model_time),
        routes.0,
        routes.1,
        stats.padding_routes,
        stats.exchanged_bytes,
        stats.synchronization_count,
        milliseconds(stats.synchronization_time),
    );
}
