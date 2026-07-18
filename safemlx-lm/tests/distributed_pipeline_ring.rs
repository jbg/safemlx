#![cfg(unix)]

use std::{
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    pipeline::{load_pipeline_model, load_pipeline_model_with_options, PipelineStep},
    sampler::DefaultSampler,
    DenseDiskStreamLoadOptions, DeviceAssignment, ModelLoadOptions, ParallelTopology,
    WeightResidency,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_PIPELINE_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_PIPELINE_CHECKPOINT";
const DENSE_STREAM: &str = "SAFEMLX_LM_PIPELINE_DENSE_STREAM";

#[test]
fn pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, 1, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let dense_stream = std::env::var_os(DENSE_STREAM).is_some();
    let mut model = if dense_stream {
        let dense = DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap();
        load_pipeline_model_with_options(
            &checkpoint,
            ModelLoadOptions::with_parallel(topology)
                .with_weight_residency(WeightResidency::DenseDiskStream(dense)),
            &stream,
            &stream,
        )
        .unwrap()
    } else {
        load_pipeline_model(&checkpoint, topology, &stream, &stream).unwrap()
    };
    let info = model.stage_info();
    assert_eq!(info.global_layer_range, expected_rank..expected_rank + 1);
    assert_eq!(
        info.owned_tensors
            .iter()
            .any(|name| name.starts_with(&format!("model.layers.{expected_rank}."))),
        !dense_stream
    );
    assert!(!info
        .owned_tensors
        .iter()
        .any(|name| name.starts_with(&format!("model.layers.{}.", 1 - expected_rank))));
    assert_eq!(
        info.owned_tensors
            .iter()
            .any(|name| name == "model.embed_tokens.weight"),
        expected_rank == 0
    );
    assert_eq!(
        info.owned_tensors
            .iter()
            .any(|name| name == "lm_head.weight"),
        expected_rank == 1
    );
    assert!(info.local_parameter_bytes < 1_616);
    let opened = info
        .opened_checkpoint_shards
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        opened.contains(&format!("layer-{expected_rank}.safetensors")),
        !dense_stream
    );
    assert!(!opened.contains(&format!("layer-{}.safetensors", 1 - expected_rank)));
    assert_eq!(
        opened.contains(&"input.safetensors".into()),
        expected_rank == 0
    );
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.planned_layer_count(), 1);
        assert!(report
            .residency()
            .units()
            .iter()
            .all(|unit| !unit.host_resident() && !unit.device_resident()));
    }
    assert_eq!(
        opened.contains(&"output.safetensors".into()),
        expected_rank == 1
    );

    let mut cache = model.new_cache();
    assert_eq!(cache.global_layers(), vec![expected_rank]);
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let mut logits = model
        .forward_pipeline(
            (expected_rank == 0).then_some(&prompt),
            PipelineStep::new(1, 2).unwrap(),
            None,
            &mut cache,
            &group,
            &stream,
        )
        .unwrap();
    assert_eq!(logits.is_some(), expected_rank == 1);

    let mut sampler = DefaultSampler;
    for _ in 0..2 {
        let synchronized = model
            .sample_and_synchronize(
                logits.as_ref(),
                PipelineStep::new(1, 1).unwrap(),
                &mut sampler,
                0.0,
                None,
                false,
                &group,
                &stream,
            )
            .unwrap();
        let token = synchronized.token.evaluated().unwrap();
        assert_eq!(token.as_array().shape(), &[1, 1]);
        // The synthetic fixture uses identical positive projection rows, so
        // greedy selection deterministically matches the full-model reference
        // tie break at vocabulary id zero.
        assert_eq!(token.as_slice::<u32>(), &[0]);
        drop(token);
        logits = model
            .forward_pipeline(
                (expected_rank == 0).then_some(&synchronized.token),
                PipelineStep::new(1, 1).unwrap(),
                None,
                &mut cache,
                &group,
                &stream,
            )
            .unwrap();
    }
    if dense_stream {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert!(report.prefill_forwards() >= 1);
        assert!(report.decode_forwards() >= 2);
    }
}

struct ChildGuard {
    children: Vec<Child>,
}

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.children
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
        }
        for child in &mut self.children {
            let _ = child.wait();
        }
    }
}

fn write_f32_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let buffers = tensors
        .iter()
        .map(|(_, shape, value)| {
            let count = shape.iter().product::<usize>();
            (0..count)
                .flat_map(|_| value.to_le_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .zip(&buffers)
        .map(|((name, shape, _), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
    serialize_to_file(views, None, path).unwrap();
}

fn write_fixture(directory: &Path) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "intermediate_size": 8,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 8,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false
        }))
        .unwrap(),
    )
    .unwrap();
    write_f32_shard(
        &directory.join("input.safetensors"),
        &[("model.embed_tokens.weight", vec![8, 4], 0.01)],
    );
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let names = [
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![4, 4],
                0.01,
            ),
            (format!("{prefix}.mlp.gate_proj.weight"), vec![8, 4], 0.01),
            (format!("{prefix}.mlp.up_proj.weight"), vec![8, 4], 0.01),
            (format!("{prefix}.mlp.down_proj.weight"), vec![4, 8], 0.01),
            (format!("{prefix}.input_layernorm.weight"), vec![4], 1.0),
            (
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![4],
                1.0,
            ),
        ];
        let borrowed = names
            .iter()
            .map(|(name, shape, value)| (name.as_str(), shape.clone(), *value))
            .collect::<Vec<_>>();
        write_f32_shard(
            &directory.join(format!("layer-{layer}.safetensors")),
            &borrowed,
        );
    }
    write_f32_shard(
        &directory.join("output.safetensors"),
        &[
            ("model.norm.weight", vec![4], 1.0),
            ("lm_head.weight", vec![8, 4], 0.01),
        ],
    );
    let mut weight_map = serde_json::Map::new();
    weight_map.insert(
        "model.embed_tokens.weight".into(),
        serde_json::json!("input.safetensors"),
    );
    for layer in 0..2 {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            weight_map.insert(
                format!("model.layers.{layer}.{suffix}"),
                serde_json::json!(format!("layer-{layer}.safetensors")),
            );
        }
    }
    weight_map.insert(
        "model.norm.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    weight_map.insert(
        "lm_head.weight".into(),
        serde_json::json!("output.safetensors"),
    );
    std::fs::write(
        directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": weight_map
        }))
        .unwrap(),
    )
    .unwrap();
}

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "pipeline Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_pipeline_ring ring_two_process_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_pipeline() {
    run_ring_pipeline(false);
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_pipeline_ring ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_dense_stream_pipeline() {
    run_ring_pipeline(true);
}

fn run_ring_pipeline(dense_stream: bool) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first_socket);
    drop(second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(2),
    };
    for rank in 0..2 {
        let mut command = Command::new(&executable);
        command
            .args(["--exact", "pipeline_ring_worker", "--nocapture"])
            .env(WORKER_RANK, rank.to_string())
            .env(CHECKPOINT_DIR, checkpoint.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if dense_stream {
            command.env(DENSE_STREAM, "1");
        }
        children.children.push(command.spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut timed_out = false;
    loop {
        let statuses = children
            .children
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.children {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let outputs = children.finish();
    let failures = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process pipeline Ring test failed:\n{}",
        if timed_out {
            format!("timed out after 45 seconds\n\n{}", failures.join("\n\n"))
        } else {
            failures.join("\n\n")
        }
    );
}
