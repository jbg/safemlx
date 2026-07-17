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
    sampler::DefaultSampler, tensor_parallel::load_tensor_parallel_model, DeviceAssignment,
    ParallelTopology,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_TENSOR_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_TENSOR_CHECKPOINT";

#[test]
fn tensor_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let mut model = load_tensor_parallel_model(&checkpoint, topology, &stream, &stream).unwrap();
    let info = model.info();
    assert_eq!(info.local_attention_heads, 1);
    assert_eq!(info.local_kv_heads, 1);
    assert_eq!(
        info.local_vocabulary_range,
        if expected_rank == 0 { 0..3 } else { 3..5 }
    );
    assert!(info.local_parameter_bytes < 656);
    assert!(info
        .owned_tensors
        .iter()
        .any(|name| name == "model.embed_tokens.weight"));

    let mut cache = model.new_cache();
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let mut logits = model.prefill(&prompt, &mut cache, &group, &stream).unwrap();
    assert_eq!(logits.shape(), &[1, 2, 5]);
    let mut sampler = DefaultSampler;
    for _ in 0..2 {
        let synchronized = model
            .sample_and_synchronize(&logits, &mut sampler, 0.0, None, false, 0, &group, &stream)
            .unwrap();
        let token = synchronized.token.evaluated().unwrap();
        assert_eq!(token.as_slice::<u32>(), &[0]);
        drop(token);
        logits = model
            .decode(&synchronized.token, &mut cache, &group, &stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 5]);
    }
    assert_eq!(cache.offset(), 4);
}

fn write_f32_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
    let buffers = tensors
        .iter()
        .map(|(_, shape, value)| {
            (0..shape.iter().product::<usize>())
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
            "num_hidden_layers": 1,
            "intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 5,
            "max_position_embeddings": 32,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "mlp_bias": false
        }))
        .unwrap(),
    )
    .unwrap();
    let tensors = [
        ("model.embed_tokens.weight", vec![5, 4], 0.01),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.k_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.v_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.gate_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.up_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 4], 0.01),
        ("model.layers.0.input_layernorm.weight", vec![4], 1.0),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![4],
            1.0,
        ),
        ("model.norm.weight", vec![4], 1.0),
        ("lm_head.weight", vec![5, 4], 0.01),
    ];
    write_f32_shard(&directory.join("model.safetensors"), &tensors);
    let weight_map = tensors
        .iter()
        .map(|(name, _, _)| ((*name).to_string(), serde_json::json!("model.safetensors")))
        .collect::<serde_json::Map<_, _>>();
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

struct ChildGuard(Vec<Child>);

impl ChildGuard {
    fn finish(mut self) -> Vec<Output> {
        self.0
            .drain(..)
            .map(|child| child.wait_with_output().unwrap())
            .collect()
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
        }
        for child in &mut self.0 {
            let _ = child.wait();
        }
    }
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
        "tensor Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_tensor_parallel_ring ring_two_process_tensor_parallel -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_tensor_parallel() {
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
    let mut children = ChildGuard(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args(["--exact", "tensor_ring_worker", "--nocapture"])
                .env(WORKER_RANK, rank.to_string())
                .env(CHECKPOINT_DIR, checkpoint.path())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process tensor-parallel Ring test failed:\n{}",
        failures.join("\n\n")
    );
}
