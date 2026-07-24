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
    module::ModuleParameters,
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_lm::{
    models::deepseek_v3, module_binding::canonical_checkpoint_name, sampler::DefaultSampler,
    tensor_parallel::load_tensor_parallel_model, CacheResidencyPolicy, DeviceAssignment,
    PagedCacheOptions, ParallelTopology, PromptCacheDescriptor, PromptCacheOptions,
    PromptCacheTopology,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_TENSOR_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_TENSOR_CHECKPOINT";
const PROMPT_CACHE_ROOT: &str = "SAFEMLX_LM_TENSOR_PROMPT_CACHE";

#[test]
fn tensor_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    let deepseek = config["model_type"] == "deepseek_v3";
    let layer_count = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let prompt_cache_root = PathBuf::from(std::env::var_os(PROMPT_CACHE_ROOT).unwrap());
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
        if deepseek {
            if expected_rank == 0 {
                0..4
            } else {
                4..8
            }
        } else if expected_rank == 0 {
            0..3
        } else {
            3..5
        }
    );
    if !deepseek {
        assert!(info.local_parameter_bytes < 656);
    }
    assert!(info
        .owned_tensors
        .iter()
        .any(|name| name == "model.embed_tokens.weight"));

    let paged = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model
        .new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
        .unwrap();
    let prompt = safemlx::Array::from_slice(&[1u32, 2], &[1, 2]);
    let logits = model.prefill(&prompt, &mut cache, &group, &stream).unwrap();
    assert_eq!(logits.shape(), &[1, 2, vocab_size as i32]);
    let descriptor = PromptCacheDescriptor {
        model_family: if deepseek { "deepseek_v3" } else { "llama" }.into(),
        effective_model_type: if deepseek { "deepseek_v3" } else { "llama" }.into(),
        checkpoint_fingerprint: "tensor-ring-fixture".into(),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        batch_size: 1,
        sliding_window: None,
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: None,
            tensor_parallel: Some((2, expected_rank)),
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        },
    };
    model
        .save_prompt_cache(
            &mut cache,
            &prompt_cache_root,
            descriptor.clone(),
            &[1, 2],
            &PromptCacheOptions::default(),
        )
        .unwrap();
    let token = safemlx::Array::from_slice(&[0u32], &[1, 1]);
    let uninterrupted = model.decode(&token, &mut cache, &group, &stream).unwrap();
    let uninterrupted = uninterrupted.evaluated().unwrap();
    let uninterrupted_values = uninterrupted.as_slice::<f32>().to_vec();
    drop(uninterrupted);
    let (mut cache, manifest) = model
        .load_prompt_cache(&prompt_cache_root, &descriptor, &[1, 2], paged)
        .unwrap();
    assert_eq!(manifest.topology, descriptor.topology);
    let restored = model.decode(&token, &mut cache, &group, &stream).unwrap();
    let restored = restored.evaluated().unwrap();
    assert_eq!(uninterrupted_values, restored.as_slice::<f32>());
    let mut logits = restored.as_array().clone();
    drop(restored);
    let mut sampler = DefaultSampler;
    for _ in 0..1 {
        let synchronized = model
            .sample_and_synchronize(&logits, &mut sampler, 0.0, None, false, 0, &group, &stream)
            .unwrap();
        let token = synchronized.token.evaluated().unwrap();
        assert_eq!(token.as_slice::<u32>(), &[0]);
        drop(token);
        logits = model
            .decode(&synchronized.token, &mut cache, &group, &stream)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, vocab_size as i32]);
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

fn write_deepseek_fixture(directory: &Path, layers: i32) {
    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_hidden_layers": layers,
        "num_attention_heads": 2,
        "vocab_size": 8,
        "rms_norm_eps": 0.000001,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": null,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "first_k_dense_replace": layers,
        "moe_layer_freq": 1,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.0,
        "num_nextn_predict_layers": 0,
        "split_kv_b": false,
        "tie_word_embeddings": false
    });
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args: deepseek_v3::ModelArgs = serde_json::from_value(config.clone()).unwrap();
    let mut model = deepseek_v3::Model::new(args, stream).unwrap();
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if name.ends_with("norm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(&shape, Array::from_f32(0.01), stream).unwrap()
        };
    }
    let arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| (canonical_checkpoint_name(&name), value.clone()))
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
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
    run_ring_tensor_parallel(false);
}

/// Verifies DeepSeek MLA paged-prefix persistence across two tensor ranks.
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_deepseek_tensor_parallel_persistence() {
    run_ring_tensor_parallel(true);
}

fn run_ring_tensor_parallel(deepseek: bool) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    if deepseek {
        write_deepseek_fixture(checkpoint.path(), 1);
    } else {
        write_fixture(checkpoint.path());
    }
    let prompt_cache = tempfile::tempdir().unwrap();
    let (first_socket, second_socket, first_port, second_port) = reserve_two_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    let mut reservations = [Some(first_socket), Some(second_socket)];
    for (rank, reservation) in reservations.iter_mut().enumerate() {
        // Release only the address this rank will bind immediately before its
        // process is spawned. Keeping the peer address reserved closes the
        // previous socket-setup race where either port could be stolen between
        // dropping both listeners and launching the workers.
        drop(reservation.take());
        children.0.push(
            Command::new(&executable)
                .args(["--exact", "tensor_ring_worker", "--nocapture"])
                .env(WORKER_RANK, rank.to_string())
                .env(CHECKPOINT_DIR, checkpoint.path())
                .env(PROMPT_CACHE_ROOT, prompt_cache.path())
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
