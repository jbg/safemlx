#![cfg(unix)]

use std::{
    collections::HashMap,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    argmax_axis,
    distributed::{self, Backend},
    error::Exception,
    module::{Module, ModuleParameters, Param},
    ops::{indexing::TryIndexOp, ones_dtype, zeros_dtype},
    transforms::eval,
    Array, Device, DeviceType, Dtype, ExecutionContext, Stream,
};
use safemlx_lm::{
    cache::{ConcatKeyValueCache, SlidingKeyValueCache},
    expert_cache::ExpertCacheLoadOptions,
    expert_parallel::{
        load_expert_parallel_model_with_options,
        load_expert_parallel_model_with_options_and_assignment, profile_expert_parallel_timings,
        ExpertAssignment,
    },
    inspection::{ActivationObserver, MoeRoutingObservation},
    models::{
        deepseek_v3, gpt_oss, inkling, lfm2, nemotron_h, qwen3, qwen3_5_moe, qwen3_vl,
        ModelLoadOptions,
    },
    quantization::{AffineQuantization, WeightQuantization},
    sampler::DefaultSampler,
    CacheResidencyPolicy, DeviceAssignment, PagedCacheOptions, ParallelTopology,
    PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology, WeightResidency,
};

const WORKER_RANK: &str = "SAFEMLX_LM_EXPERT_MODEL_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_EXPERT_MODEL_CHECKPOINT";
const PAGED_PROMPT_MARKER: &str = ".paged-prompt-parity";
const EXPECTED_FILE: &str = "SAFEMLX_LM_EXPERT_MODEL_EXPECTED";
const ARCHITECTURE: &str = "SAFEMLX_LM_EXPERT_MODEL_ARCHITECTURE";
const ENCODING: &str = "SAFEMLX_LM_EXPERT_MODEL_ENCODING";
const ASSIGNMENT: &str = "SAFEMLX_LM_EXPERT_MODEL_ASSIGNMENT";
const RESIDENCY: &str = "SAFEMLX_LM_EXPERT_MODEL_RESIDENCY";

struct EpObserver {
    names: Vec<String>,
    routing_observations: usize,
    saw_local: bool,
    saw_reduced: bool,
    saw_shared: bool,
    hidden_size: i32,
    routes: i32,
}

impl ActivationObserver for EpObserver {
    fn observe(&mut self, name: &str, _value: &Array) -> Result<(), Exception> {
        self.names.push(name.to_string());
        Ok(())
    }

    fn observe_moe_routing(&mut self, routing: MoeRoutingObservation<'_>) -> Result<(), Exception> {
        assert_eq!(routing.selected_experts.shape(), &[self.routes, 2]);
        assert_eq!(routing.selected_scores.shape(), &[self.routes, 2]);
        assert_eq!(routing.routing_weights.shape(), &[self.routes, 2]);
        assert_eq!(
            routing.routed_output.shape(),
            &[self.routes, self.hidden_size]
        );
        self.saw_local |= routing.local_routed_output.is_some();
        self.saw_reduced |= routing.reduced_routed_output.is_some();
        self.saw_shared |= routing.shared_output.is_some();
        self.routing_observations += 1;
        Ok(())
    }
}

fn assert_close(actual: &Array, expected: &Array) {
    eval([actual, expected]).unwrap();
    let actual = actual.evaluated().unwrap();
    let expected = expected.evaluated().unwrap();
    assert_eq!(actual.as_array().shape(), expected.as_array().shape());
    for (actual, expected) in actual
        .as_slice::<f32>()
        .iter()
        .zip(expected.as_slice::<f32>())
    {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "EP logit {actual} differs from single-model logit {expected}"
        );
    }
}

fn greedy_token(logits: &Array, stream: &Stream) -> Array {
    let last = if logits.ndim() == 3 {
        logits.try_index_device((.., -1, ..), stream).unwrap()
    } else {
        logits.clone()
    };
    argmax_axis!(&last, -1, stream = stream)
        .unwrap()
        .reshape(&[last.dim(0), 1], stream)
        .unwrap()
}

#[test]
fn expert_parallel_model_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT_DIR).unwrap());
    let architecture = std::env::var(ARCHITECTURE).unwrap();
    let encoding = std::env::var(ENCODING).unwrap_or_else(|_| "dense".into());
    let assignment_kind = std::env::var(ASSIGNMENT).unwrap_or_else(|_| "balanced".into());
    let residency = std::env::var(RESIDENCY).unwrap_or_else(|_| "resident".into());
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    let hidden_size = config["hidden_size"].as_i64().unwrap() as i32;
    let moe_intermediate_size = config["moe_intermediate_size"].as_i64().unwrap() as usize;
    let num_layers = config["num_hidden_layers"].as_i64().unwrap() as usize;
    let moe_layers = if let Some(value) = config.get("test_moe_layers") {
        value.as_u64().unwrap() as usize
    } else if architecture == "DeepSeekV3" {
        let dense_prefix = config["first_k_dense_replace"].as_i64().unwrap() as usize;
        let frequency = config["moe_layer_freq"].as_i64().unwrap() as usize;
        (0..num_layers)
            .filter(|layer| *layer >= dense_prefix && *layer % frequency == 0)
            .count()
    } else {
        num_layers
    };
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, 1, 1, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let context = ExecutionContext::new(topology.device.device().unwrap());
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let weights_stream = weights_context.stream();
    let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
    let mut options = ModelLoadOptions::with_parallel(topology);
    if encoding == "affine" {
        options.quantization = Some(quantization);
    }
    if residency == "sparse-cache" {
        options.weight_residency =
            WeightResidency::SparseExpertCache(ExpertCacheLoadOptions::default());
    }
    let assignment = match assignment_kind.as_str() {
        "balanced" => None,
        "round-robin" => Some(ExpertAssignment::round_robin(4, 2, expected_rank).unwrap()),
        "explicit" => Some(ExpertAssignment::explicit(vec![1, 0, 0, 1], 2, expected_rank).unwrap()),
        other => panic!("unknown expert assignment {other}"),
    };
    let mut model = if let Some(assignment) = assignment {
        load_expert_parallel_model_with_options_and_assignment(
            &checkpoint,
            options,
            assignment,
            stream,
            weights_stream,
        )
        .unwrap()
    } else {
        load_expert_parallel_model_with_options(&checkpoint, options, stream, weights_stream)
            .unwrap()
    };
    let info = model.info();
    let expected_experts: &[usize] = match (assignment_kind.as_str(), expected_rank) {
        ("balanced", 0) => &[0, 1],
        ("balanced", 1) => &[2, 3],
        ("round-robin", 0) => &[0, 2],
        ("round-robin", 1) => &[1, 3],
        ("explicit", 0) => &[1, 2],
        ("explicit", 1) => &[0, 3],
        _ => unreachable!(),
    };
    assert_eq!(info.assignment.local_global_expert_ids(), expected_experts);
    let dense_routed_bytes = 2 * moe_layers * 3 * moe_intermediate_size * hidden_size as usize * 4;
    if residency == "sparse-cache" {
        assert_eq!(info.routed_expert_bytes, 0);
        assert!(info.owned_expert_bytes > 0);
    } else if encoding == "dense" {
        assert_eq!(info.routed_expert_bytes, dense_routed_bytes);
    } else {
        assert!(
            info.routed_expert_bytes < dense_routed_bytes,
            "{encoding} routed bank was not physically packed"
        );
    }
    assert!(info.replicated_parameter_bytes > 0);
    assert_eq!(
        info.local_parameter_bytes,
        info.replicated_parameter_bytes + info.routed_expert_bytes
    );
    let opened = info
        .opened_checkpoint_shards
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if checkpoint.join("replicated.safetensors").exists() {
        assert!(opened.contains(&"replicated.safetensors".to_string()));
        for expert in 0..4 {
            let expected_open = residency != "sparse-cache" && expected_experts.contains(&expert);
            assert_eq!(
                opened.contains(&format!("expert-{expert}.safetensors")),
                expected_open,
                "rank {expected_rank} opened the wrong expert shards: {opened:?}"
            );
        }
    } else {
        assert_eq!(opened, vec!["model.safetensors"]);
    }

    let expected =
        Array::load_safetensors(std::env::var_os(EXPECTED_FILE).unwrap(), weights_stream)
            .unwrap()
            .into_iter()
            .map(|(name, value)| (name, value.copy(stream).unwrap()))
            .collect::<HashMap<_, _>>();
    let _profiling = profile_expert_parallel_timings();
    let prompt = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let paged_prompt = checkpoint.join(PAGED_PROMPT_MARKER).exists();
    let paged = PagedCacheOptions::new(1, 16 * 1024, 16 * 1024, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = if paged_prompt {
        model
            .new_cache_with_options(CacheResidencyPolicy::Paged(paged.clone()))
            .unwrap()
    } else {
        model.new_cache()
    };
    let prefill = model.prefill(&prompt, &mut cache, &group, stream).unwrap();
    assert_close(&prefill, &expected["prefill"]);
    assert_eq!(
        model.latest_routing_statistics().total_routes,
        6 * moe_layers
    );
    if architecture == "DeepSeekV3" && assignment_kind == "balanced" && expected_rank == 0 {
        assert_eq!(model.latest_routing_statistics().local_routes, 0);
    } else if assignment_kind == "balanced" && residency != "sparse-cache" {
        assert!(model.latest_routing_statistics().local_routes > 0);
    }

    let mut sampler = DefaultSampler;
    let first = model
        .sample_and_synchronize(&prefill, &mut sampler, 0.0, None, false, 0, &group, stream)
        .unwrap();
    assert_close(
        &first.token.as_type::<f32>(stream).unwrap(),
        &expected["first_token"],
    );
    assert!(!first.finished);

    let uninterrupted = model
        .decode(&first.token, &mut cache, &group, stream)
        .unwrap();
    let decode = if paged_prompt {
        let uninterrupted_values = uninterrupted.evaluated().unwrap();
        let uninterrupted_values = uninterrupted_values.as_slice::<f32>().to_vec();
        let model_type = config["model_type"].as_str().unwrap().to_string();
        let descriptor = PromptCacheDescriptor {
            model_family: if architecture == "DeepSeekV3" {
                "deepseek_v3".into()
            } else {
                "gpt_oss".into()
            },
            effective_model_type: model_type,
            checkpoint_fingerprint: "expert-ring-fixture".into(),
            architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
            layer_count: num_layers,
            global_layer_start: 0,
            global_layer_end: num_layers,
            batch_size: 1,
            sliding_window: (architecture == "GptOss")
                .then(|| config["sliding_window"].as_i64().unwrap() as i32),
            sink_tokens: 0,
            topology: PromptCacheTopology {
                pipeline: None,
                tensor_parallel: None,
                expert_parallel: Some((2, expected_rank)),
                expert_parallel_cache_replicated: true,
            },
        };
        let root = checkpoint.join("prompt-cache");
        // Persist the prefix state, not the uninterrupted suffix token.
        cache.reset().unwrap();
        let _ = model.prefill(&prompt, &mut cache, &group, stream).unwrap();
        model
            .save_prompt_cache(
                &mut cache,
                &root,
                descriptor.clone(),
                &[1, 2, 3],
                &PromptCacheOptions::default(),
            )
            .unwrap();
        let (mut restored, manifest) = model
            .load_prompt_cache(&root, &descriptor, &[1, 2, 3], paged)
            .unwrap();
        assert_eq!(manifest.topology, descriptor.topology);
        let restored = model
            .decode(&first.token, &mut restored, &group, stream)
            .map(|restored_logits| (restored, restored_logits))
            .unwrap();
        let restored_values = restored.1.evaluated().unwrap();
        assert_eq!(uninterrupted_values, restored_values.as_slice::<f32>());
        drop(restored_values);
        cache = restored.0;
        restored.1
    } else {
        uninterrupted
    };
    assert_close(&decode, &expected["decode"]);
    assert_eq!(
        model.latest_routing_statistics().total_routes,
        2 * moe_layers
    );
    if architecture == "DeepSeekV3" && assignment_kind == "balanced" && expected_rank == 0 {
        assert_eq!(model.latest_routing_statistics().local_routes, 0);
    } else if assignment_kind == "balanced" && residency != "sparse-cache" {
        assert!(model.latest_routing_statistics().local_routes > 0);
    }
    assert_eq!(cache.offset(), 4);

    let second = model
        .sample_and_synchronize(&decode, &mut sampler, 0.0, None, false, 1, &group, stream)
        .unwrap();
    assert_close(
        &second.token.as_type::<f32>(stream).unwrap(),
        &expected["second_token"],
    );
    let decode_second = model
        .decode(&second.token, &mut cache, &group, stream)
        .unwrap();
    assert_close(&decode_second, &expected["decode_second"]);
    assert_eq!(
        model.latest_routing_statistics().total_routes,
        2 * moe_layers
    );
    if architecture == "DeepSeekV3" && assignment_kind == "balanced" && expected_rank == 0 {
        assert_eq!(model.latest_routing_statistics().local_routes, 0);
    } else if assignment_kind == "balanced" && residency != "sparse-cache" {
        assert!(model.latest_routing_statistics().local_routes > 0);
    }
    let third = model
        .sample_and_synchronize(
            &decode_second,
            &mut sampler,
            0.0,
            None,
            false,
            0,
            &group,
            stream,
        )
        .unwrap();
    assert_close(
        &third.token.as_type::<f32>(stream).unwrap(),
        &expected["third_token"],
    );
    assert_eq!(cache.offset(), 5);

    if residency != "sparse-cache" {
        let mut observed_cache = model.new_cache();
        let mut observer = EpObserver {
            names: Vec::new(),
            routing_observations: 0,
            saw_local: false,
            saw_reduced: false,
            saw_shared: false,
            hidden_size,
            routes: 3,
        };
        let observed = model
            .forward_with_observer(
                &prompt,
                None,
                &mut observed_cache,
                &group,
                &mut observer,
                stream,
            )
            .unwrap();
        assert_close(&observed, &expected["prefill"]);
        assert_eq!(observer.routing_observations, moe_layers);
        assert!(observer.saw_local);
        assert!(observer.saw_reduced);
        assert_eq!(observer.saw_shared, architecture == "DeepSeekV3");
        assert!(observer.names.iter().any(|name| name.contains("local")));
        assert!(observer.names.iter().any(|name| name.contains("reduced")));
    }
    let timings = model.latest_routing_statistics();
    assert!(timings.total_time > Duration::ZERO);
    assert!(timings.model_time > Duration::ZERO);
    assert!(timings.compaction_time > Duration::ZERO);
    assert!(timings.expert_time > Duration::ZERO);
    assert!(timings.reduction_time > Duration::ZERO);
    assert_eq!(timings.exchange_time, Duration::ZERO);
    if residency != "sparse-cache" {
        assert!(timings.router_time > Duration::ZERO);
        assert_eq!(
            timings.shared_expert_time > Duration::ZERO,
            architecture == "DeepSeekV3"
        );
    }

    if architecture == "Qwen3" {
        let paging = PagedCacheOptions::new(1, 1 << 20, 1 << 20, 1).unwrap();
        let mut sliding_cache = model.new_qwen3_sliding_cache(2, paging).unwrap();
        let sliding_prefill = model
            .prefill(&prompt, &mut sliding_cache, &group, stream)
            .unwrap();
        assert_close(&sliding_prefill, &expected["sliding_prefill"]);
        let sliding_first = model
            .sample_and_synchronize(
                &sliding_prefill,
                &mut sampler,
                0.0,
                None,
                false,
                0,
                &group,
                stream,
            )
            .unwrap();
        assert_close(
            &sliding_first.token.as_type::<f32>(stream).unwrap(),
            &expected["sliding_first_token"],
        );
        let sliding_decode = model
            .decode(&sliding_first.token, &mut sliding_cache, &group, stream)
            .unwrap();
        assert_close(&sliding_decode, &expected["sliding_decode"]);
        let sliding_second = model
            .sample_and_synchronize(
                &sliding_decode,
                &mut sampler,
                0.0,
                None,
                false,
                1,
                &group,
                stream,
            )
            .unwrap();
        assert_close(
            &sliding_second.token.as_type::<f32>(stream).unwrap(),
            &expected["sliding_second_token"],
        );
        let sliding_decode_second = model
            .decode(&sliding_second.token, &mut sliding_cache, &group, stream)
            .unwrap();
        assert_close(&sliding_decode_second, &expected["sliding_decode_second"]);
        assert_eq!(sliding_cache.offset(), 5);
    } else {
        let paging = PagedCacheOptions::new(1, 1 << 20, 1 << 20, 1).unwrap();
        assert!(model.new_qwen3_sliding_cache(2, paging).is_err());
    }
    assert_eq!(architecture, format!("{:?}", model.info().model_kind));
}

fn patterned_parameter(name: &str, shape: &[i32], stream: &Stream) -> Array {
    let count = shape.iter().map(|value| *value as usize).product::<usize>();
    let values = if name.ends_with("norm.weight") {
        vec![1.0f32; count]
    } else if name.ends_with("mlp.gate.weight") {
        let row_width = count / 4;
        [0.04f32, 0.01, 0.02, 0.03]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, row_width))
            .collect()
    } else if name.ends_with("e_score_correction_bias") {
        [0.0f32, 0.0, 0.03, 0.02]
            .into_iter()
            .cycle()
            .take(count)
            .collect()
    } else if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        let row_width = count / shape[0] as usize;
        (0..shape[0])
            .flat_map(|row| std::iter::repeat_n(0.001f32 * (row + 1) as f32, row_width))
            .collect()
    } else if name.contains(".mlp.experts.") {
        let expert_width = count / 4;
        (0..4)
            .flat_map(|expert| std::iter::repeat_n(0.005f32 * (expert + 1) as f32, expert_width))
            .collect()
    } else {
        vec![0.01f32; count]
    };
    Array::from_slice(&values, shape).copy(stream).unwrap()
}

fn initialize_parameters(model: &mut impl ModuleParameters, stream: &Stream) {
    for (name, parameter) in model.parameters_mut().flatten() {
        *parameter = patterned_parameter(&name, parameter.shape(), stream);
    }
}

fn save_arrays(path: &Path, arrays: &[(String, Array)]) {
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        path,
    )
    .unwrap();
}

fn save_index(directory: &Path, shards: &[(&str, &[(String, Array)])]) {
    let mut weight_map = serde_json::Map::new();
    for (file, tensors) in shards {
        for (name, _) in *tensors {
            weight_map.insert(name.clone(), serde_json::json!(file));
        }
    }
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

fn split_qwen_checkpoint(model: &qwen3::Model, directory: &Path, stream: &Stream) {
    let mut replicated = Vec::new();
    let mut experts = (0..4).map(|_| Vec::new()).collect::<Vec<_>>();
    for (name, value) in model.parameters().flatten() {
        let name = name.to_string();
        if let Some(prefix) = name.strip_suffix(".gate_up_proj") {
            for expert in 0..4 {
                let local = value.try_index_device(expert, stream).unwrap();
                experts[expert as usize].push((
                    format!("{prefix}.{expert}.gate_proj.weight"),
                    local.try_index_device((..4, ..), stream).unwrap(),
                ));
                experts[expert as usize].push((
                    format!("{prefix}.{expert}.up_proj.weight"),
                    local.try_index_device((4.., ..), stream).unwrap(),
                ));
            }
        } else if let Some(prefix) = name.strip_suffix(".down_proj") {
            if prefix.ends_with(".mlp.experts") {
                for expert in 0..4 {
                    experts[expert as usize].push((
                        format!("{prefix}.{expert}.down_proj.weight"),
                        value.try_index_device(expert, stream).unwrap(),
                    ));
                }
            } else {
                replicated.push((name, value.clone()));
            }
        } else {
            replicated.push((name, value.clone()));
        }
    }
    save_partitioned_checkpoint(directory, replicated, experts);
}

fn split_deepseek_checkpoint(model: &deepseek_v3::Model, directory: &Path, stream: &Stream) {
    let mut replicated = Vec::new();
    let mut experts = (0..4).map(|_| Vec::new()).collect::<Vec<_>>();
    for (name, value) in model.parameters().flatten() {
        let name = name.to_string();
        let projection = ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .find(|projection| name.ends_with(&format!(".mlp.experts.{projection}")));
        if let Some(projection) = projection {
            let prefix = name.strip_suffix(&format!(".{projection}")).unwrap();
            for expert in 0..4 {
                experts[expert as usize].push((
                    format!("{prefix}.{expert}.{projection}.weight"),
                    value.try_index_device(expert, stream).unwrap(),
                ));
            }
        } else if let Some(projection) = ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .find(|projection| name.ends_with(&format!(".mlp.experts.{projection}_scale_inv")))
        {
            let prefix = name
                .strip_suffix(&format!(".{projection}_scale_inv"))
                .unwrap();
            for expert in 0..4 {
                experts[expert as usize].push((
                    format!("{prefix}.{expert}.{projection}.weight_scale_inv"),
                    value.try_index_device(expert, stream).unwrap(),
                ));
            }
        } else {
            replicated.push((name, value.clone()));
        }
    }
    save_partitioned_checkpoint(directory, replicated, experts);
}

fn save_partitioned_checkpoint(
    directory: &Path,
    replicated: Vec<(String, Array)>,
    experts: Vec<Vec<(String, Array)>>,
) {
    save_arrays(&directory.join("replicated.safetensors"), &replicated);
    for (expert, arrays) in experts.iter().enumerate() {
        save_arrays(
            &directory.join(format!("expert-{expert}.safetensors")),
            arrays,
        );
    }
    let expert_files = (0..4)
        .map(|expert| format!("expert-{expert}.safetensors"))
        .collect::<Vec<_>>();
    let mut shards = vec![("replicated.safetensors", replicated.as_slice())];
    shards.extend(
        expert_files
            .iter()
            .zip(&experts)
            .map(|(file, arrays)| (file.as_str(), arrays.as_slice())),
    );
    save_index(directory, &shards);
}

#[allow(clippy::too_many_arguments)]
fn save_expected(
    path: &Path,
    prefill: Array,
    decode: Array,
    decode_second: Array,
    first: Array,
    second: Array,
    third: Array,
    extras: Vec<(String, Array)>,
    stream: &Stream,
) {
    let first = first.as_type::<f32>(stream).unwrap();
    let second = second.as_type::<f32>(stream).unwrap();
    let third = third.as_type::<f32>(stream).unwrap();
    let mut arrays = vec![
        ("prefill".into(), prefill),
        ("decode".into(), decode),
        ("decode_second".into(), decode_second),
        ("first_token".into(), first),
        ("second_token".into(), second),
        ("third_token".into(), third),
    ];
    arrays.extend(extras);
    save_arrays(path, &arrays);
}

fn save_qwen_expected(model: &mut qwen3::Model, path: &Path, stream: &Stream) {
    let prompt = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
    let prefill = model
        .forward(
            qwen3::ModelInput {
                inputs: &prompt,
                mask: None,
                cache: &mut cache,
            },
            stream,
        )
        .unwrap();
    let first = greedy_token(&prefill, stream);
    let decode = model
        .forward(
            qwen3::ModelInput {
                inputs: &first,
                mask: None,
                cache: &mut cache,
            },
            stream,
        )
        .unwrap();
    let second = greedy_token(&decode, stream);
    let decode_second = model
        .forward(
            qwen3::ModelInput {
                inputs: &second,
                mask: None,
                cache: &mut cache,
            },
            stream,
        )
        .unwrap();
    let third = greedy_token(&decode_second, stream);

    let mut sliding_cache = vec![Some(SlidingKeyValueCache::new(2))];
    let sliding_prefill = model
        .forward(
            qwen3::ModelInput {
                inputs: &prompt,
                mask: None,
                cache: &mut sliding_cache,
            },
            stream,
        )
        .unwrap();
    let sliding_first = greedy_token(&sliding_prefill, stream);
    let sliding_decode = model
        .forward(
            qwen3::ModelInput {
                inputs: &sliding_first,
                mask: None,
                cache: &mut sliding_cache,
            },
            stream,
        )
        .unwrap();
    let sliding_second = greedy_token(&sliding_decode, stream);
    let sliding_decode_second = model
        .forward(
            qwen3::ModelInput {
                inputs: &sliding_second,
                mask: None,
                cache: &mut sliding_cache,
            },
            stream,
        )
        .unwrap();
    save_expected(
        path,
        prefill,
        decode,
        decode_second,
        first,
        second,
        third,
        vec![
            ("sliding_prefill".into(), sliding_prefill),
            ("sliding_decode".into(), sliding_decode),
            ("sliding_decode_second".into(), sliding_decode_second),
            (
                "sliding_first_token".into(),
                sliding_first.as_type::<f32>(stream).unwrap(),
            ),
            (
                "sliding_second_token".into(),
                sliding_second.as_type::<f32>(stream).unwrap(),
            ),
        ],
        stream,
    );
}

fn save_deepseek_expected(model: &mut deepseek_v3::Model, path: &Path, stream: &Stream) {
    let prompt = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
    let mut cache = model.new_cache();
    let prefill = model
        .forward(
            deepseek_v3::ModelInput {
                inputs: &prompt,
                mask: None,
                cache: Some(&mut cache),
            },
            stream,
        )
        .unwrap();
    let first = greedy_token(&prefill, stream);
    let decode = model
        .forward(
            deepseek_v3::ModelInput {
                inputs: &first,
                mask: None,
                cache: Some(&mut cache),
            },
            stream,
        )
        .unwrap();
    let second = greedy_token(&decode, stream);
    let decode_second = model
        .forward(
            deepseek_v3::ModelInput {
                inputs: &second,
                mask: None,
                cache: Some(&mut cache),
            },
            stream,
        )
        .unwrap();
    let third = greedy_token(&decode_second, stream);
    save_expected(
        path,
        prefill,
        decode,
        decode_second,
        first,
        second,
        third,
        Vec::new(),
        stream,
    );
}

fn write_qwen_fixture(directory: &Path, packed_directory: &Path) {
    let config = r#"{
      "model_type":"qwen3_moe","hidden_size":32,"num_hidden_layers":1,
      "intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,
      "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":16,
      "max_position_embeddings":64,"rope_theta":10000.0,
      "tie_word_embeddings":false,"rope_scaling":null,
      "moe_intermediate_size":32,"num_experts":4,
      "num_experts_per_tok":2,"norm_topk_prob":true
    }"#;
    std::fs::write(directory.join("config.json"), config).unwrap();
    std::fs::write(packed_directory.join("config.json"), config).unwrap();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = qwen3::get_qwen3_model_args(directory).unwrap();
    let mut model = qwen3::Model::new(args, stream).unwrap();
    initialize_parameters(&mut model, stream);
    save_arrays(
        &packed_directory.join("model.safetensors"),
        &model
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect::<Vec<_>>(),
    );
    split_qwen_checkpoint(&model, directory, stream);
    save_qwen_expected(&mut model, &directory.join("expected.safetensors"), stream);
    let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
    let mut quantized =
        qwen3::load_qwen3_model_quantized(packed_directory, quantization, stream, stream).unwrap();
    save_qwen_expected(
        &mut quantized,
        &packed_directory.join("expected-affine.safetensors"),
        stream,
    );
}

fn write_deepseek_fixture(directory: &Path) {
    let config = r#"{
      "model_type":"deepseek_v3","hidden_size":32,"intermediate_size":64,
      "moe_intermediate_size":32,"num_hidden_layers":3,"num_attention_heads":2,
      "vocab_size":16,"rms_norm_eps":0.000001,"max_position_embeddings":64,
      "rope_theta":10000,"q_lora_rank":null,"kv_lora_rank":32,
      "qk_nope_head_dim":16,"qk_rope_head_dim":8,"v_head_dim":16,
      "first_k_dense_replace":1,"moe_layer_freq":1,"n_routed_experts":4,
      "n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,
      "topk_group":1,"topk_method":"noaux_tc","scoring_func":"sigmoid",
      "norm_topk_prob":true,"routed_scaling_factor":1.0,
      "num_nextn_predict_layers":0,"tie_word_embeddings":false
    }"#;
    std::fs::write(directory.join("config.json"), config).unwrap();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = deepseek_v3::get_model_args(directory).unwrap();
    let mut model = deepseek_v3::Model::new(args.clone(), stream).unwrap();
    for layer in &mut model.model.layers {
        if let deepseek_v3::FeedForward::Moe(moe) = &mut layer.mlp {
            moe.experts.gate_proj = Param::new(Some(
                Array::zeros::<f32>(&[4, args.moe_intermediate_size, args.hidden_size], stream)
                    .unwrap(),
            ));
            moe.experts.up_proj = Param::new(Some(
                Array::zeros::<f32>(&[4, args.moe_intermediate_size, args.hidden_size], stream)
                    .unwrap(),
            ));
            moe.experts.down_proj = Param::new(Some(
                Array::zeros::<f32>(&[4, args.hidden_size, args.moe_intermediate_size], stream)
                    .unwrap(),
            ));
        }
    }
    initialize_parameters(&mut model, stream);
    split_deepseek_checkpoint(&model, directory, stream);

    save_deepseek_expected(&mut model, &directory.join("expected.safetensors"), stream);
    let quantization = WeightQuantization::Affine(AffineQuantization::new(32, 4).unwrap());
    let mut quantized =
        deepseek_v3::load_model_quantized(directory, quantization, stream, stream).unwrap();
    save_deepseek_expected(
        &mut quantized,
        &directory.join("expected-affine.safetensors"),
        stream,
    );
}

fn initialize_deepseek_fp8_parameters(model: &mut deepseek_v3::Model, stream: &Stream) {
    let experts = model.args.n_routed_experts;
    let hidden = model.args.hidden_size;
    let intermediate = model.args.moe_intermediate_size;
    for layer in &mut model.model.layers {
        if let deepseek_v3::FeedForward::Moe(moe) = &mut layer.mlp {
            for parameter in [&mut moe.experts.gate_proj, &mut moe.experts.up_proj] {
                *parameter = Param::new(Some(
                    Array::from_slice(
                        &vec![0x38u8; (experts * intermediate * hidden) as usize],
                        &[experts, intermediate, hidden],
                    )
                    .copy(stream)
                    .unwrap(),
                ));
            }
            moe.experts.down_proj = Param::new(Some(
                Array::from_slice(
                    &vec![0x38u8; (experts * hidden * intermediate) as usize],
                    &[experts, hidden, intermediate],
                )
                .copy(stream)
                .unwrap(),
            ));
            for parameter in [
                &mut moe.experts.gate_proj_scale_inv,
                &mut moe.experts.up_proj_scale_inv,
                &mut moe.experts.down_proj_scale_inv,
            ] {
                *parameter = Param::new(Some(
                    Array::from_slice(&vec![0.01f32; experts as usize], &[experts, 1, 1])
                        .copy(stream)
                        .unwrap(),
                ));
            }
        }
    }
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        *parameter = if parameter.dtype() == Dtype::Uint8 {
            Array::from_slice(
                &vec![0x38u8; shape.iter().map(|value| *value as usize).product()],
                &shape,
            )
            .copy(stream)
            .unwrap()
        } else if name.ends_with("weight_scale_inv") || name.ends_with("_scale_inv") {
            Array::from_slice(
                &vec![0.01f32; shape.iter().map(|value| *value as usize).product()],
                &shape,
            )
            .copy(stream)
            .unwrap()
        } else {
            patterned_parameter(&name, &shape, stream)
        };
    }
}

fn write_deepseek_fp8_fixture(directory: &Path) {
    let config = r#"{
      "model_type":"deepseek_v3","hidden_size":32,"intermediate_size":64,
      "moe_intermediate_size":32,"num_hidden_layers":3,"num_attention_heads":2,
      "vocab_size":16,"rms_norm_eps":0.000001,"max_position_embeddings":64,
      "rope_theta":10000,"q_lora_rank":null,"kv_lora_rank":32,
      "qk_nope_head_dim":16,"qk_rope_head_dim":8,"v_head_dim":16,
      "first_k_dense_replace":1,"moe_layer_freq":1,"n_routed_experts":4,
      "n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,
      "topk_group":1,"topk_method":"noaux_tc","scoring_func":"sigmoid",
      "norm_topk_prob":true,"routed_scaling_factor":1.0,
      "num_nextn_predict_layers":0,"tie_word_embeddings":false,
      "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3","quant_method":"fp8","weight_block_size":[128,128]}
    }"#;
    std::fs::write(directory.join("config.json"), config).unwrap();
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let args = deepseek_v3::get_model_args(directory).unwrap();
    let mut model = deepseek_v3::Model::new(args, stream).unwrap();
    initialize_deepseek_fp8_parameters(&mut model, stream);
    save_deepseek_expected(
        &mut model,
        &directory.join("expected-fp8.safetensors"),
        stream,
    );
    split_deepseek_checkpoint(&model, directory, stream);
}

fn initialize_zero_fixture(model: &mut impl ModuleParameters, stream: &Stream) {
    for (name, parameter) in model.parameters_mut().flatten() {
        let shape = parameter.shape().to_vec();
        let dtype = parameter.dtype();
        *parameter = if name.ends_with("_scales") {
            Array::full::<u8>(&shape, Array::from_slice(&[127u8], &[]), stream).unwrap()
        } else if name.ends_with("norm.weight")
            || name.ends_with("layernorm.weight")
            || name.ends_with("global_scale")
            || name.as_ref() == "model.norm_f.weight"
        {
            ones_dtype(&shape, dtype, stream).unwrap()
        } else {
            zeros_dtype(&shape, dtype, stream).unwrap()
        };
    }
}

fn save_zero_fixture(
    model: &mut impl ModuleParameters,
    config: &serde_json::Value,
    directory: &Path,
    stream: &Stream,
    output_vocab: i32,
) {
    initialize_zero_fixture(model, stream);
    let arrays = model
        .parameters()
        .flatten()
        .into_iter()
        .map(|(name, value)| {
            (
                safemlx_lm::module_binding::canonical_checkpoint_name(&name),
                value.clone(),
            )
        })
        .collect::<Vec<_>>();
    save_arrays(&directory.join("model.safetensors"), &arrays);
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let zero_token = zeros_dtype(&[1, 1], Dtype::Float32, stream).unwrap();
    save_arrays(
        &directory.join("expected.safetensors"),
        &[
            (
                "prefill".into(),
                zeros_dtype(&[1, 3, output_vocab], Dtype::Float32, stream).unwrap(),
            ),
            (
                "decode".into(),
                zeros_dtype(&[1, 1, output_vocab], Dtype::Float32, stream).unwrap(),
            ),
            (
                "decode_second".into(),
                zeros_dtype(&[1, 1, output_vocab], Dtype::Float32, stream).unwrap(),
            ),
            ("first_token".into(), zero_token.clone()),
            ("second_token".into(), zero_token.clone()),
            ("third_token".into(), zero_token),
        ],
    );
}

fn write_additional_sparse_fixtures(root: &Path) -> Vec<(&'static str, &'static str, PathBuf)> {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut fixtures = Vec::new();

    let config = serde_json::json!({
        "model_type": "gpt_oss", "hidden_size": 32, "intermediate_size": 32,
        "moe_intermediate_size": 32, "num_hidden_layers": 2, "test_moe_layers": 2,
        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 32,
        "vocab_size": 32, "num_local_experts": 4, "num_experts_per_tok": 2,
        "rms_norm_eps": 1e-5, "sliding_window": 4, "max_position_embeddings": 64,
        "rope_theta": 150000.0, "layer_types": ["sliding_attention", "full_attention"],
        "quantization_config": {"quant_method": "mxfp4"}, "swiglu_limit": 7.0
    });
    let directory = root.join("gpt-oss-sparse");
    std::fs::create_dir_all(&directory).unwrap();
    let mut model =
        gpt_oss::Model::new(serde_json::from_value(config.clone()).unwrap(), stream).unwrap();
    save_zero_fixture(&mut model, &config, &directory, stream, 32);
    fixtures.push(("GPT-OSS sparse expert cache", "GptOss", directory));

    let config = serde_json::json!({
        "model_type": "lfm2_moe", "vocab_size": 32, "hidden_size": 16,
        "intermediate_size": 24, "moe_intermediate_size": 8, "num_hidden_layers": 2,
        "test_moe_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64, "norm_eps": 1e-5,
        "layer_types": ["full_attention", "full_attention"], "conv_L_cache": 3,
        "conv_bias": false, "block_auto_adjust_ff_dim": false,
        "tie_word_embeddings": false, "num_dense_layers": 0, "num_experts": 4,
        "num_experts_per_tok": 2, "norm_topk_prob": true, "use_expert_bias": true
    });
    let directory = root.join("lfm2-sparse");
    std::fs::create_dir_all(&directory).unwrap();
    let mut model =
        lfm2::Model::new(serde_json::from_value(config.clone()).unwrap(), stream).unwrap();
    save_zero_fixture(&mut model, &config, &directory, stream, 32);
    fixtures.push(("LFM2 sparse expert cache", "Lfm2", directory));

    let config = serde_json::json!({
        "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 8,
        "intermediate_size": 12, "moe_intermediate_size": 6,
        "moe_shared_expert_intermediate_size": 10, "num_hidden_layers": 2,
        "test_moe_layers": 1, "hybrid_override_pattern": "E*", "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "max_position_embeddings": 64,
        "mamba_num_heads": 2, "mamba_head_dim": 4, "n_groups": 1,
        "ssm_state_size": 4, "conv_kernel": 3, "chunk_size": 2,
        "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "tie_word_embeddings": false, "torch_dtype": "float32"
    });
    let directory = root.join("nemotron-h-sparse");
    std::fs::create_dir_all(&directory).unwrap();
    let mut model =
        nemotron_h::Model::new(serde_json::from_value(config.clone()).unwrap(), stream).unwrap();
    save_zero_fixture(&mut model, &config, &directory, stream, 32);
    fixtures.push(("Nemotron-H sparse expert cache", "NemotronH", directory));

    let config = serde_json::json!({
        "model_type": "inkling_mm_model", "hidden_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "test_moe_layers": 2, "eos_token_id": 1,
        "text_config": {
            "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 8,
            "sliding_window_size": 4, "local_layer_ids": [0], "dense_mlp_idx": 0,
            "sconv_kernel_size": 3, "d_rel": 4, "rel_extent": 8,
            "intermediate_size": 8, "dense_intermediate_size": 16,
            "moe_intermediate_size": 8, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "n_shared_experts": 1, "route_scale": 1.0, "use_sconv": true,
            "use_embed_norm": true, "shared_expert_sink": true, "use_gate_bias": true,
            "norm_after_topk": true, "use_global_scale": true, "gate_activation": "sigmoid",
            "hidden_act": "silu", "attention_dropout": 0.0, "q_bias": false,
            "o_bias": false, "logits_mup_width_multiplier": 2.0
        }
    });
    let directory = root.join("inkling-sparse");
    std::fs::create_dir_all(&directory).unwrap();
    let mut model =
        inkling::Model::new(serde_json::from_value(config.clone()).unwrap(), stream).unwrap();
    save_zero_fixture(&mut model, &config, &directory, stream, 32);
    fixtures.push(("Inkling sparse expert cache", "Inkling", directory));

    for (next, label, architecture) in [
        (true, "Qwen3-Next sparse expert cache", "Qwen3Next"),
        (false, "Qwen3.5 sparse expert cache", "Qwen35Moe"),
    ] {
        let config = serde_json::json!({
            "model_type": if next { "qwen3_next" } else { "qwen3_5_moe_text" },
            "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 2,
            "test_moe_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "max_position_embeddings": 64, "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false, "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "intermediate_size": 0, "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8, "num_experts_per_tok": 2,
            "num_experts": 4, "norm_topk_prob": true,
            "layer_types": ["full_attention", "full_attention"]
        });
        let directory = root.join(if next {
            "qwen3-next-sparse"
        } else {
            "qwen35-sparse"
        });
        std::fs::create_dir_all(&directory).unwrap();
        let mut model = qwen3_5_moe::Model::new(
            serde_json::from_value(config.clone()).unwrap(),
            None,
            None,
            None,
            stream,
        )
        .unwrap();
        save_zero_fixture(&mut model, &config, &directory, stream, 32);
        fixtures.push((label, architecture, directory));
    }

    let config = serde_json::json!({
        "model_type": "qwen3_vl_moe", "hidden_size": 12, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "test_moe_layers": 2, "image_token_id": 30,
        "video_token_id": 31, "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_vl_moe_text", "hidden_size": 12, "num_hidden_layers": 2,
            "intermediate_size": 24, "num_attention_heads": 1, "rms_norm_eps": 1e-6,
            "vocab_size": 32, "num_key_value_heads": 1, "max_position_embeddings": 128,
            "rope_theta": 10000.0, "head_dim": 12, "tie_word_embeddings": true,
            "moe_intermediate_size": 8, "num_experts": 4, "num_experts_per_tok": 2,
            "norm_topk_prob": true, "rope_scaling": {
                "rope_type": "default", "mrope_interleaved": true, "mrope_section": [2, 2, 2]
            }
        },
        "vision_config": {
            "depth": 1, "hidden_size": 8, "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16, "num_heads": 2, "num_position_embeddings": 16,
            "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "window_size": 8, "out_hidden_size": 12,
            "fullatt_block_indexes": [0], "deepstack_visual_indexes": []
        }
    });
    let directory = root.join("qwen3-vl-moe-sparse");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let args = qwen3_vl::get_qwen3_vl_model_args(&directory).unwrap();
    let mut model = qwen3_vl::Model::new(args, stream).unwrap();
    save_zero_fixture(&mut model, &config, &directory, stream, 32);
    fixtures.push(("Qwen3-VL-MoE sparse expert cache", "Qwen3VlMoe", directory));

    fixtures
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

fn reserve_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

fn run_ring_fixture(
    label: &str,
    architecture: &str,
    encoding: &str,
    assignment: &str,
    residency: &str,
    checkpoint: &Path,
    expected_file: &str,
) {
    let (first, second, first_port, second_port) = reserve_ports();
    let ring = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();
    drop(first);
    drop(second);
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::with_capacity(2));
    for rank in 0..2 {
        children.0.push(
            Command::new(&executable)
                .args([
                    "--exact",
                    "expert_parallel_model_ring_worker",
                    "--nocapture",
                ])
                .env(WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env(CHECKPOINT_DIR, checkpoint)
                .env(EXPECTED_FILE, checkpoint.join(expected_file))
                .env(ARCHITECTURE, architecture)
                .env(ENCODING, encoding)
                .env(ASSIGNMENT, assignment)
                .env(RESIDENCY, residency)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(90);
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
        .map(|(rank, output)| {
            format!(
                "{label} EP Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{label} two-process model parity failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}

/// Runs complete-model versus EP=2 dense, affine-packed, and native-FP8 parity,
/// including round-robin packed and explicit split-expert ownership.
///
/// Every case checks prefill, two cached decode steps, three synchronized
/// tokens, local ownership, and routed-bank memory. DeepSeek additionally
/// covers a dense-to-MoE transition and grouped noaux routing.
///
/// Run with:
/// `cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_two_process_model_parity -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local Ring workers and opens loopback sockets"]
fn ring_two_process_model_parity() {
    assert!(distributed::is_available(Backend::Ring));
    let fixture = tempfile::tempdir().unwrap();
    let qwen = fixture.path().join("qwen3-moe");
    let qwen_packed = fixture.path().join("qwen3-moe-packed");
    let deepseek = fixture.path().join("deepseek-v3");
    let deepseek_fp8 = fixture.path().join("deepseek-v3-fp8");
    std::fs::create_dir_all(&qwen).unwrap();
    std::fs::create_dir_all(&qwen_packed).unwrap();
    std::fs::create_dir_all(&deepseek).unwrap();
    std::fs::create_dir_all(&deepseek_fp8).unwrap();
    write_qwen_fixture(&qwen, &qwen_packed);
    write_deepseek_fixture(&deepseek);
    write_deepseek_fp8_fixture(&deepseek_fp8);
    run_ring_fixture(
        "Qwen3 dense",
        "Qwen3",
        "dense",
        "balanced",
        "resident",
        &qwen,
        "expected.safetensors",
    );
    run_ring_fixture(
        "DeepSeekV3 dense-to-MoE grouped",
        "DeepSeekV3",
        "dense",
        "balanced",
        "resident",
        &deepseek,
        "expected.safetensors",
    );
    run_ring_fixture(
        "Qwen3 affine packed experts",
        "Qwen3",
        "affine",
        "balanced",
        "resident",
        &qwen_packed,
        "expected-affine.safetensors",
    );
    run_ring_fixture(
        "DeepSeekV3 affine packed experts",
        "DeepSeekV3",
        "affine",
        "balanced",
        "resident",
        &deepseek,
        "expected-affine.safetensors",
    );
    run_ring_fixture(
        "DeepSeekV3 native block-FP8 experts",
        "DeepSeekV3",
        "fp8",
        "balanced",
        "resident",
        &deepseek_fp8,
        "expected-fp8.safetensors",
    );
    run_ring_fixture(
        "Qwen3 affine packed round-robin experts",
        "Qwen3",
        "affine",
        "round-robin",
        "resident",
        &qwen_packed,
        "expected-affine.safetensors",
    );
    run_ring_fixture(
        "DeepSeekV3 explicit experts",
        "DeepSeekV3",
        "dense",
        "explicit",
        "resident",
        &deepseek,
        "expected.safetensors",
    );
    run_ring_fixture(
        "Qwen3 sparse expert cache",
        "Qwen3",
        "dense",
        "balanced",
        "sparse-cache",
        &qwen,
        "expected.safetensors",
    );
    for (label, architecture, checkpoint) in write_additional_sparse_fixtures(fixture.path()) {
        run_ring_fixture(
            label,
            architecture,
            "dense",
            "balanced",
            "sparse-cache",
            &checkpoint,
            "expected.safetensors",
        );
    }
}

/// Verifies rank-local paged prompt save/load parity for the two EP cache
/// representations supported by persistence: DeepSeek MLA and GPT-OSS KV.
#[test]
#[ignore = "spawns local Ring workers and opens loopback sockets"]
fn ring_two_process_paged_prompt_cache_parity() {
    assert!(distributed::is_available(Backend::Ring));
    let fixture = tempfile::tempdir().unwrap();

    let deepseek = fixture.path().join("deepseek-v3-prompt");
    std::fs::create_dir_all(&deepseek).unwrap();
    write_deepseek_fixture(&deepseek);
    std::fs::write(deepseek.join(PAGED_PROMPT_MARKER), b"1").unwrap();
    run_ring_fixture(
        "DeepSeekV3 paged prompt cache",
        "DeepSeekV3",
        "dense",
        "balanced",
        "resident",
        &deepseek,
        "expected.safetensors",
    );

    let gpt = write_additional_sparse_fixtures(fixture.path())
        .into_iter()
        .find(|(_, architecture, _)| *architecture == "GptOss")
        .map(|(_, _, directory)| directory)
        .unwrap();
    std::fs::write(gpt.join(PAGED_PROMPT_MARKER), b"1").unwrap();
    run_ring_fixture(
        "GPT-OSS paged prompt cache",
        "GptOss",
        "dense",
        "balanced",
        "sparse-cache",
        &gpt,
        "expected.safetensors",
    );
}
