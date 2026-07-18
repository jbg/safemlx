#![cfg(unix)]

use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    module::Param,
    transforms::eval,
    Array, Device, DeviceType, Stream,
};
use safemlx_lm::{
    expert_parallel::{
        all_to_all_v, dispatch_sharded, profile_expert_parallel_timings, ExpertAssignment,
        ShardedRouteBlocks,
    },
    models::{common::moe::PackedRelu2Experts, deepseek_v3::RoutedExperts},
};
use safetensors::tensor::{serialize_to_file, Dtype as TensorDtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_EXPERT_EXCHANGE_RING_WORKER";
const PAYLOAD_FILE: &str = "SAFEMLX_LM_EXPERT_EXCHANGE_PAYLOAD";

fn f32_array(values: &[f32], shape: &[i32], stream: &Stream) -> Array {
    Array::from_slice(values, shape).copy(stream).unwrap()
}

fn i32_array(values: &[i32], shape: &[i32], stream: &Stream) -> Array {
    Array::from_slice(values, shape).copy(stream).unwrap()
}

fn u8_array(values: &[u8], shape: &[i32], stream: &Stream) -> Array {
    Array::from_slice(values, shape).copy(stream).unwrap()
}

fn assert_f32_close(actual: &Array, expected: &[f32]) {
    eval([actual]).unwrap();
    let actual = actual.evaluated().unwrap();
    assert_eq!(actual.as_slice::<f32>().len(), expected.len());
    for (index, (actual, expected)) in actual.as_slice::<f32>().iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "dispatch output {index} was {actual}, expected {expected}"
        );
    }
}

fn full_dispatch_blocks(rank: usize, stream: &Stream) -> ShardedRouteBlocks {
    let (hidden, global_expert_ids, original_route_indices, weights) = if rank == 0 {
        (
            [vec![2.0, 1.0], vec![1.0, 2.0]],
            [vec![1, 0], vec![3, 2]],
            [vec![3, 1], vec![0, 2]],
            [vec![0.2, 0.25], vec![0.5, 0.1]],
        )
    } else {
        (
            [vec![4.0, 3.0], vec![3.0, 4.0]],
            [vec![0, 1], vec![2, 3]],
            [vec![2, 0], vec![1, 3]],
            [vec![0.25, 0.2], vec![0.5, 0.1]],
        )
    };
    ShardedRouteBlocks {
        hidden: hidden
            .iter()
            .map(|values| f32_array(values, &[2, 1], stream))
            .collect(),
        global_expert_ids: global_expert_ids
            .iter()
            .map(|values| i32_array(values, &[2], stream))
            .collect(),
        original_route_indices: original_route_indices
            .iter()
            .map(|values| i32_array(values, &[2], stream))
            .collect(),
        weights: weights
            .iter()
            .map(|values| f32_array(values, &[2], stream))
            .collect(),
        top_k: 2,
        source_tokens: 2,
    }
}

fn relu2_bank(stream: &Stream) -> PackedRelu2Experts {
    PackedRelu2Experts {
        num_experts: 2,
        hidden_size: 1,
        intermediate_size: 1,
        up_proj: Param::new(f32_array(&[1.0, 2.0], &[2, 1, 1], stream)),
        down_proj: Param::new(f32_array(&[1.0, 10.0], &[2, 1, 1], stream)),
    }
}

fn fp8_bank(stream: &Stream) -> RoutedExperts {
    let weights = u8_array(&[0x38, 0x38], &[2, 1, 1], stream);
    let scales = f32_array(&[1.0, 2.0], &[2, 1, 1], stream);
    RoutedExperts {
        num_experts: 2,
        intermediate_size: 1,
        use_fp8: true,
        gate_affine: None,
        up_affine: None,
        down_affine: None,
        gate_proj: Param::new(Some(weights.clone())),
        gate_proj_scale_inv: Param::new(Some(scales.clone())),
        gate_proj_scales: Param::new(None),
        gate_proj_biases: Param::new(None),
        up_proj: Param::new(Some(weights.clone())),
        up_proj_scale_inv: Param::new(Some(scales.clone())),
        up_proj_scales: Param::new(None),
        up_proj_biases: Param::new(None),
        down_proj: Param::new(Some(weights)),
        down_proj_scale_inv: Param::new(Some(scales)),
        down_proj_scales: Param::new(None),
        down_proj_biases: Param::new(None),
    }
}

fn fp8_route_output(hidden: f32, local_expert: usize, weight: f32) -> f32 {
    let scale = if local_expert == 0 { 1.0 } else { 2.0 };
    let gate = scale * hidden;
    let silu = gate / (1.0 + (-gate).exp());
    weight * silu * (scale * hidden) * scale
}

#[test]
fn expert_exchange_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let group = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(group.rank(), expected_rank);
    assert_eq!(group.size(), 2);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let arrays = Array::load_safetensors(std::env::var_os(PAYLOAD_FILE).unwrap(), &stream).unwrap();
    let blocks = (0..2)
        .map(|destination| arrays[&format!("r{expected_rank}d{destination}")].clone())
        .collect::<Vec<_>>();
    let _profiling = profile_expert_parallel_timings();
    let exchanged = all_to_all_v(&blocks, &group, &stream).unwrap();
    eval([&exchanged.received]).unwrap();
    let received = exchanged.received.evaluated().unwrap();
    if expected_rank == 0 {
        assert_eq!(exchanged.source_counts, vec![1, 0]);
        assert_eq!(received.as_slice::<i32>(), &[10]);
    } else {
        assert_eq!(exchanged.source_counts, vec![2, 1]);
        assert_eq!(received.as_slice::<i32>(), &[11, 12, 21]);
    }
    assert_eq!(exchanged.statistics.padding_routes, 4);
    assert_eq!(exchanged.statistics.exchanged_bytes, 32);
    assert_eq!(exchanged.statistics.synchronization_count, 1);
    assert!(exchanged.statistics.exchange_time > Duration::ZERO);
    assert_eq!(
        exchanged.statistics.total_time,
        exchanged.statistics.exchange_time
    );

    let assignment = ExpertAssignment::balanced(4, 2, expected_rank).unwrap();
    let mut relu2 = relu2_bank(&stream);
    let dispatched = dispatch_sharded(
        full_dispatch_blocks(expected_rank, &stream),
        &assignment,
        &mut relu2,
        &group,
        &stream,
    )
    .unwrap();
    if expected_rank == 0 {
        assert_f32_close(&dispatched.output, &[20.25, 32.4]);
    } else {
        assert_f32_close(&dispatched.output, &[76.5, 68.0]);
    }
    assert_eq!(dispatched.statistics.total_routes, 4);
    assert_eq!(dispatched.statistics.sent_routes, 4);
    assert_eq!(dispatched.statistics.received_routes, 4);
    assert_eq!(dispatched.statistics.synchronization_count, 6);

    let empty_hidden = f32_array(&[], &[0, 1], &stream);
    let empty_i32 = i32_array(&[], &[0], &stream);
    let empty_f32 = f32_array(&[], &[0], &stream);
    let empty_blocks = if expected_rank == 0 {
        ShardedRouteBlocks {
            hidden: vec![empty_hidden.clone(), empty_hidden.clone()],
            global_expert_ids: vec![empty_i32.clone(), empty_i32.clone()],
            original_route_indices: vec![empty_i32.clone(), empty_i32.clone()],
            weights: vec![empty_f32.clone(), empty_f32.clone()],
            top_k: 2,
            source_tokens: 1,
        }
    } else {
        ShardedRouteBlocks {
            hidden: vec![empty_hidden, f32_array(&[2.0], &[1, 1], &stream)],
            global_expert_ids: vec![empty_i32.clone(), i32_array(&[2], &[1], &stream)],
            original_route_indices: vec![empty_i32, i32_array(&[1], &[1], &stream)],
            weights: vec![empty_f32, f32_array(&[0.5], &[1], &stream)],
            top_k: 2,
            source_tokens: 1,
        }
    };
    let empty_dispatched =
        dispatch_sharded(empty_blocks, &assignment, &mut relu2, &group, &stream).unwrap();
    assert_f32_close(
        &empty_dispatched.output,
        if expected_rank == 0 { &[0.0] } else { &[2.0] },
    );
    assert_eq!(
        empty_dispatched.statistics.total_routes,
        usize::from(expected_rank == 1)
    );
    assert_eq!(
        empty_dispatched.statistics.received_routes,
        usize::from(expected_rank == 1)
    );
    assert_eq!(empty_dispatched.statistics.synchronization_count, 6);

    let mut fp8 = fp8_bank(&stream);
    let fp8_dispatched = dispatch_sharded(
        full_dispatch_blocks(expected_rank, &stream),
        &assignment,
        &mut fp8,
        &group,
        &stream,
    )
    .unwrap();
    let expected = if expected_rank == 0 {
        [
            fp8_route_output(1.0, 1, 0.5) + fp8_route_output(1.0, 0, 0.25),
            fp8_route_output(2.0, 0, 0.1) + fp8_route_output(2.0, 1, 0.2),
        ]
    } else {
        [
            fp8_route_output(3.0, 1, 0.2) + fp8_route_output(3.0, 0, 0.5),
            fp8_route_output(4.0, 0, 0.25) + fp8_route_output(4.0, 1, 0.1),
        ]
    };
    assert_f32_close(&fp8_dispatched.output, &expected);
    assert_eq!(fp8_dispatched.statistics.total_routes, 4);
    assert_eq!(fp8_dispatched.statistics.sent_routes, 4);
    assert_eq!(fp8_dispatched.statistics.received_routes, 4);
    assert_eq!(fp8_dispatched.statistics.synchronization_count, 6);
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

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_expert_exchange_ring ring_two_process_all_to_all_v_and_dispatch_sharded -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_all_to_all_v_and_dispatch_sharded() {
    assert!(distributed::is_available(Backend::Ring));
    let fixture = tempfile::tempdir().unwrap();
    let payload = fixture.path().join("payload.safetensors");
    let i32_bytes = |values: &[i32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let r0d0 = i32_bytes(&[10]);
    let r0d1 = i32_bytes(&[11, 12]);
    let r1d0 = Vec::<u8>::new();
    let r1d1 = i32_bytes(&[21]);
    serialize_to_file(
        [
            (
                "r0d0",
                TensorView::new(TensorDtype::I32, vec![1, 1], &r0d0).unwrap(),
            ),
            (
                "r0d1",
                TensorView::new(TensorDtype::I32, vec![2, 1], &r0d1).unwrap(),
            ),
            (
                "r1d0",
                TensorView::new(TensorDtype::I32, vec![0, 1], &r1d0).unwrap(),
            ),
            (
                "r1d1",
                TensorView::new(TensorDtype::I32, vec![1, 1], &r1d1).unwrap(),
            ),
        ],
        None,
        &payload,
    )
    .unwrap();
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
                .args(["--exact", "expert_exchange_ring_worker", "--nocapture"])
                .env(WORKER_RANK, rank.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env(PAYLOAD_FILE, &payload)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(60);
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
                "exchange Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process all-to-all-v and sharded dispatch failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );
}
