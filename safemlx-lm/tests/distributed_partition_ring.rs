#![cfg(unix)]

use std::{
    net::TcpListener,
    path::Path,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    Device, DeviceType, Stream,
};
use safemlx_lm::{
    parallel::load_safetensors_partition, weights::StrictLoadConfig, DeviceAssignment,
    ParallelTopology, PlacementPlan, TensorPlacement,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "SAFEMLX_LM_PARTITION_RING_WORKER";
const CHECKPOINT_DIR: &str = "SAFEMLX_LM_PARTITION_CHECKPOINT";

#[test]
fn partition_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = std::env::var_os(CHECKPOINT_DIR).unwrap();
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);

    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let mut plan = PlacementPlan::new(topology);
    plan.insert_expected(
        "projection.weight",
        vec![2, 4],
        TensorPlacement::Shard {
            axis: 1,
            index: topology.tensor_parallel_rank,
            parts: topology.tensor_parallel_size,
        },
    )
    .unwrap();
    plan.insert("remote.weight", TensorPlacement::Omit);
    let config = StrictLoadConfig::default().strip_prefix("model.");
    let partition =
        load_safetensors_partition(Path::new(&checkpoint), &plan, &stream, &config).unwrap();
    assert_eq!(partition.len(), 1);
    assert_eq!(partition.opened_shards().len(), 1);
    assert_eq!(
        partition.opened_shards()[0].file_name().unwrap(),
        "local.safetensors"
    );

    let local = partition.get("projection.weight").unwrap();
    assert_eq!(local.shape(), &[2, 2]);
    let gathered = distributed::all_gather(local, &group, &stream).unwrap();
    let gathered = gathered.evaluated().unwrap();
    assert_eq!(gathered.as_array().shape(), &[4, 2]);
    assert_eq!(gathered.as_slice::<i32>(), &[0, 1, 10, 11, 2, 3, 12, 13]);
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

fn reserve_two_ports() -> (TcpListener, TcpListener, u16, u16) {
    let first = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let second = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let first_port = first.local_addr().unwrap().port();
    let second_port = second.local_addr().unwrap().port();
    (first, second, first_port, second_port)
}

fn render_failure(rank: usize, output: &Output) -> String {
    format!(
        "partition Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx-lm --test distributed_partition_ring ring_two_process_partition_load -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_partition_load() {
    assert!(distributed::is_available(Backend::Ring));

    let checkpoint = tempfile::tempdir().unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let bytes = [0i32, 1, 2, 3, 10, 11, 12, 13]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::I32, vec![2, 4], &bytes).unwrap();
    serialize_to_file(
        [("model.projection.weight", view)],
        None,
        &checkpoint.path().join("local.safetensors"),
    )
    .unwrap();
    std::fs::write(
        checkpoint.path().join("remote.safetensors"),
        b"this remote-only shard must not be opened",
    )
    .unwrap();
    std::fs::write(
        checkpoint.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.projection.weight": "local.safetensors",
                "model.remote.weight": "remote.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();

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
        children.children.push(
            Command::new(&executable)
                .args(["--exact", "partition_ring_worker", "--nocapture"])
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

    let deadline = Instant::now() + Duration::from_secs(30);
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

    let failures = children
        .finish()
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process rank-aware loading failed (timed_out={timed_out}):\n{}",
        failures.join("\n\n")
    );

    stream.synchronize().unwrap();
}
