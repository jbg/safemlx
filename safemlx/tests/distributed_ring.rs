#![cfg(unix)]

use std::{
    net::TcpListener,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    Array, Device, DeviceType, Stream,
};

const WORKER_ENV: &str = "SAFEMLX_RING_TEST_WORKER";

#[test]
fn ring_worker() {
    let Some(expected_rank) = std::env::var_os(WORKER_ENV) else {
        return;
    };
    let expected_rank: usize = expected_rank.to_string_lossy().parse().unwrap();

    let group = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!(group.rank(), expected_rank);
    assert_eq!(group.size(), 2);

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::arange::<_, f32>(
        Some(expected_rank as i32),
        expected_rank as i32 + 2,
        None::<i32>,
        &stream,
    )
    .unwrap();
    let sum = distributed::all_sum(&input, &group, &stream).unwrap();
    let sum = sum.evaluated().unwrap();
    assert_eq!(sum.as_slice::<f32>(), &[1.0, 3.0]);

    let start = expected_rank as i32 * 10;
    let input = Array::arange::<_, i32>(Some(start), start + 2, None::<i32>, &stream).unwrap();
    let gathered = distributed::all_gather(&input, &group, &stream).unwrap();
    let gathered = gathered.evaluated().unwrap();
    assert_eq!(gathered.as_array().shape(), &[4]);
    assert_eq!(gathered.as_slice::<i32>(), &[0, 1, 10, 11]);
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
        "Ring rank {rank} exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Run with:
/// `cargo test -p safemlx --test distributed_ring ring_two_process_loopback -- --ignored --exact --nocapture`
#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_loopback() {
    assert!(distributed::is_available(Backend::Ring));

    let (_first_socket, _second_socket, first_port, second_port) = reserve_two_ports();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("ring-hosts.json");
    std::fs::write(
        &hostfile,
        format!("[[\"127.0.0.1:{first_port}\"],[\"127.0.0.1:{second_port}\"]]"),
    )
    .unwrap();

    // Release the reservations immediately before launching both ranks. The
    // kernel-selected ports make collisions unlikely, while concurrent launch
    // is required because each Ring rank waits for its peer during init.
    drop(_first_socket);
    drop(_second_socket);

    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard {
        children: Vec::with_capacity(2),
    };
    for rank in 0..2 {
        let child = Command::new(&executable)
            .args(["--exact", "ring_worker", "--nocapture"])
            .env(WORKER_ENV, rank.to_string())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        children.children.push(child);
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut timed_out = false;
    loop {
        let statuses: Vec<_> = children
            .children
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if statuses.iter().flatten().any(|status| !status.success()) || timed_out {
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
    let failures: Vec<_> = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| render_failure(rank, output))
        .collect();
    assert!(
        failures.is_empty() && !timed_out,
        "two-process Ring integration test failed:\n{}",
        if timed_out {
            format!("timed out after 30 seconds\n\n{}", failures.join("\n\n"))
        } else {
            failures.join("\n\n")
        }
    );
}
