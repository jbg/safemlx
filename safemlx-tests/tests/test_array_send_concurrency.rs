use std::{
    sync::{Arc, Barrier},
    thread,
};

use safemlx::{
    error::Exception, transforms::compile::compile, Array, Device, DeviceType, EvaluatedArray,
    Stream,
};

const THREADS: usize = 8;
const ITERS: usize = 50;
const LEN: usize = 256;

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn array_send_marker_allows_owned_transfer() {
    assert_send::<Array>();
}

#[test]
fn array_sync_marker_allows_shared_references() {
    assert_sync::<Array>();
    assert_sync::<EvaluatedArray<'static>>();
}

#[test]
fn owned_array_can_move_to_another_thread_and_drop() {
    let mut handles = Vec::new();
    for i in 0..(THREADS * ITERS) {
        let array = Array::from_slice(&[i as i32, (i + 1) as i32], &[2]);
        handles.push(thread::spawn(move || drop(array)));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn owned_array_can_move_to_another_thread_and_evaluate() {
    let mut handles = Vec::new();
    for i in 0..THREADS {
        let data = vec![i as f32 + 1.0; LEN];
        let array = Array::from_slice(&data, &[LEN as i32]);
        handles.push(thread::spawn(move || {
            let stream = cpu_stream();
            let value = array
                .square(&stream)
                .unwrap()
                .sum(None, &stream)
                .unwrap()
                .item::<f32>(&stream);
            let expected = (i as f32 + 1.0).powi(2) * LEN as f32;
            assert!((value - expected).abs() < 0.01, "{value} != {expected}");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn cloned_arrays_can_be_used_concurrently_on_distinct_streams() {
    let base = Array::from_slice(&vec![1.0f32; LEN], &[LEN as i32]);
    let mut handles = Vec::new();

    for i in 0..THREADS {
        let array = base.clone();
        handles.push(thread::spawn(move || {
            let stream = cpu_stream();
            let offset = Array::from_f32(i as f32);
            let expected = (1.0 + i as f32).powi(2) * LEN as f32;

            for _ in 0..ITERS {
                let value = array
                    .add(&offset, &stream)
                    .unwrap()
                    .square(&stream)
                    .unwrap()
                    .sum(None, &stream)
                    .unwrap()
                    .item::<f32>(&stream);
                assert!((value - expected).abs() < 0.01, "{value} != {expected}");
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn arrays_can_be_evaluated_concurrently_for_host_reads() {
    let mut handles = Vec::new();

    for i in 0..THREADS {
        let data = vec![i as i32; LEN];
        handles.push(thread::spawn(move || {
            let evaluated = Array::from_slice(&data, &[LEN as i32])
                .into_evaluated()
                .unwrap();
            for _ in 0..ITERS {
                assert_eq!(evaluated.as_slice::<i32>(), &data[..]);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn compile_paths_are_stable_under_parallel_threads() {
    let mut handles = Vec::new();

    for i in 0..THREADS {
        handles.push(thread::spawn(move || {
            let stream: &'static Stream = Box::leak(Box::new(cpu_stream()));
            let f = move |x: &Array| -> Result<Array, Exception> {
                x.square(stream)?.sum(None, stream)
            };
            let mut compiled = compile(f, None);
            let expected = (i as f32 + 1.0).powi(2) * LEN as f32;

            for _ in 0..ITERS {
                let data = vec![i as f32 + 1.0; LEN];
                let input = Array::from_slice(&data, &[LEN as i32]);
                let value = compiled(&input).unwrap().item::<f32>(stream);
                assert!((value - expected).abs() < 0.01, "{value} != {expected}");
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn shared_lazy_array_can_be_evaluated_concurrently() {
    let data = vec![7i32; LEN];
    let array = Arc::new(Array::from_slice(&data, &[LEN as i32]));
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();

    for _ in 0..THREADS {
        let array = Arc::clone(&array);
        let barrier = Arc::clone(&barrier);
        let expected = data.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..ITERS {
                let evaluated = array.evaluated().unwrap();
                assert_eq!(evaluated.as_slice::<i32>(), &expected[..]);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn shared_array_metadata_reads_are_stable_while_evaluating() {
    let stream = cpu_stream();
    let array = Array::from_slice(&vec![2.0f32; LEN], &[LEN as i32])
        .square(&stream)
        .unwrap();
    let array = Arc::new(array);
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();

    {
        let array = Arc::clone(&array);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..ITERS {
                let evaluated = array.evaluated().unwrap();
                assert_eq!(evaluated.as_slice::<f32>(), &vec![4.0f32; LEN][..]);
            }
        }));
    }

    for _ in 1..THREADS {
        let array = Arc::clone(&array);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..(ITERS * 10) {
                assert_eq!(array.shape(), &[LEN as i32]);
                assert_eq!(array.ndim(), 1);
                assert_eq!(array.size(), LEN);
                assert_eq!(array.dtype(), safemlx::Dtype::Float32);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn shared_evaluated_array_supports_concurrent_host_reads() {
    let data = vec![11i32; LEN];
    let evaluated = Arc::new(
        Array::from_slice(&data, &[LEN as i32])
            .into_evaluated()
            .unwrap(),
    );
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();

    for _ in 0..THREADS {
        let evaluated = Arc::clone(&evaluated);
        let barrier = Arc::clone(&barrier);
        let expected = data.clone();
        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..(ITERS * 10) {
                assert_eq!(evaluated.as_slice::<i32>(), &expected[..]);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn shared_input_array_can_feed_concurrent_ops_on_distinct_streams() {
    let array = Arc::new(Array::from_slice(&vec![1.0f32; LEN], &[LEN as i32]));
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();

    for i in 0..THREADS {
        let array = Arc::clone(&array);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let stream = cpu_stream();
            let offset = Array::from_f32(i as f32);
            let expected = (1.0 + i as f32).powi(2) * LEN as f32;

            barrier.wait();
            for _ in 0..ITERS {
                let value = array
                    .add(&offset, &stream)
                    .unwrap()
                    .square(&stream)
                    .unwrap()
                    .sum(None, &stream)
                    .unwrap()
                    .item::<f32>(&stream);
                assert!((value - expected).abs() < 0.01, "{value} != {expected}");
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn shared_array_clone_drop_churn_is_stable_while_reading() {
    let stream = cpu_stream();
    let array = Array::from_slice(&vec![3.0f32; LEN], &[LEN as i32])
        .multiply(&Array::from_f32(2.0), &stream)
        .unwrap();
    let array = Arc::new(array);
    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();

    for i in 0..THREADS {
        let array = Arc::clone(&array);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier.wait();
            if i % 2 == 0 {
                for _ in 0..(ITERS * 20) {
                    drop(array.clone());
                }
            } else {
                let stream = cpu_stream();
                for _ in 0..ITERS {
                    let value = array.sum(None, &stream).unwrap().item::<f32>(&stream);
                    let expected = 6.0 * LEN as f32;
                    assert!((value - expected).abs() < 0.01, "{value} != {expected}");
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
