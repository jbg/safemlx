//! Tests for the optimizers. These tests are placed here because the models
//! used for testing make use of `ModuleParameter` macro.

use std::collections::HashMap;

use safemlx::{
    array, assert_array_eq,
    builder::Builder,
    losses::{LossReduction, MseLossBuilder},
    macros::ModuleParameters,
    module::{FlattenedModuleParam, Module, ModuleParameters, Param},
    nn,
    ops::{full, ones, zeros},
    optimizers::{
        AdaDelta, AdaGrad, AdafactorBuilder, Adam, AdamW, Adamax, Lion, LionBuilder, Optimizer,
        RmsProp, RmsPropBuilder, Sgd, SgdBuilder,
    },
    random::{uniform, RandomState},
    transforms::{eval, eval_params},
    Array, Dtype, Stream,
};

mod common;

use common::*;

/* -------------------------------------------------------------------------- */
/*                              Convergence tests                             */
/* -------------------------------------------------------------------------- */

pub fn train<F, O>(f: F, steps: usize) -> Result<Array, Box<dyn std::error::Error>>
where
    F: FnOnce() -> O,
    O: Optimizer,
{
    let stream = test_stream();
    let mut rng = RandomState::with_seed(0)?;
    let mut optimizer = f();

    let mse_loss = MseLossBuilder::new()
        .reduction(LossReduction::Mean)
        .build()?;
    let loss = |model: &mut LinearFunctionModel, (x, y): (&Array, &Array)| {
        mse_loss.apply(model.forward(x, stream)?, y, stream)
    };

    // TODO: check compiled model once we have it
    let mut model = LinearFunctionModel::new(None, stream)?;
    eval_params(model.parameters())?;

    let m = array!(0.25);
    let b = array!(7.0);

    let mut lg = nn::value_and_grad(loss);

    let mut last_loss = None;
    for _ in 0..steps {
        // println!("target: b = {}, m = {}", b, m);
        // println!("parameters: {:?}", model.parameters());

        // generate random training data along with the ground truth.
        // notice that the shape is [B, 1] where B is the batch
        // dimension -- this allows us to train on 10 samples simultaneously
        let x_key = rng.next_key(stream)?;
        let x = uniform::<_, f32>(-5.0, 5.0, &[10, 1], &x_key, stream)?;
        let y = m.multiply(&x, stream)?.add(&b, stream)?;
        eval([&x, &y])?;

        // compute the loss and gradients.  use the optimizer
        // to adjust the parameters closer to the target
        let (loss, g) = lg(&mut model, (&x, &y))?;
        optimizer.update(&mut model, g, stream)?;

        eval_params(model.parameters())?;

        last_loss = Some(loss);
    }

    Ok(last_loss.unwrap())
}

const NUM_TRIALS: usize = 3;

#[test]
fn test_sgd_converges() {
    let stream = test_stream();
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        let loss = train(|| Sgd::new(0.1), 30).unwrap();
        total_loss += loss.item::<f32>(&stream);
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {avg_loss}");
}

#[test]
fn test_rmsprop_converges() {
    let stream = test_stream();
    let mut total_loss = 0.0;
    for _ in 0..NUM_TRIALS {
        // RMSProp doesn't seem to converge as fast as SGD
        let loss = train(|| RmsProp::new(0.1).unwrap(), 100).unwrap();
        total_loss += loss.item::<f32>(&stream);
    }
    // It sometimes doesn't converge that fast, so we take the average loss
    // across multiple trials
    let avg_loss = total_loss / NUM_TRIALS as f32;
    assert!(avg_loss < 0.1, "avg loss: {avg_loss}");
}

/* -------------------------------------------------------------------------- */
/*                            Optimizer unit tests                            */
/* -------------------------------------------------------------------------- */

#[derive(Debug, ModuleParameters)]
struct SimpleModel {
    #[param]
    a: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct First {
    #[param]
    pub a: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

#[derive(Debug, ModuleParameters)]
struct NestedModel {
    #[param]
    pub first: First,

    #[param]
    pub second: Param<Array>,
}

type GradsMap = FlattenedModuleParam;

fn next_normal(rng: &mut RandomState, shape: &[i32], stream: &Stream) -> Array {
    let key = rng.next_key(stream).unwrap();
    safemlx::random::normal::<f32>(shape, None, None, &key, stream).unwrap()
}

fn next_uniform(rng: &mut RandomState, shape: &[i32], stream: &Stream) -> Array {
    let key = rng.next_key(stream).unwrap();
    safemlx::random::uniform::<_, f32>(0.0, 1.0, shape, &key, stream).unwrap()
}

fn constant(shape: &[i32], value: f32, stream: &Stream) -> Array {
    full::<f32>(shape, array!(value), stream).unwrap()
}

fn assert_save_and_load<O>(optimizer: O, new_optimizer: O) -> Result<(), Box<dyn std::error::Error>>
where
    O: Optimizer,
{
    use safemlx::optimizers::OptimizerState;

    let tmp_dir = tempfile::tempdir()?;
    let path = tmp_dir.path().join("optimizer.safetensors");

    optimizer.state().save_safetensors(&path)?;

    let mut loaded_optimizer = new_optimizer;
    loaded_optimizer
        .state_mut()
        .load_safetensors(&path, test_stream())?;

    let original_state: HashMap<_, _> = optimizer.state().flatten().collect();
    let loaded_state: HashMap<_, _> = loaded_optimizer.state().flatten().collect();

    assert!(!loaded_state.is_empty());
    assert_eq!(original_state.len(), loaded_state.len());
    for (key, original) in original_state {
        let loaded = loaded_state
            .get(&key)
            .unwrap_or_else(|| panic!("missing optimizer state key {key}"));
        assert!(eval_equal_values(original, loaded));
    }

    Ok(())
}

fn create_default_test_model_and_grads(stream: &Stream) -> (NestedModel, GradsMap) {
    let first = First {
        a: Param::new(zeros::<f32>(&[10], stream).unwrap()),
        b: Param::new(zeros::<f32>(&[1], stream).unwrap()),
    };
    let model = NestedModel {
        first,
        second: Param::new(zeros::<f32>(&[1], stream).unwrap()),
    };

    let grads_map: GradsMap = model
        .parameters()
        .flatten()
        .iter()
        .map(|(k, v)| {
            let g = ones::<f32>(v.shape(), stream).unwrap();
            (k.clone(), g)
        })
        .collect();

    (model, grads_map)
}

const ATOL: f64 = 1e-5;

// This unit test is adapted from the swift binding unit test `testAdaDelta` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_ada_delta() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(547).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), safemlx::Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.348_337_02),
        0.006966740489006043,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-4.180_044),
        0.08360088348388672,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), safemlx::Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.522_678_4),
        0.010453567504882813,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(6.272_14),
        0.12544280052185058,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdaDelta::new(0.1).unwrap();

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();

    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), safemlx::Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.348_442_4),
        0.348442405462265,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-4.181_308_7),
        0.08362617492675782,
        stream = stream
    );

    assert_save_and_load(optimizer, AdaDelta::new(0.1).unwrap()).unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdaGrad` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adagrad() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(958).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.045_843_333),
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-0.550_12),
        ATOL,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.232_503_94),
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(2.790_047_2),
        ATOL,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdaGrad::new(0.1);

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.062_509_984),
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-0.750_119_8),
        ATOL,
        stream = stream
    );

    assert_save_and_load(optimizer, AdaGrad::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdam` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adam() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(616).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(0.112_293_06),
        0.002245861142873764,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(1.347_516_7),
        0.02695033311843872,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.305_597_72),
        0.0061119544506073,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(3.667_172_7),
        0.0733434534072876,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adam::new(0.1);

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(0.112_292_78),
        0.0022458556294441224,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(1.347_513_3),
        0.026950266361236572,
        stream = stream
    );

    assert_save_and_load(optimizer, Adam::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testAdamW` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_adamw() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(696).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.363_391_88),
        0.007267837524414063,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-4.360_702_5),
        0.08721405029296875,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.221_754_48),
        0.0044350895285606385,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(2.661_053_7),
        0.05322107315063477,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdamW::new(0.1);

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.468_437_6),
        0.009368752241134645,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-5.621_251),
        0.11242502212524415,
        stream = stream
    );

    assert_save_and_load(optimizer, AdamW::new(0.1)).unwrap();
}

// This unit test is adapted from the python unit test `test_adamax` in
// `mlx/python/tests/test_optimizers.py`.
#[test]
fn test_adamax() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(75).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.303_923_6),
        0.006078472137451172,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-3.647_083_3),
        0.07294166564941407,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(-0.242_717_24),
        0.004854344725608826,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(-2.912_606_7),
        0.05825213432312012,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Adamax::new(0.1);

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.303_923_6),
        0.006078472137451172,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-3.647_083_3),
        0.07294166564941407,
        stream = stream
    );

    assert_save_and_load(optimizer, Adamax::new(0.1)).unwrap();
}

// This unit test is adapted from the python unit test `test_rmsprop` in
// `tests/test_optimizer.py`.
#[test]
fn test_rmsprop() {
    const LR: f32 = 1e-2;
    const ALPHA: f32 = 0.99;
    let stream = test_stream();

    let (mut model, gradients) = create_default_test_model_and_grads(stream);

    let mut optim = RmsPropBuilder::new(LR).alpha(ALPHA).build().unwrap();
    optim.update(&mut model, gradients, stream).unwrap();

    let expected_first_a = constant(&[10], -0.1, stream);
    let expected_first_b = constant(&[1], -0.1, stream);
    let expected_second = constant(&[1], -0.1, stream);

    assert_array_eq!(
        model.first.a.as_ref(),
        expected_first_a,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        model.first.b.as_ref(),
        expected_first_b,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        model.second.as_ref(),
        expected_second,
        ATOL,
        stream = stream
    );

    let expected_state_first_a = constant(&[10], 0.01, stream);
    let expected_state_first_b = constant(&[1], 0.01, stream);
    let expected_state_second = constant(&[1], 0.01, stream);

    assert_array_eq!(
        optim.state.get("first.a").unwrap(),
        expected_state_first_a,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        optim.state.get("first.b").unwrap(),
        expected_state_first_b,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        optim.state.get("second").unwrap(),
        expected_state_second,
        ATOL,
        stream = stream
    );

    assert_save_and_load(optim, RmsPropBuilder::new(LR).alpha(ALPHA).build().unwrap()).unwrap();
}

// This unit test is adapted from the python unit test `test_sgd` in
// `mlx/python/tests/test_optimizers.py`
#[test]
fn test_sgd() {
    let stream = test_stream();
    let (mut model, gradients) = create_default_test_model_and_grads(stream);

    let mut optim = SgdBuilder::new(1e-2).momentum(0.9).build().unwrap();
    optim.update(&mut model, gradients, stream).unwrap();

    let expected_first_a = constant(&[10], -0.01, stream);
    let expected_first_b = constant(&[1], -0.01, stream);
    let expected_second = constant(&[1], -0.01, stream);

    assert_array_eq!(
        model.first.a.as_ref(),
        expected_first_a,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        model.first.b.as_ref(),
        expected_first_b,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        model.second.as_ref(),
        expected_second,
        ATOL,
        stream = stream
    );

    let expected_state_first_a = ones::<f32>(&[10], stream).unwrap();
    let expected_state_first_b = ones::<f32>(&[1], stream).unwrap();
    let expected_state_second = ones::<f32>(&[1], stream).unwrap();

    assert_array_eq!(
        optim.state["first.a"].as_ref(),
        expected_state_first_a,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        optim.state["first.b"].as_ref(),
        expected_state_first_b,
        ATOL,
        stream = stream
    );
    assert_array_eq!(
        optim.state["second"].as_ref(),
        expected_state_second,
        ATOL,
        stream = stream
    );
}

// This unit test is adapted from the swift binding unit test `testLion` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_lion() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(27).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(0.177_692_23),
        0.003553844690322876,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(2.132_306_8),
        0.042646136283874515,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(-0.021_187_237),
        0.00042374473065137863,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(-0.254_246_83),
        0.005084936618804932,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = Lion::new(0.1);

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(0.211_025_57),
        0.004220511317253113,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(2.532_306_7),
        0.05064613342285156,
        stream = stream
    );

    assert_save_and_load(optimizer, Lion::new(0.1)).unwrap();
}

// This unit test is adapted from the swift binding unit test `testLion1` in
// `mlx-swift/Tests/MLXTests/IntegrationTests.swift`
#[test]
fn test_lion1() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(127).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.184_610_6),
        0.0036922121047973633,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-2.215_327_3),
        0.04430654525756836,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(-0.036_004_007),
        0.0007200801372528076,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(-0.432_048_08),
        0.008640961647033691,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = LionBuilder::new(0.1).weight_decay(0.1).build().unwrap();

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.182_764_5),
        0.003655290007591248,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-2.193_174),
        0.04386347770690918,
        stream = stream
    );

    assert_save_and_load(
        optimizer,
        LionBuilder::new(0.1).weight_decay(0.1).build().unwrap(),
    )
    .unwrap();
}

#[test]
fn test_adafactor() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(650).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(-0.520_713_7),
        0.010414273738861083,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(-6.248_564),
        0.12497127532958985,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.433_303_65),
        0.008666073083877564,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(5.199_643_6),
        0.10399287223815919,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).build().unwrap();

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    println!(
        "a_model.a.mean(None, stream).unwrap(): {:?}",
        a_model.a.mean(None, stream).unwrap()
    );
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(-0.526_828_47),
        0.010536569356918336,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(-6.321_941_4),
        0.12643882751464844,
        stream = stream
    );

    assert_save_and_load(optimizer, AdafactorBuilder::new().lr(0.1).build().unwrap()).unwrap();
}

#[test]
fn test_adafactor1() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(193).unwrap();
    let a = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a.shape(), &[4, 3]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(0.400_818_17),
        0.008016363382339478,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(4.809_818),
        0.09619635581970215,
        stream = stream
    );

    let a_grad = next_normal(&mut rng, &[4, 3], stream);
    assert_eq!(a_grad.shape(), &[4, 3]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.214_474_72),
        0.004289494454860688,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(2.573_696_6),
        0.05147393226623535,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).beta1(0.1).build().unwrap();

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[4, 3]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(0.399_430_7),
        0.007988613843917847,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(4.793_168),
        0.09586336135864258,
        stream = stream
    );
}

#[test]
fn test_adafactor2() {
    let stream = test_stream();
    let mut rng = RandomState::with_seed(620).unwrap();
    let a = next_uniform(&mut rng, &[10], stream);
    assert_eq!(a.shape(), &[10]);
    assert_eq!(a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a.mean(None, stream).unwrap(),
        array!(0.489_024_55),
        0.00978049099445343,
        stream = stream
    );
    assert_array_eq!(
        a.sum(None, stream).unwrap(),
        array!(4.890_245_4),
        0.09780490875244141,
        stream = stream
    );

    let a_grad = next_uniform(&mut rng, &[10], stream);
    assert_eq!(a_grad.shape(), &[10]);
    assert_eq!(a_grad.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_grad.mean(None, stream).unwrap(),
        array!(0.681_890_2),
        0.013637803792953491,
        stream = stream
    );
    assert_array_eq!(
        a_grad.sum(None, stream).unwrap(),
        array!(6.818_902),
        0.1363780403137207,
        stream = stream
    );

    let mut a_model = SimpleModel {
        a: Param::new(a.clone()),
    };
    let mut a_grad_params = FlattenedModuleParam::new();
    a_grad_params.insert("a".into(), a_grad.clone());

    let mut optimizer = AdafactorBuilder::new().lr(0.1).build().unwrap();

    optimizer
        .update(&mut a_model, a_grad_params, stream)
        .unwrap();
    assert_eq!(a_model.a.shape(), &[10]);
    assert_eq!(a_model.a.dtype(), Dtype::Float32);
    assert_array_eq!(
        a_model.a.mean(None, stream).unwrap(),
        array!(0.483_533_05),
        0.009670661091804504,
        stream = stream
    );
    assert_array_eq!(
        a_model.a.sum(None, stream).unwrap(),
        array!(4.835_330_5),
        0.09670660972595214,
        stream = stream
    );
}
