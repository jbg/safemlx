//! Canonical checkpoint bindings for unloaded module parameter trees.
//!
//! These helpers keep checkpoint-name expansion, shape validation, byte
//! accounting, and resident-lease assignment independent of model families.

use std::collections::{BTreeMap, BTreeSet};

use safemlx::module::ModuleParameters;

use crate::{
    residency::{ResidentUnitLease, WeightBinding},
    weight_recipe::{DerivedWeightRecipe, RecipeDtype},
    weight_store::{TensorSelection, WeightStore},
};

/// Converts a module parameter name to its canonical checkpoint spelling.
///
/// Quantized modules wrap their packed matrix (and ordinary biased modules
/// wrap their bias) in an `inner` module. MLX-compatible checkpoints omit that
/// implementation detail while retaining companion `.scales` and `.biases`
/// tensors unchanged.
pub fn canonical_checkpoint_name(parameter_name: &str) -> String {
    parameter_name
        .replace(".inner.weight", ".weight")
        .replace(".inner.bias", ".bias")
}

/// Returns the full parameter names exposed by `module` under `prefix`.
pub fn full_parameter_names(module: &impl ModuleParameters, prefix: &str) -> Vec<String> {
    let mut names = module
        .parameters()
        .flatten()
        .keys()
        .map(|name| qualify(prefix, name))
        .collect::<Vec<_>>();
    names.sort();
    names
}

/// Builds exact full-tensor residency bindings for an unloaded module.
///
/// Every module parameter must resolve to exactly one checkpoint key and have
/// the same shape. Binding names are local module parameter names so a lease
/// can later populate a freshly constructed module without architecture-aware
/// rewriting.
pub fn build_module_bindings(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    build_module_bindings_excluding(module, prefix, store, |_| false)
}

/// Builds exact bindings for non-excluded local module parameters.
///
/// The predicate receives module-local flattened names and runs before any
/// checkpoint lookup, allowing independently managed parameter groups to use a
/// different checkpoint layout.
pub fn build_module_bindings_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    build_module_bindings_with_recipes_excluding(module, prefix, store, BTreeMap::new(), exclude)
}

/// Builds module bindings while replacing selected local parameters with recipes.
///
/// Recipe keys use the module-local flattened parameter names. Every override
/// is shape- and dtype-checked against the unloaded runtime parameter before
/// residency initialization.
pub fn build_module_bindings_with_recipes(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    recipes: BTreeMap<String, DerivedWeightRecipe>,
) -> Result<Vec<WeightBinding>, ModuleBindingError> {
    build_module_bindings_with_recipes_excluding(module, prefix, store, recipes, |_| false)
}

fn build_module_bindings_with_recipes_excluding<F>(
    module: &impl ModuleParameters,
    prefix: &str,
    store: &dyn WeightStore,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
    exclude: F,
) -> Result<Vec<WeightBinding>, ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let params = module.parameters().flatten();
    let mut local_names = params
        .keys()
        .map(ToString::to_string)
        .filter(|name| !exclude(name))
        .collect::<Vec<_>>();
    local_names.sort();
    let mut claimed = BTreeMap::<String, String>::new();
    let mut bindings = Vec::with_capacity(local_names.len());

    for local_name in local_names {
        let parameter = params
            .get(local_name.as_str())
            .expect("parameter name came from the same flattened tree");
        if let Some(recipe) = recipes.remove(&local_name) {
            let metadata = recipe.infer(store)?;
            let expected_shape = parameter
                .shape()
                .iter()
                .map(|&dimension| usize::try_from(dimension))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ModuleBindingError::InvalidModuleShape {
                    parameter: qualify(prefix, &local_name),
                    shape: parameter.shape().to_vec(),
                })?;
            if metadata.shape() != expected_shape {
                return Err(ModuleBindingError::RecipeShapeMismatch {
                    parameter: qualify(prefix, &local_name),
                    expected: expected_shape,
                    actual: metadata.shape().to_vec(),
                });
            }
            let expected_dtype = RecipeDtype::from(parameter.dtype());
            if !recipe_dtype_matches(&expected_dtype, metadata.dtype()) {
                return Err(ModuleBindingError::RecipeDtypeMismatch {
                    parameter: qualify(prefix, &local_name),
                    expected: expected_dtype,
                    actual: metadata.dtype().clone(),
                });
            }
            bindings.push(WeightBinding::from_recipe(
                local_name,
                recipe,
                metadata.byte_len(),
            )?);
            continue;
        }
        let destination = qualify(prefix, &local_name);
        let canonical = canonical_checkpoint_name(&destination);
        let checkpoint_key = if keys.contains(&destination) {
            destination.clone()
        } else if keys.contains(&canonical) {
            canonical
        } else {
            return Err(ModuleBindingError::MissingParameter { destination });
        };

        if let Some(previous) = claimed.insert(checkpoint_key.clone(), destination.clone()) {
            return Err(ModuleBindingError::DuplicateCheckpointBinding {
                checkpoint_key,
                first: previous,
                second: destination,
            });
        }

        let metadata = store.metadata(&checkpoint_key)?;
        let expected_shape = parameter
            .shape()
            .iter()
            .map(|&dimension| usize::try_from(dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ModuleBindingError::InvalidModuleShape {
                parameter: destination.clone(),
                shape: parameter.shape().to_vec(),
            })?;
        if metadata.shape != expected_shape {
            return Err(ModuleBindingError::ShapeMismatch {
                checkpoint_key,
                parameter: destination,
                expected: expected_shape,
                actual: metadata.shape,
            });
        }
        let expected_bytes = u64::try_from(metadata.logical_byte_len).map_err(|_| {
            ModuleBindingError::ArithmeticOverflow {
                context: "checkpoint tensor byte length",
            }
        })?;
        bindings.push(WeightBinding::new(
            local_name,
            metadata.name,
            TensorSelection::Full,
            expected_bytes,
        )?);
    }

    if !recipes.is_empty() {
        return Err(ModuleBindingError::UnknownRecipeParameters {
            parameters: recipes.into_keys().collect(),
        });
    }

    Ok(bindings)
}

/// Assigns every module parameter from a protected resident unit.
///
/// `Array::clone` only clones the MLX handle; it does not copy the resident
/// allocation. The caller must therefore keep `lease` alive through forward
/// execution and synchronize before releasing it.
pub fn populate_module_from_lease(
    module: &mut impl ModuleParameters,
    lease: &ResidentUnitLease,
) -> Result<(), ModuleBindingError> {
    populate_module_from_lease_excluding(module, lease, |_| false)
}

/// Assigns non-excluded module parameters from a protected resident unit.
pub fn populate_module_from_lease_excluding<F>(
    module: &mut impl ModuleParameters,
    lease: &ResidentUnitLease,
    excluded: F,
) -> Result<(), ModuleBindingError>
where
    F: Fn(&str) -> bool,
{
    let resident_names = lease.binding_names().collect::<BTreeSet<_>>();
    let mut params = module.parameters_mut().flatten();
    let expected_names = params
        .keys()
        .filter(|name| !excluded(name))
        .map(|name| name.to_string())
        .collect::<BTreeSet<_>>();

    let missing = expected_names
        .iter()
        .filter(|name| !resident_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected = resident_names
        .iter()
        .filter(|name| !expected_names.contains(**name))
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(ModuleBindingError::LeaseContents {
            unit: lease.id().to_string(),
            missing,
            unexpected,
        });
    }

    for (name, parameter) in &mut params {
        if excluded(name) {
            continue;
        }
        let value = lease.array(name)?;
        if parameter.shape() != value.shape() {
            return Err(ModuleBindingError::ResidentShapeMismatch {
                unit: lease.id().to_string(),
                parameter: name.to_string(),
                expected: parameter.shape().to_vec(),
                actual: value.shape().to_vec(),
            });
        }
        **parameter = value.clone();
    }
    Ok(())
}

/// Returns the checked total byte count of a binding collection.
pub fn binding_bytes(bindings: &[WeightBinding]) -> Result<u64, ModuleBindingError> {
    bindings.iter().try_fold(0u64, |total, binding| {
        total
            .checked_add(binding.expected_bytes())
            .ok_or(ModuleBindingError::ArithmeticOverflow {
                context: "module binding byte total",
            })
    })
}

fn qualify(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

fn recipe_dtype_matches(expected: &RecipeDtype, actual: &RecipeDtype) -> bool {
    expected == actual || matches!((expected, actual), (RecipeDtype::U8, RecipeDtype::F8E4M3))
}

/// Structured module-to-checkpoint binding failures.
#[derive(Debug, thiserror::Error)]
pub enum ModuleBindingError {
    /// A recipe override did not name a runtime parameter.
    #[error("derived-weight recipes name unknown local parameters: {parameters:?}")]
    UnknownRecipeParameters {
        /// Unknown local parameter names.
        parameters: Vec<String>,
    },
    /// A recipe output shape differed from its runtime placeholder.
    #[error("derived weight for {parameter:?} has shape {actual:?}, expected {expected:?}")]
    RecipeShapeMismatch {
        /// Fully qualified runtime parameter.
        parameter: String,
        /// Runtime placeholder shape.
        expected: Vec<usize>,
        /// Recipe output shape.
        actual: Vec<usize>,
    },
    /// A recipe output dtype differed from its runtime placeholder.
    #[error("derived weight for {parameter:?} has dtype {actual:?}, expected {expected:?}")]
    RecipeDtypeMismatch {
        /// Fully qualified runtime parameter.
        parameter: String,
        /// Runtime placeholder dtype.
        expected: RecipeDtype,
        /// Recipe output dtype.
        actual: RecipeDtype,
    },
    /// A module parameter had no matching checkpoint tensor.
    #[error("checkpoint is missing module parameter {destination:?}")]
    MissingParameter {
        /// Full module parameter name.
        destination: String,
    },
    /// Two parameters resolved to one checkpoint tensor.
    #[error("checkpoint tensor {checkpoint_key:?} resolves to both {first:?} and {second:?}")]
    DuplicateCheckpointBinding {
        /// Ambiguous checkpoint tensor.
        checkpoint_key: String,
        /// First module parameter.
        first: String,
        /// Second module parameter.
        second: String,
    },
    /// A module placeholder exposed an invalid dimension.
    #[error("module parameter {parameter:?} has invalid shape {shape:?}")]
    InvalidModuleShape {
        /// Full parameter name.
        parameter: String,
        /// Invalid MLX shape.
        shape: Vec<i32>,
    },
    /// Checkpoint and unloaded-module shapes differed.
    #[error("checkpoint tensor {checkpoint_key:?} for {parameter:?} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Source checkpoint key.
        checkpoint_key: String,
        /// Destination module parameter.
        parameter: String,
        /// Unloaded module shape.
        expected: Vec<usize>,
        /// Checkpoint shape.
        actual: Vec<usize>,
    },
    /// A resident unit did not exactly match the module parameter tree.
    #[error("resident unit {unit} cannot populate module: missing {missing:?}, unexpected {unexpected:?}")]
    LeaseContents {
        /// Resident unit identifier.
        unit: String,
        /// Expected module parameters absent from the lease.
        missing: Vec<String>,
        /// Lease bindings absent from the module.
        unexpected: Vec<String>,
    },
    /// A resident array no longer matched its unloaded placeholder.
    #[error(
        "resident unit {unit} parameter {parameter:?} has shape {actual:?}, expected {expected:?}"
    )]
    ResidentShapeMismatch {
        /// Resident unit identifier.
        unit: String,
        /// Local module parameter.
        parameter: String,
        /// Unloaded module shape.
        expected: Vec<i32>,
        /// Resident array shape.
        actual: Vec<i32>,
    },
    /// Checked accounting overflowed.
    #[error("module binding arithmetic overflow: {context}")]
    ArithmeticOverflow {
        /// Failed calculation.
        context: &'static str,
    },
    /// Persistent checkpoint inspection failed.
    #[error(transparent)]
    WeightStore(#[from] crate::weight_store::WeightStoreError),
    /// Derived-weight metadata validation failed.
    #[error(transparent)]
    WeightRecipe(#[from] crate::weight_recipe::WeightRecipeError),
    /// Residency binding or lookup failed.
    #[error(transparent)]
    Residency(#[from] crate::residency::ResidencyError),
}

#[cfg(test)]
mod tests {
    use safemlx::{
        module::ModuleParameters, nn, Array, Device, DeviceType, Dtype, ExecutionContext,
    };

    use super::*;
    use crate::{
        models::common::linear::unloaded_maybe_quantized_linear, quantization::AffineQuantization,
        weight_store::SafetensorsWeightStore,
    };

    fn cpu() -> ExecutionContext {
        ExecutionContext::new(Device::new(DeviceType::Cpu, 0))
    }

    #[test]
    fn packed_e4m3_recipe_matches_u8_runtime_storage_only() {
        assert!(recipe_dtype_matches(&RecipeDtype::U8, &RecipeDtype::F8E4M3));
        assert!(!recipe_dtype_matches(
            &RecipeDtype::U8,
            &RecipeDtype::F8E5M2
        ));
        assert!(!recipe_dtype_matches(
            &RecipeDtype::F32,
            &RecipeDtype::F8E4M3
        ));
    }

    #[test]
    fn dense_binding_validates_names_shapes_and_exact_bytes() {
        let context = cpu();
        let linear = nn::Linear::unloaded(2, 3, true, Dtype::Float32, context.stream()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(&[0.1f32; 6], &[3, 2]);
        let bias = Array::from_slice(&[0.2f32; 3], &[3]);
        Array::save_safetensors(
            [("proj.weight", &weight), ("proj.bias", &bias)],
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let bindings = build_module_bindings(&linear, "proj", &store).unwrap();
        assert_eq!(bindings.len(), 2);
        assert_eq!(binding_bytes(&bindings).unwrap(), 36);
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.checkpoint_key())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["proj.bias", "proj.weight"])
        );
    }

    #[test]
    fn quantized_binding_preserves_packed_companions() {
        let context = cpu();
        let quantization = AffineQuantization::new(16, 4).unwrap().into();
        let module =
            unloaded_maybe_quantized_linear(32, 2, true, Some(quantization), context.stream())
                .unwrap();
        let params = module.parameters().flatten();
        let arrays = params
            .iter()
            .map(|(name, value)| (canonical_checkpoint_name(&format!("proj.{name}")), *value))
            .collect::<Vec<_>>();
        let dir = tempfile::tempdir().unwrap();
        Array::save_safetensors(
            arrays.iter().map(|(name, value)| (name.as_str(), *value)),
            None,
            dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let bindings = build_module_bindings(&module, "proj", &store).unwrap();
        let keys = bindings
            .iter()
            .map(|binding| binding.checkpoint_key())
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("proj.weight"));
        assert!(keys.contains("proj.scales"));
        assert!(keys.contains("proj.biases"));
        assert!(keys.contains("proj.bias"));
    }

    #[test]
    fn missing_and_shape_mismatch_are_rejected() {
        let context = cpu();
        let linear = nn::Linear::unloaded(2, 3, true, Dtype::Float32, context.stream()).unwrap();
        let missing = tempfile::tempdir().unwrap();
        let weight = Array::from_slice(&[0.1f32; 6], &[3, 2]);
        Array::save_safetensors(
            [("proj.weight", &weight)],
            None,
            missing.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(missing.path()).unwrap();
        assert!(matches!(
            build_module_bindings(&linear, "proj", &store),
            Err(ModuleBindingError::MissingParameter { .. })
        ));

        let mismatch = tempfile::tempdir().unwrap();
        let wrong = Array::from_slice(&[0.1f32; 4], &[2, 2]);
        let bias = Array::from_slice(&[0.2f32; 3], &[3]);
        Array::save_safetensors(
            [("proj.weight", &wrong), ("proj.bias", &bias)],
            None,
            mismatch.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(mismatch.path()).unwrap();
        assert!(matches!(
            build_module_bindings(&linear, "proj", &store),
            Err(ModuleBindingError::ShapeMismatch { .. })
        ));
    }
}
