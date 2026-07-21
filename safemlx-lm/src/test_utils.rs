use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use safemlx::ops::GgufMetadataValue;
use safemlx::{Array, Dtype};
use safemlx_gguf::{GgmlType, TensorInput, Writer};

struct OwnedTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: GgmlType,
    data: Vec<u8>,
}

pub(crate) struct SyntheticGguf {
    _directory: tempfile::TempDir,
    path: PathBuf,
}

impl SyntheticGguf {
    pub(crate) fn dense(
        arrays: &HashMap<String, Array>,
        metadata: &HashMap<String, GgufMetadataValue>,
    ) -> Self {
        Self::with_packed_tensors(arrays, metadata, |_, _| None)
    }

    pub(crate) fn with_packed_tensors<F>(
        arrays: &HashMap<String, Array>,
        metadata: &HashMap<String, GgufMetadataValue>,
        packed_type: F,
    ) -> Self
    where
        F: FnMut(&str, &Array) -> Option<GgmlType>,
    {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("synthetic.gguf");
        let owned = owned_tensors(arrays, packed_type);
        write_tensors(&path, metadata_map(metadata), &owned);
        Self {
            _directory: directory,
            path,
        }
    }

    pub(crate) fn sharded_dense<F>(
        arrays: &HashMap<String, Array>,
        metadata: &HashMap<String, GgufMetadataValue>,
        shard_count: usize,
        mut shard_for: F,
    ) -> Self
    where
        F: FnMut(&str) -> usize,
    {
        assert!(shard_count > 1);
        let directory = tempfile::tempdir().unwrap();
        let mut shards = (0..shard_count).map(|_| Vec::new()).collect::<Vec<_>>();
        let owned = owned_tensors(arrays, |_, _| None);
        let tensor_count = owned.len();
        for tensor in owned {
            let shard = shard_for(&tensor.name);
            assert!(
                shard < shard_count,
                "invalid shard {shard} for {}",
                tensor.name
            );
            shards[shard].push(tensor);
        }

        let first_path = directory
            .path()
            .join(format!("synthetic-{:05}-of-{shard_count:05}.gguf", 1));
        for (shard, tensors) in shards.iter().enumerate() {
            let path = directory.path().join(format!(
                "synthetic-{:05}-of-{shard_count:05}.gguf",
                shard + 1
            ));
            let mut shard_metadata = metadata_map(metadata);
            shard_metadata.extend([
                (
                    "split.no".into(),
                    GgufMetadataValue::Uint64(u64::try_from(shard).unwrap()),
                ),
                (
                    "split.count".into(),
                    GgufMetadataValue::Uint64(u64::try_from(shard_count).unwrap()),
                ),
                (
                    "split.tensors.count".into(),
                    GgufMetadataValue::Uint64(u64::try_from(tensor_count).unwrap()),
                ),
            ]);
            write_tensors(&path, shard_metadata, tensors);
        }
        Self {
            _directory: directory,
            path: first_path,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

fn owned_tensors<F>(arrays: &HashMap<String, Array>, mut packed_type: F) -> Vec<OwnedTensor>
where
    F: FnMut(&str, &Array) -> Option<GgmlType>,
{
    let mut names = arrays.keys().collect::<Vec<_>>();
    names.sort_unstable();
    names
        .into_iter()
        .map(|name| {
            let evaluated = arrays[name].evaluated().unwrap();
            let array = evaluated.as_array();
            let dimensions = array
                .shape()
                .iter()
                .rev()
                .map(|&dimension| u64::try_from(dimension).unwrap())
                .collect::<Vec<_>>();
            let (ggml_type, data) = match packed_type(name, array) {
                Some(ggml_type) => {
                    assert!(array.dtype().is_float());
                    let elements = dimensions.iter().product::<u64>();
                    let (block, bytes) = ggml_type.block_and_bytes().unwrap();
                    assert_eq!(elements % block, 0, "{name} is not block aligned");
                    let byte_count = usize::try_from(elements / block * bytes).unwrap();
                    (ggml_type, vec![0; byte_count])
                }
                None => {
                    assert_eq!(array.dtype(), Dtype::Float32, "unsupported fixture dtype");
                    let data = evaluated
                        .as_slice::<f32>()
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect();
                    (GgmlType::F32, data)
                }
            };
            OwnedTensor {
                name: name.clone(),
                dimensions,
                ggml_type,
                data,
            }
        })
        .collect()
}

fn metadata_map(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> BTreeMap<String, GgufMetadataValue> {
    metadata
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn write_tensors(
    path: &Path,
    metadata: BTreeMap<String, GgufMetadataValue>,
    tensors: &[OwnedTensor],
) {
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: tensor.ggml_type,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}
