//! Mixture-of-experts routing helpers.

use crate::{
    error::Result,
    ops::{arange, argsort, gather_mm, indexing::take_axis, ones_dtype, segment_sum},
    Array, Dtype, Stream,
};

/// Device-side routing plan for grouped expert execution.
///
/// The plan is produced by sorting flattened group/expert ids. `route_indices` maps every sorted
/// route back to its original flattened route position. For top-k routing, `token_indices` and
/// `slot_indices` split that flattened position back into `(token, slot)`.
#[derive(Debug)]
pub struct GroupedRoutePlan {
    /// Group or expert id for each sorted route.
    pub sorted_group_ids: Array,
    /// Original flattened route index for each sorted route.
    pub route_indices: Array,
    /// Source row for each sorted route.
    pub token_indices: Array,
    /// Top-k slot for each sorted route, or zeros for a 1-D grouping input.
    pub slot_indices: Array,
    /// Number of routes assigned to each group.
    pub group_counts: Array,
    /// Exclusive prefix sum of `group_counts`.
    pub group_offsets: Array,
}

/// Sort flattened group ids on-device and return indices useful for grouped kernels.
///
/// `group_ids` can be 1-D (`[routes]`) or 2-D (`[tokens, slots]`). The returned
/// `sorted_group_ids` are suitable for `grouped_matmul(..., sorted_indices = true)`, while
/// `token_indices` can be used to gather source rows and later reduce routed outputs back to
/// tokens with [`segment_sum_by_index`].
pub fn group_by_id(
    group_ids: impl AsRef<Array>,
    num_groups: i32,
    stream: impl AsRef<Stream>,
) -> Result<GroupedRoutePlan> {
    let stream = stream.as_ref();
    let group_ids = group_ids.as_ref();
    let top_k = if group_ids.ndim() >= 2 {
        group_ids.dim(-1)
    } else {
        1
    };

    let flat_group_ids = group_ids
        .reshape(&[-1], stream)?
        .as_dtype(Dtype::Int32, stream)?;
    let order = argsort(&flat_group_ids, stream)?;
    let sorted_group_ids = flat_group_ids.take(&order, stream)?;
    let route_indices = order.as_dtype(Dtype::Int32, stream)?;
    let divisor = Array::from_int(top_k);
    let token_indices = route_indices.floor_divide(&divisor, stream)?;
    let slot_indices = route_indices.remainder(&divisor, stream)?;

    let counts = ones_dtype(&[flat_group_ids.size() as i32], Dtype::Int32, stream)?.segment_sum(
        &flat_group_ids,
        num_groups,
        0,
        stream,
    )?;
    let offsets = counts.cumsum(0, None, false, stream)?;

    Ok(GroupedRoutePlan {
        sorted_group_ids,
        route_indices,
        token_indices,
        slot_indices,
        group_counts: counts,
        group_offsets: offsets,
    })
}

/// Matrix multiplication for rows assigned to variable-sized groups.
///
/// `inputs` has shape `[routes, in_dim]`, `weights` has shape
/// `[num_groups, in_dim, out_dim]`, and `group_ids` has shape `[routes]`. When `group_ids` are
/// already sorted, pass `sorted_indices = true` so MLX can use its sorted gather-matmul path.
pub fn grouped_matmul(
    inputs: impl AsRef<Array>,
    weights: impl AsRef<Array>,
    group_ids: impl AsRef<Array>,
    sorted_indices: bool,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let inputs = inputs.as_ref();
    let weights = weights.as_ref();
    let routes = inputs.dim(0);
    let in_dim = inputs.dim(-1);
    let out_dim = weights.dim(-1);
    let inputs = inputs.reshape(&[routes, 1, in_dim], stream)?;
    gather_mm(
        &inputs,
        weights,
        None::<&Array>,
        group_ids.as_ref(),
        sorted_indices,
        stream,
    )?
    .reshape(&[routes, out_dim], stream)
}

/// Gather source rows according to a routing plan.
pub fn gather_grouped_rows(
    rows: impl AsRef<Array>,
    plan: &GroupedRoutePlan,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    take_axis(rows, &plan.token_indices, 0, stream)
}

/// Gather flattened per-route values according to a routing plan.
///
/// This is useful for top-k routing weights with shape `[tokens, top_k]`.
pub fn gather_route_values(
    values: impl AsRef<Array>,
    plan: &GroupedRoutePlan,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    values
        .as_ref()
        .reshape(&[-1], stream.as_ref())?
        .take(&plan.route_indices, stream)
}

/// Reduce routed values back to source rows using summation.
///
/// `values` should have shape `[routes, ...]`, and `indices` should have shape `[routes]`.
pub fn segment_sum_by_index(
    values: impl AsRef<Array>,
    indices: impl AsRef<Array>,
    num_segments: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    segment_sum(values, indices, num_segments, 0, stream)
}

/// Build a sorted top-k route plan from `[tokens, top_k]` expert ids.
pub fn topk_route_plan(
    expert_ids: impl AsRef<Array>,
    num_experts: i32,
    stream: impl AsRef<Stream>,
) -> Result<GroupedRoutePlan> {
    group_by_id(expert_ids, num_experts, stream)
}

/// Convenience helper for a single expert-major projection followed by reduce-back.
///
/// This gathers `hidden_states` by the route plan, runs grouped matmul with expert weights, applies
/// route weights, and sums duplicate token routes back into `[tokens, out_dim]`.
pub fn routed_grouped_matmul(
    hidden_states: impl AsRef<Array>,
    expert_weights: impl AsRef<Array>,
    expert_ids: impl AsRef<Array>,
    route_weights: impl AsRef<Array>,
    num_experts: i32,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let hidden_states = hidden_states.as_ref();
    let plan = topk_route_plan(expert_ids, num_experts, stream)?;
    let routed = gather_grouped_rows(hidden_states, &plan, stream)?;
    let projected = grouped_matmul(
        &routed,
        expert_weights,
        &plan.sorted_group_ids,
        true,
        stream,
    )?;
    let weights = gather_route_values(route_weights, &plan, stream)?
        .reshape(&[projected.dim(0), 1], stream)?;
    let weighted = projected.multiply(&weights, stream)?;
    segment_sum_by_index(weighted, &plan.token_indices, hidden_states.dim(0), stream)
}

/// Build `[0, 1, ..., routes - 1]` on the same stream.
pub fn route_arange(routes: i32, stream: impl AsRef<Stream>) -> Result<Array> {
    arange::<_, i32>(0, routes, None, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::{
            all_close,
            indexing::{take_axis, IndexOp},
            matmul, reshape, sum_axis,
        },
        Stream,
    };

    #[test]
    fn test_group_by_id_topk_plan() {
        let stream = crate::test_stream();
        let experts = Array::from_slice(&[2i32, 0, 1, 2, 0, 1], &[3, 2]);
        let plan = topk_route_plan(&experts, 3, stream).unwrap();

        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.sorted_group_ids),
            &[0, 0, 1, 1, 2, 2]
        );
        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.route_indices),
            &[1, 4, 2, 5, 0, 3]
        );
        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.token_indices),
            &[0, 2, 1, 2, 0, 1]
        );
        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.slot_indices),
            &[1, 0, 0, 1, 0, 1]
        );
        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.group_counts),
            &[2, 2, 2]
        );
        assert_eq!(
            crate::array::eval_vec::<i32>(&plan.group_offsets),
            &[0, 2, 4]
        );
    }

    #[test]
    fn test_grouped_matmul_matches_gathered_reference() {
        let stream = crate::test_stream();
        let inputs = reshape(
            Array::arange::<_, f32>(0.0, 12.0, None, stream).unwrap(),
            &[4, 3],
            stream,
        )
        .unwrap();
        let weights = reshape(
            Array::arange::<_, f32>(0.0, 18.0, None, stream).unwrap(),
            &[3, 3, 2],
            stream,
        )
        .unwrap();
        let group_ids = Array::from_slice(&[2i32, 0, 1, 2], &[4]);
        let plan = group_by_id(&group_ids, 3, stream).unwrap();
        let sorted_inputs = take_axis(&inputs, &plan.token_indices, 0, stream).unwrap();
        let grouped = grouped_matmul(
            &sorted_inputs,
            &weights,
            &plan.sorted_group_ids,
            true,
            stream,
        )
        .unwrap();
        let selected_weights = take_axis(&weights, &plan.sorted_group_ids, 0, stream).unwrap();
        let expected = matmul(
            sorted_inputs.index_device((.., crate::ops::indexing::NewAxis, ..), stream),
            selected_weights,
            stream,
        )
        .unwrap()
        .reshape(&[4, 2], stream)
        .unwrap();

        assert!(all_close(&grouped, &expected, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(&stream));
    }

    #[test]
    fn test_routed_grouped_matmul_matches_reference() {
        let stream = crate::test_stream();
        let hidden = reshape(
            Array::arange::<_, f32>(0.0, 12.0, None, stream).unwrap(),
            &[3, 4],
            stream,
        )
        .unwrap();
        let weights = reshape(
            Array::arange::<_, f32>(0.0, 40.0, None, stream).unwrap(),
            &[5, 4, 2],
            stream,
        )
        .unwrap();
        let expert_ids = Array::from_slice(&[2i32, 0, 1, 2, 0, 1], &[3, 2]);
        let route_weights = Array::from_slice(&[0.25f32, 0.75, 1.0, 0.5, 0.2, 0.8], &[3, 2]);

        let actual =
            routed_grouped_matmul(&hidden, &weights, &expert_ids, &route_weights, 5, stream)
                .unwrap();

        let selected_weights = take_axis(&weights, &expert_ids, 0, stream).unwrap();
        let current = matmul(
            hidden.index_device(
                (
                    ..,
                    crate::ops::indexing::NewAxis,
                    crate::ops::indexing::NewAxis,
                    ..,
                ),
                stream,
            ),
            selected_weights,
            stream,
        )
        .unwrap()
        .reshape(&[3, 2, 2], stream)
        .unwrap();
        let expected = sum_axis(
            &current
                .multiply(
                    route_weights.index_device((.., .., crate::ops::indexing::NewAxis), stream),
                    stream,
                )
                .unwrap(),
            -2,
            false,
            stream,
        )
        .unwrap();

        assert!(all_close(&actual, &expected, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(&stream));

        let _ = route_arange(
            4,
            Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0)),
        )
        .unwrap();
    }
}
