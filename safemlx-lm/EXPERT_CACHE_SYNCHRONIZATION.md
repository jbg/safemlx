# Sparse expert cache synchronization

Sparse expert caching has one unavoidable data dependency: exact checkpoint
experts cannot be selected until the router result is available. The cache
keeps this boundary bounded by reducing the route tensor on-device to a demand
histogram with one element per global expert plus an invalid-route scalar. Only
that metadata is read by the host. Original route rows remain on-device and are
rewritten through a global-to-compact lookup table, preserving route order and
weights.

Cold and warm acquisitions are processed as one residency batch. Capacity for
every missing unit is reserved before materialization, requested units are
temporarily protected from eviction, and mmap leases plus source arrays remain
owned by pending materializations. Host copies are evaluated together before
publication. Device copies are published with a shared pending-completion token
and pinned leases; they become ordinary ready residents only after evaluation
of the dependent expert output. A concurrent synchronous acquisition evaluates
and completes an existing pending copy before returning it. Early drop or
unwinding conservatively synchronizes before releasing mmap storage.
Host-to-device copies, compact-bank operations, and grouped expert execution
use the same execution stream, so normal stream ordering carries their
dependencies without an additional whole-stream barrier between each
operation.

The remaining waits are synchronous array evaluations:

- the host must observe the bounded route-demand metadata before it can select
  checkpoint ranges and make eviction decisions;
- pending mmap-backed materializations must complete before their mappings can
  be released safely; and
- the final expert output is evaluated before its expert leases are released.

MLX documents lazy computation and recommends evaluating related outputs
together because every evaluation has fixed overhead:
[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).
The public MLX C API exposes synchronous `mlx_eval`, asynchronous submission via
`mlx_async_eval`, and whole-stream `mlx_synchronize`:
[MLX C transforms](https://ml-explore.github.io/mlx-c/build/html/transforms.html)
and [MLX C streams](https://ml-explore.github.io/mlx-c/build/html/stream.html).
It does not currently expose an event object that can be recorded on one stream,
waited on by another stream, queried by the host, and used to prove when source
storage may be released.

## Event-backed completion design

True transfer/compute overlap requires an upstream or vendored MLX C extension
with these semantics:

- Create and destroy an opaque completion event.
- Record the event after asynchronous evaluation on a producer stream.
- Enqueue a nonblocking wait for that event on a consumer stream.
- Query or wait for completion on the host and report execution errors.
- Define event behavior for both Metal and CUDA streams.

The Rust layer would wrap the object in an owned `CompletionFence`. A pending
expert batch would retain mmap leases, source arrays, host arrays, and its byte
reservations until the fence completes. The device-residency record could then
move from pending to ready without blocking the router thread. The execution
stream would wait on the fence before compact-bank construction, while
independent work such as the DeepSeek shared-expert branch could proceed.

`mlx_async_eval` alone is insufficient because it returns no completion handle
through MLX C. Upstream discussion of nonblocking bindings likewise identifies
the need for a returned synchronizer event, while noting that a host worker
thread only moves the blocking wait and does not provide a device-side
cross-stream dependency:
[MLX nonblocking API discussion](https://github.com/ml-explore/mlx/issues/1251).

Until such an API exists, the cache deliberately prefers bounded batched
evaluation and conservative source lifetime over unsafe overlap. Predictive
prefetch may be added independently, but it can only warm likely experts; exact
routing must still load every selected expert and never substitute another.
