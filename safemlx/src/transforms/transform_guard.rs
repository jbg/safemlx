use crate::utils::runtime_lock;

pub(crate) struct TransformGuard {
    _guard: runtime_lock::RuntimeLockGuard,
}

pub(crate) fn enter() -> TransformGuard {
    TransformGuard {
        _guard: runtime_lock::enter(),
    }
}
