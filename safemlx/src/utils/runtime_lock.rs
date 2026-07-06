use parking_lot::{ReentrantMutex, ReentrantMutexGuard};

static RUNTIME_LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

pub(crate) struct RuntimeLockGuard {
    _guard: ReentrantMutexGuard<'static, ()>,
}

pub(crate) fn enter() -> RuntimeLockGuard {
    RuntimeLockGuard {
        _guard: RUNTIME_LOCK.lock(),
    }
}
