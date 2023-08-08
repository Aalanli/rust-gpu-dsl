use criterion::{black_box, criterion_group, criterion_main, Criterion};

use std::sync::atomic::AtomicU64;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("atomic_u64_seqcst", |b| {
        b.iter(|| {
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
    });

    c.bench_function("atomic_u64_relaxed", |b| {
        b.iter(|| {
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
    });

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
