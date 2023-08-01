use criterion::{black_box, criterion_group, criterion_main, Criterion};


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("boxed_iter", |b| b.iter(|| {
        let it: Box<dyn Iterator<Item = i32>> = if black_box(true) {
            Box::new(
                std::iter::empty()
                    .chain(std::iter::once(black_box(1)))
                    .chain(std::iter::once(black_box(2)))
                    .chain(std::iter::once(black_box(3)))
                    .chain(std::iter::once(black_box(3)))
                    .chain(std::iter::once(black_box(3)))
                    .chain(std::iter::once(black_box(3)))
                    .chain(std::iter::once(black_box(3)))
                    .chain(std::iter::once(black_box(3)))
            )
        } else {
            Box::new(
                std::iter::empty().chain(std::iter::once(1))
            )
        };
        let mut s = 0;
        for i in it {
            s += i;
        }
        s
    }));

    c.bench_function("vector_boxed_iter", |b| b.iter(|| {
        let it: Box<dyn Iterator<Item = i32>> = if black_box(true) {
            Box::new(
                // std::iter::empty()
                //     .chain(std::iter::once(black_box(1)))
                //     .chain(std::iter::once(black_box(2)))
                //     .chain(std::iter::once(black_box(3)))
                vec![black_box(1), black_box(2), black_box(3), black_box(3), black_box(3), black_box(3), black_box(3), black_box(3)].into_iter()
            )
        } else {
            Box::new(
                std::iter::empty().chain(std::iter::once(1))
            )
        };
        let mut s = 0;
        for i in it {
            s += i;
        }
        s
    }));

    c.bench_function("vector_iter", |b| b.iter(|| {
        let it: Vec<i32> = if black_box(true) {
            vec![black_box(1), black_box(2), black_box(3), black_box(3), black_box(3), black_box(3), black_box(3), black_box(3)]
        } else {
            vec![black_box(1)]
        };
        let mut s = 0;
        for i in it {
            s += i;
        }
        s
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);