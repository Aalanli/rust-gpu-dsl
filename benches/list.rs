use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand;
use rust_gpu_dsl::utils::DoubleList;
enum ListInst {
    GoForward,
    GoBackward,
    InsertAfter,
    InsertBefore,
    PushStack,
    PopStack,
    Remove,
    GetFront,
    GetBack,
}

use ListInst::*;

fn make_inst() -> Vec<ListInst> {
    let mut inst = vec![];
    
    inst
}


fn simulate_inst<T: Default, L: DoubleList<T>>(list: &mut L, instructions: &[ListInst]) {
    let mut key_stack: Vec<L::Key> = vec![];
    for inst in instructions {
        match inst {
            GoForward => {
                let key = key_stack.pop().unwrap();
                let next = list.next(&key);
                key_stack.push(next.unwrap());
            }
            GoBackward => {
                let key = key_stack.pop().unwrap();
                let next = list.prev(&key);
                key_stack.push(next.unwrap());
            }
            InsertAfter => {
                list.insert_after(key_stack.last().unwrap(), T::default());
            }
            InsertBefore => {
                list.insert_before(key_stack.last().unwrap(), T::default());
            }
            PushStack => {
                key_stack.push(key_stack.last().unwrap().clone());   
            }
            PopStack => {
                key_stack.pop();
            }
            Remove => {
                list.remove(&key_stack.pop().unwrap());
            }
            GetFront => {
                key_stack.push(list.front().unwrap());
            }
            GetBack => {
                key_stack.push(list.back().unwrap());
            }
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("boxed_iter", |b| {
        b.iter(|| {
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
                        .chain(std::iter::once(black_box(3))),
                )
            } else {
                Box::new(std::iter::empty().chain(std::iter::once(1)))
            };
            let mut s = 0;
            for i in it {
                s += i;
            }
            s
        })
    });

    c.bench_function("vector_boxed_iter", |b| {
        b.iter(|| {
            let it: Box<dyn Iterator<Item = i32>> = if black_box(true) {
                Box::new(
                    // std::iter::empty()
                    //     .chain(std::iter::once(black_box(1)))
                    //     .chain(std::iter::once(black_box(2)))
                    //     .chain(std::iter::once(black_box(3)))
                    vec![
                        black_box(1),
                        black_box(2),
                        black_box(3),
                        black_box(3),
                        black_box(3),
                        black_box(3),
                        black_box(3),
                        black_box(3),
                    ]
                    .into_iter(),
                )
            } else {
                Box::new(std::iter::empty().chain(std::iter::once(1)))
            };
            let mut s = 0;
            for i in it {
                s += i;
            }
            s
        })
    });

    c.bench_function("vector_iter", |b| {
        b.iter(|| {
            let it: Vec<i32> = if black_box(true) {
                vec![
                    black_box(1),
                    black_box(2),
                    black_box(3),
                    black_box(3),
                    black_box(3),
                    black_box(3),
                    black_box(3),
                    black_box(3),
                ]
            } else {
                vec![black_box(1)]
            };
            let mut s = 0;
            for i in it {
                s += i;
            }
            s
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
