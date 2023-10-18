use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::{self, Rng, distributions::{Uniform, Distribution}};
use rust_gpu_dsl::utils::{self as utils, DoubleList};

#[derive(PartialEq, Eq, Debug)]
enum ListInst {
    GoForward,
    GoBackward,
    GoFront,
    GoBack,
    InsertAfter,
    InsertBefore,
    PushFront,
    PushBack,
    RemoveGoF,
    RemoveGoB,
}

// move inst: 0..4, insert inst: 4..8, remove inst: 8..10
fn int_to_inst(i: u8) -> ListInst {
    match i {
        0 => GoForward,
        1 => GoBackward,
        2 => GoFront,
        3 => GoBack,
        4 => InsertAfter,
        5 => InsertBefore,
        6 => PushFront,
        7 => PushBack,
        8 => RemoveGoF,
        _ => RemoveGoB,
    }
}

use ListInst::*;

fn make_inst(n: usize, remove_prob: f64, pos_shift_prob: f64, insert_prob: f64) -> Vec<ListInst> {
    let mut insts = vec![];
    let mut len = 0;
    let mut cur_pos = 0;
    let mut init_key = false;
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0.0..=1.0);
    while insts.len() < n {
        if len <= 0 {
            insts.push(PushFront);
            len += 1;
            init_key = false;
            continue;
        }
        if !init_key {
            insts.push(GoFront);
            cur_pos = 0;
            init_key = true;
            continue;
        }
        if cur_pos > len || cur_pos <= 0 {
            init_key = false;
            continue;
        }

        let cat = die.sample(&mut rng);
        let i: u8 = if cat < remove_prob {
            rng.gen_range(8..10)
        } else if cat < remove_prob + pos_shift_prob {
            rng.gen_range(0..4)
        } else {
            rng.gen_range(4..8)
        };
        
        let inst = int_to_inst(i);

        if (inst == GoForward || inst == RemoveGoF) && cur_pos == len - 1 {
            continue;
        }
        if (inst == GoBackward || inst == RemoveGoB) && cur_pos == 0 {
            continue;
        }

        match inst {
            GoForward => { cur_pos += 1; },
            GoBackward => { cur_pos -= 1; }
            GoFront => { cur_pos = 0; }
            GoBack => { cur_pos = len - 1; }
            InsertAfter => { len += 1; }
            InsertBefore => { cur_pos += 1; len += 1; }
            RemoveGoF  => { len -= 1; }
            RemoveGoB => { cur_pos -= 1; len -= 1; }
            PushFront => { cur_pos += 1; len += 1; }
            PushBack => { len += 1; }
        }
        insts.push(inst);

    }

    insts
}


fn simulate_inst<T: Default, L: DoubleList<T>>(list: &mut L, instructions: &[ListInst]) {
    let mut key: Option<L::Key> = None;
    for inst in instructions {
        match inst {
            GoForward => {
                let Some(old_key) = &key else { panic!("invalid execution"); };
                let Some(next) = list.next(&old_key) else { panic!("invalid execution"); };
                key = Some(next);
            }
            GoBackward => {
                let Some(old_key) = &key else { panic!("invalid execution"); };
                let Some(next) = list.prev(&old_key) else { panic!("invalid execution"); };
                key = Some(next);
            }
            InsertAfter => {
                let Some(key) = &key else { panic!("invalid execution"); };
                list.insert_after(key, T::default());
            }
            InsertBefore => {
                let Some(key) = &key else { panic!("invalid execution"); };
                list.insert_before(key, T::default());
            }
            RemoveGoF => {
                let Some(old_key) = &key else { panic!("invalid execution"); };
                let new_key = list.next(old_key);
                list.remove(old_key);
                key = new_key;
            }
            RemoveGoB => {
                let Some(old_key) = &key else { panic!("invalid execution"); };
                let new_key = list.prev(old_key);
                list.remove(old_key);
                key = new_key;
            }
            PushFront => {
                list.push_front(T::default());
            }
            PushBack => {
                list.push_back(T::default());
            }
            GoFront => {
                key = list.front();
            }
            GoBack => {
                key = list.back();
            }
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let inst = make_inst(100000, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    let inst = black_box(inst);
    c.bench_function("map_list_i32", |b| {
        let mut list = utils::MapDoubleList::<i32>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });

    c.bench_function("map_list_i128", |b| {
        let mut list = utils::MapDoubleList::<i128>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
