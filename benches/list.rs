use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::{self, Rng, distributions::{Uniform, Distribution}};
use rust_gpu_dsl::utils::{self as utils, DoubleList};

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
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
use ListInst::*;

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

fn simulate_inst<T: Default, L: DoubleList<T>>(list: &mut L, instructions: &[ListInst]) {
    let mut key: Option<L::Key> = None;
    for inst in instructions {        
        match inst {
            GoForward => {
                let Some(old_key) = &key else { panic!("invalid execution go-forward"); };
                let Some(next) = list.next(&old_key) else { panic!("invalid execution go-forward-next"); };
                key = Some(next);
            }
            GoBackward => {
                let Some(old_key) = &key else { panic!("invalid execution go-backward"); };
                let Some(next) = list.prev(&old_key) else { panic!("invalid execution go-backward-prev"); };
                key = Some(next);
            }
            InsertAfter => {
                let Some(key) = &key else { panic!("invalid execution insert-after"); };
                list.insert_after(key, T::default());
            }
            InsertBefore => {
                let Some(key) = &key else { panic!("invalid execution insert-before"); };
                list.insert_before(key, T::default());
            }
            RemoveGoF => {
                let Some(old_key) = &key else { panic!("invalid execution remove-go-f"); };
                let new_key = list.next(old_key);
                list.remove(old_key);
                key = new_key;
            }
            RemoveGoB => {
                let Some(old_key) = &key else { panic!("invalid execution remove-go-b"); };
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

fn bench_inst(c: &mut Criterion, header_name: &str, inst: &[ListInst]) {
    let inst = black_box(inst);
    c.bench_function(&format!("{}_map_list_i32", header_name), |b| {
        let mut list = utils::MapDoubleList::<i32>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });

    c.bench_function(&format!("{}_map_list_i128", header_name), |b| {
        let mut list = utils::MapDoubleList::<i128>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });

    c.bench_function(&format!("{}_vec_list_i32", header_name), |b| {
        let mut list = utils::VecDoubleList::<i32>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });

    c.bench_function(&format!("{}_vec_list_i128", header_name), |b| {
        let mut list = utils::VecDoubleList::<i128>::new();
        b.iter(|| {
            simulate_inst(&mut list, &inst);
            list.clear();
        })
    });

    let mut inst_count = [0i32; 10];
    for i in inst {
        inst_count[*i as usize] += 1;
    }

    for i in 0..inst_count.len() {
        println!("{:?}: {}", int_to_inst(i as u8), inst_count[i]);
    }
}

fn make_random_inst(n: usize, remove_prob: f64, pos_shift_prob: f64, insert_prob: f64) -> Vec<ListInst> {
    let mut insts = vec![];
    let mut len = 0;
    let mut cur_pos = 0;
    let mut init_key = false;
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0.0..=1.0);
    while insts.len() < n {
        if len <= 0 {
            insts.push(PushFront);
            len = 1;
            init_key = false;
            continue;
        }
        if !init_key {
            insts.push(GoFront);
            cur_pos = 0;
            init_key = true;
            continue;
        }
        if cur_pos > len || cur_pos < 0 {
            init_key = false;
            continue;
        }

        let cat = die.sample(&mut rng);
        // println!("{}", cat);
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

fn make_move_inst(n: usize, moves: usize) -> Vec<ListInst> {
    let mut insts = vec![];
    insts.push(PushFront);
    insts.push(GoFront);
    for _ in 0..n {
        insts.push(InsertAfter);
    }

    for _ in 0..moves {
        for _ in 0..n {
            insts.push(GoForward);
        }
        for _ in 0..n {
            insts.push(GoBackward);
        }
    }
    insts
}

fn make_insert_inst(n: usize) -> Vec<ListInst> {
    let mut insts = vec![PushFront, GoFront];
    for _ in 0..n {
        insts.push(InsertAfter);
        insts.push(InsertBefore);
    }
    insts
}

fn make_dilated_insert_inst(n: usize, repeat: usize) -> Vec<ListInst> {
    let mut insts = vec![];
    insts.push(PushFront);
    insts.push(GoFront);
    for _ in 0..repeat {
        for _ in 0..n { 
            insts.extend_from_slice(&[
                InsertAfter,
                InsertAfter,
                GoForward,
                RemoveGoF,
            ]);
        }
        for _ in 0..n {
            insts.push(RemoveGoB);
        }
    }

    insts
}

fn criterion_benchmark(c: &mut Criterion) {
    let inst = make_random_inst(1000000, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    bench_inst(c, "random", &inst);

    let inst = make_move_inst(1000, 10);
    bench_inst(c, "move", &inst);

    let inst = make_insert_inst(1000);
    bench_inst(c, "insert", &inst);

    let inst = make_dilated_insert_inst(1000, 10);
    bench_inst(c, "dilated_insert", &inst);
    
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

