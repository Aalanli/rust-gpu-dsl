// use crate::lang::*;
// use crate::lang::ir;

// // #[test]
// // fn test_softmax() -> Result<()> {
// //     let mut builder = FunctionBuilder::new("softmax_kernel");
// //     let [x_ptr, y_ptr, row_shape] =
// //         builder.arg([Type::f32_ptr(), Type::f32_ptr(), Type::i32_scalar()]);

// //     let tid = builder.program_id()?;
// //     let idx = builder.arange(0, 512)?;
// //     let mask = builder.lt(&idx, &row_shape)?;
// //     let offset = builder.mul(&tid, &row_shape)?;
// //     let idx = builder.add(&idx, &offset)?;

// //     let load_ptr = builder.add(&x_ptr, &idx)?;

// //     let x = builder.load(&load_ptr, Some(&mask), None)?;
// //     let x = builder.exp(&x)?;
// //     let sum = builder.reduce(&x, 0, ReduceOpOption::Sum)?;
// //     let x = builder.div(&x, &sum)?;

// //     let write_ptr = builder.add(&y_ptr, &idx)?;
// //     builder.store(&write_ptr, &x, Some(&mask))?;

// //     let softmax_kernel = builder.build()?;
// //     println!("{:#?}", softmax_kernel);
// //     Ok(())
// // }

// fn softmax(block_size: usize) -> ir::FunctionOp {
//     build_fn("softmax_kernel", 
//         [
//             ir::ElType::Ptr(ir::Dtype::F32), 
//             ir::ElType::Ptr(ir::Dtype::F32),
//             ir::ElType::Val(ir::Dtype::I32)],
//         |[xs, ys, col]| {
//             let pid = program_id();
//             let x_offsets = arange(block_size as i32);
//             let mask = lt(&x_offsets, &col);
//             let x_offsets = add(&x_offsets, &mul(&pid, &col));
//             let x_ptrs = add(&xs, &x_offsets);
//             let x = load(&x_ptrs, Some(&mask), None);
//             let maxes = maximum(&x, &constant(0.0));
//             let x = exp(&sub(&x, &maxes));
//             let s = sum(&x, 0);
//             let x = div(&x, &s);
//             let y_ptrs = add(&ys, &x_offsets);
//             store(&y_ptrs, &x, Some(&mask));
//         })
// }


// fn cdiv(a: &ir::Value, b: &ir::Value) -> ir::Value {
//     div(&sub(&add(a, b), &constant(1)), &b)
// }


// fn matmul(block_m: usize, block_k: usize, block_n: usize) -> ir::FunctionOp {
//     build_fn("matmul_kernel", 
//         [
//             ir::ElType::Ptr(ir::Dtype::F32), 
//             ir::ElType::Ptr(ir::Dtype::F32),
//             ir::ElType::Ptr(ir::Dtype::F32),

//             ir::ElType::Val(ir::Dtype::I32),
//             ir::ElType::Val(ir::Dtype::I32),
//             ir::ElType::Val(ir::Dtype::I32)],
//         |[a, b, c, m, k, n]| {
//             let pid = program_id();
//             let block_m_ = constant(block_m as i32);
//             let block_k_ = constant(block_k as i32);
//             let block_n_ = constant(block_n as i32);

//             let num_n_blocks = cdiv(&n, &block_n_);
//             let pid_m = div(&pid, &num_n_blocks);
//             let pid_n = rem(&pid, &num_n_blocks);
//             let m_offsets = add(&mul(&pid_m, &block_m_), &arange(block_m as i32));
//             let n_offsets = add(&mul(&pid_n, &block_n_), &arange(block_n as i32));
//             let k_offsets = arange(block_k as i32);
//             let a_ptrs = add(&a, &add(
//                 &expand_dims(&mul(&m_offsets, &k), 1), 
//                 &expand_dims(&k_offsets, 0)));
//             let b_ptrs = add(&b, &add(
//                 &expand_dims(&mul(&k_offsets, &n), 1), 
//                 &expand_dims(&n_offsets, 0)));
//             let acc = full(&[block_m, block_n], 0.0);

//             build_for(&constant(0), &cdiv(&k, &block_k_), &constant(1), |i| {
//                 let a_xs = load(&a_ptrs, None, None);
//                 let b_xs = load(&b_ptrs, None, None);
//                 let acc1 = dot(&a_xs, &b_xs);
//                 assign(&acc, &add(&acc, &acc1));

//                 assign(&a_ptrs, &add(&a_ptrs, &block_k_));
//                 assign(&b_ptrs, &add(&b_ptrs, &mul(&block_k_, &n)));
//             });

//             let c_ptrs = add(&c, &add(
//                 &expand_dims(&mul(&m_offsets, &n), 1), 
//                 &expand_dims(&n_offsets, 0)));
//             store(&c_ptrs, &acc, None);
//         })
// }

// #[test]
// fn print_softmax() {
//     let softmax = softmax(512);
//     println!("{:#?}", softmax);
//     println!("{}", print(&ir::Op::new(ir::OpEnum::FunctionOp(softmax), std::panic::Location::caller().into())));
    
// }

// #[test]
// fn print_matmul() {
//     let matmul = matmul(32, 32, 32);
//     // println!("{:#?}", matmul);
//     println!("{}", print(&ir::Op::new(ir::OpEnum::FunctionOp(matmul), std::panic::Location::caller().into())));
    
// }