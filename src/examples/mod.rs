use crate::lang::*;
use crate::lang::ir;

// #[test]
// fn test_softmax() -> Result<()> {
//     let mut builder = FunctionBuilder::new("softmax_kernel");
//     let [x_ptr, y_ptr, row_shape] =
//         builder.arg([Type::f32_ptr(), Type::f32_ptr(), Type::i32_scalar()]);

//     let tid = builder.program_id()?;
//     let idx = builder.arange(0, 512)?;
//     let mask = builder.lt(&idx, &row_shape)?;
//     let offset = builder.mul(&tid, &row_shape)?;
//     let idx = builder.add(&idx, &offset)?;

//     let load_ptr = builder.add(&x_ptr, &idx)?;

//     let x = builder.load(&load_ptr, Some(&mask), None)?;
//     let x = builder.exp(&x)?;
//     let sum = builder.reduce(&x, 0, ReduceOpOption::Sum)?;
//     let x = builder.div(&x, &sum)?;

//     let write_ptr = builder.add(&y_ptr, &idx)?;
//     builder.store(&write_ptr, &x, Some(&mask))?;

//     let softmax_kernel = builder.build()?;
//     println!("{:#?}", softmax_kernel);
//     Ok(())
// }

fn softmax(block_size: usize) -> ir::FunctionOp {
    build_fn("softmax_kernel", 
        [
            ir::ElType::Ptr(ir::Dtype::F32), 
            ir::ElType::Ptr(ir::Dtype::F32),
            ir::ElType::Val(ir::Dtype::I32)],
        |[xs, ys, col]| {
            let pid = program_id();
            let x_offsets = arange(block_size as i32);
        })
}

#[test]
fn print_softmax() {
    let softmax = softmax(512);
    println!("{:#?}", softmax);
    println!("{}", print(&ir::Op::new(ir::OpEnum::FunctionOp(softmax), std::panic::Location::caller().into())));
    
}