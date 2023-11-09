pub mod ir;
mod transforms;


// mutable value semantics, lol
// we adopt mutable value semantics in the tensor program for the sake of simplicity, so no aliasing, only mutation via
// access of the identifier of a value.
// only before the load boundary and store boundary are pointers allowed, as in triton

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Location {
    row: u32,
    col: u32,
    file: String,
}

impl<'a> From<std::panic::Location<'a>> for Location {
    fn from(value: std::panic::Location<'a>) -> Self {
        Location {
            row: value.line(),
            col: value.column(),
            file: value.file().to_string(),
        }
    }
}

impl<'a, 'b> From<&'b std::panic::Location<'a>> for Location {
    fn from(value: &'b std::panic::Location<'a>) -> Self {
        Location {
            row: value.line(),
            col: value.column(),
            file: value.file().to_string(),
        }
    }
}


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
