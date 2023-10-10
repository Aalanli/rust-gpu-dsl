pub mod ir;
mod transforms;

pub use transforms::print;

pub use language::*;

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

mod language {
    use super::ir::{self, Value, OpEnum, Op, ctx_pop_block, ctx_push, ctx_push_block, Constant};

    pub fn build_fn<const N: usize>(name: &str, args: [ir::ElType; N], scope: impl FnOnce([Value; N])) -> ir::FunctionOp {
        let args = args.map(|x| {
            let ty = ir::Type::scalar(x.clone());
            Value::new(ty)
        });
        ctx_push_block();
        scope(args.clone());
        let body = ctx_pop_block();
        let body = ir::Block::new(args.into_iter().collect(), body);
        ir::FunctionOp { name: name.to_string(), body }
    }

    pub fn build_for(start: &Value, end: &Value, step: &Value, scope: impl FnOnce(&Value)) {
        let loc = std::panic::Location::caller();
        let ind_var = Value::new(ir::Type::scalar(ir::ElType::Val(ir::Dtype::I32)));
        ctx_push_block();
        scope(&ind_var);
        let body = ctx_pop_block();
        let for_op = ir::ForOp { 
            induction_var: ind_var.clone(), start: start.clone(), end: end.clone(), step: step.clone(), body: ir::Block::new(vec![ind_var.clone()], body) };
        let op = Op::new(OpEnum::For(for_op), loc.into());
        ctx_push(&op);
    }

    pub fn program_id() -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::ProgramID(ir::ProgramIDOp::build()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }
    
    pub fn load(ptr: &Value, mask: Option<&Value>, value: Option<&Value>) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Load(ir::LoadOp::build(ptr, mask, value).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }
    
    
    pub fn store(ptr: &Value, value: &Value, mask: Option<&Value>) {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Store(ir::StoreOp::build(ptr, value, mask).unwrap()), loc.into());
        ctx_push(&op);
    }

    pub fn assign(lhs: &Value, rhs: &Value) {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Assign(ir::AssignOp { lhs: lhs.clone(), rhs: rhs.clone() }), loc.into());
        ctx_push(&op);
    }
    
    // pub fn reshape(input: &Value, shape: &[i32]) -> Value {
    //     let loc = std::panic::Location::caller();
    //     let op = Op::new(OpEnum::Reshape(ir::ReshapeOp::build(input, shape).unwrap()), loc.into());
    //     ctx_push(&op);
    //     op.outputs()[0].clone()
    // }
    
    // pub fn permute(input: &Value, axes: &[u32]) -> Value {
    //     let loc = std::panic::Location::caller();
    //     let op = Op::new(OpEnum::Permute(ir::PermuteOp::build(input, axes).unwrap()), loc.into());
    //     ctx_push(&op);
    //     op.outputs()[0].clone()
    // }
    
    pub fn expand_dims(input: &Value, dim: i32) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Expand(ir::ExpandOp::build(input, dim).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }
    
    pub fn reduce(input: &Value, dim: i32, op: ir::ReduceOpOption) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Reduce(ir::ReduceOp::build(input, dim, op).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }

    pub fn sum(input: &Value, dim: i32) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Reduce(ir::ReduceOp::build(input, dim, ir::ReduceOpOption::Sum).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }

    pub fn max(input: &Value, dim: i32) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Reduce(ir::ReduceOp::build(input, dim, ir::ReduceOpOption::Max).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }
    
    pub fn dot(a: &Value, b: &Value) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Dot(ir::DotOp::build(a, b).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }
    
    pub fn elementwise(values: &[&Value], op: ir::IntrinsicElementwise, loc: Option<&std::panic::Location>) -> Value {
        let loc = loc.unwrap_or(std::panic::Location::caller());
        assert!(values.len() > 0, "values must be non-empty");
        let needs_broadcast = values.iter().any(|x| x.type_of().shape() != values[0].type_of().shape());
        let mut val_list = values.iter().map(|x| (*x).clone()).collect::<Vec<_>>();
        if needs_broadcast {
            let mut common_shape = ir::BroadcastOp::broad_cast_shape(values[0].type_of().shape(), values[1].type_of().shape()).unwrap();
            for i in 2..values.len() {
                common_shape = ir::BroadcastOp::broad_cast_shape(common_shape.as_slice(), values[i].type_of().shape()).unwrap();
            }
            for i in 0..values.len() {
                let shape = values[i].type_of().shape();
                if shape != common_shape.as_slice() {
                    let broad_cast = Op::new(OpEnum::Broadcast(ir::BroadcastOp::build(&val_list[i], &common_shape).unwrap()), loc.into());
                    ctx_push(&broad_cast);
                    val_list[i] = broad_cast.outputs()[0].clone();
                }
            }
        }
        let op = Op::new(
            OpEnum::ElementWise(ir::ElementWiseOp::build(ir::ElementwiseFnOption::Intrinsic(op), &val_list).unwrap()), 
            loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }

    macro_rules! elementwise_def {
        ($op:ident, $op_id:ident) => {
            pub fn $op(a: &Value, b: &Value) -> Value {
                elementwise(&[&a, &b], ir::IntrinsicElementwise::$op_id, Some(std::panic::Location::caller()))
            }            
        };
    }

    elementwise_def!(add, Add);
    elementwise_def!(sub, Sub);
    elementwise_def!(mul, Mul);
    elementwise_def!(div, Div);
    elementwise_def!(rem, Rem);
    elementwise_def!(pow, Pow);
    elementwise_def!(eq, Eq);
    elementwise_def!(minimum, Min);
    elementwise_def!(maximum, Max);
    elementwise_def!(lt, Lt);
    elementwise_def!(le, Le);
    elementwise_def!(gt, Gt);
    elementwise_def!(ge, Ge);
    elementwise_def!(and, And);
    elementwise_def!(or, Or);
    elementwise_def!(xor, Xor);
    elementwise_def!(not, Not);
    
    pub fn exp(a: &Value) -> Value {
        elementwise(&[&a], ir::IntrinsicElementwise::Exp, Some(std::panic::Location::caller()))
    }

    pub fn arange(size: i32) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Arange(ir::ArangeOp::build(0, size).unwrap()), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }

    pub fn full(shape: &[usize], value: f32) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Full(ir::FullOp::build(value, shape)), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
    }

    pub fn constant(value: impl Into<Constant>) -> Value {
        let loc = std::panic::Location::caller();
        let op = Op::new(OpEnum::Constant(ir::ConstantOp::build(value)), loc.into());
        ctx_push(&op);
        op.outputs()[0].clone()
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
