use anyhow::{Error, Result};
use std::ops::Range;
use crate::lang::{Block, Value, Type, ElType, Dtype, Constant};


#[derive(Debug, Clone)]
pub enum OpEnum { // could be a trait, once we have a better idea of the functions
    ProgramID(ProgramIDOp),

    Load(LoadOp),
    Store(StoreOp),

    Reshape(ReshapeOp),
    Permute(PermuteOp),
    Slice(SliceOp),
    Expand(ExpandOp),
    Broadcast(BroadcastOp),

    Reduce(ReduceOp),
    ElementWise(ElementWiseOp),
    Dot(DotOp),

    Full(FullOp),
    Arange(ArangeOp),

    For(ForOp),
    If(IfOp),
    FunctionOp(FunctionOp),
    Assign(AssignOp),
}


impl OpEnum {
    pub fn inputs(&self) -> Vec<&Value> {
        match &self {
            Self::ProgramID(op) => {vec![&op.output]},
            Self::Load(op) => {
                let mut inputs = vec![&op.ptr];
                if let Some(mask) = &op.mask {
                    inputs.push(mask);
                }
                if let Some(value) = &op.value {
                    inputs.push(value);
                }
                inputs
            }
            Self::Store(op) => {
                let mut inputs = vec![&op.ptr, &op.value];
                if let Some(mask) = &op.mask {
                    inputs.push(mask);
                }
                inputs
            }
            Self::Reshape(op) => vec![&op.input],
            Self::Permute(op) => vec![&op.input],
            Self::Slice(op) => vec![&op.input],
            Self::Expand(op) => vec![&op.input],
            Self::Broadcast(op) => vec![&op.input, &op.other],
            Self::Reduce(op) => vec![&op.input],
            Self::ElementWise(op) => op.args.iter().collect(),
            Self::Dot(op) => vec![&op.a, &op.b],
            Self::Full(_op) => vec![],
            Self::Arange(_op) => vec![],
            Self::For(op) => vec![&op.start, &op.end, &op.step],
            Self::If(op) => vec![&op.cond],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(op) => vec![&op.lhs, &op.rhs],
        }
    }

    pub fn inputs_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Self::ProgramID(op) => {vec![&mut op.output]},
            Self::Load(op) => {
                let mut inputs = vec![&mut op.ptr];
                if let Some(mask) = &mut op.mask {
                    inputs.push(mask);
                }
                if let Some(value) = &mut op.value {
                    inputs.push(value);
                }
                inputs
            }
            Self::Store(op) => {
                let mut inputs = vec![&mut op.ptr, &mut op.value];
                if let Some(mask) = &mut op.mask {
                    inputs.push(mask);
                }
                inputs
            }
            Self::Reshape(op) => vec![&mut op.input],
            Self::Permute(op) => vec![&mut op.input],
            Self::Slice(op) => vec![&mut op.input],
            Self::Expand(op) => vec![&mut op.input],
            Self::Broadcast(op) => vec![&mut op.input, &mut op.other],
            Self::Reduce(op) => vec![&mut op.input],
            Self::ElementWise(op) => op.args.iter_mut().collect(),
            Self::Dot(op) => vec![&mut op.a, &mut op.b],
            Self::Full(_op) => vec![],
            Self::Arange(_op) => vec![],
            Self::For(op) => vec![&mut op.start, &mut op.end, &mut op.step],
            Self::If(op) => vec![&mut op.cond],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(op) => vec![&mut op.lhs, &mut op.rhs],
        }
    }

    pub fn outputs(&self) -> Vec<&Value> {
        match &self {
            Self::ProgramID(op) => {vec![&op.output]},
            Self::Load(op) => vec![&op.output],
            Self::Store(_op) => vec![],
            Self::Reshape(op) => vec![&op.output],
            Self::Permute(op) => vec![&op.output],
            Self::Slice(op) => vec![&op.output],
            Self::Expand(op) => vec![&op.output],
            Self::Broadcast(op) => vec![&op.output],
            Self::Reduce(op) => vec![&op.output],
            Self::ElementWise(op) => vec![&op.output],
            Self::Dot(op) => vec![&op.output],
            Self::Full(op) => vec![&op.output],
            Self::Arange(op) => vec![&op.output],
            Self::For(_op) => vec![],
            Self::If(_op) => vec![],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(_op) => vec![],
        }
    }

    pub fn outputs_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Self::ProgramID(op) => {vec![&mut op.output]},
            Self::Load(op) => vec![&mut op.output],
            Self::Store(_op) => vec![],
            Self::Reshape(op) => vec![&mut op.output],
            Self::Permute(op) => vec![&mut op.output],
            Self::Slice(op) => vec![&mut op.output],
            Self::Expand(op) => vec![&mut op.output],
            Self::Broadcast(op) => vec![&mut op.output],
            Self::Reduce(op) => vec![&mut op.output],
            Self::ElementWise(op) => vec![&mut op.output],
            Self::Dot(op) => vec![&mut op.output],
            Self::Full(op) => vec![&mut op.output],
            Self::Arange(op) => vec![&mut op.output],
            Self::For(_op) => vec![],
            Self::If(_op) => vec![],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(_op) => vec![],
        }
    }

    pub fn blocks(&self) -> Vec<&Block> {
        match &self {
            Self::If(op) => {vec![&op.then, &op.else_]},
            Self::For(op) => {vec![&op.body]},
            Self::FunctionOp(op) => {vec![&op.body]},
            _ => vec![],
        }
    }

    pub fn blocks_mut(&mut self) -> Vec<&mut Block> {
        match self {
            Self::If(op) => {vec![&mut op.then, &mut op.else_]},
            Self::For(op) => {vec![&mut op.body]},
            Self::FunctionOp(op) => {vec![&mut op.body]},
            _ => vec![],
        }
    }

    pub fn internal_as_any(&self) -> &dyn std::any::Any {
        match self {
            Self::ProgramID(x) => x,
            Self::Load(x) => x,
            Self::Store(x) => x,
            Self::Reshape(x) => x,
            Self::Permute(x) => x,
            Self::Slice(x) => x,
            Self::Expand(x) => x,
            Self::Broadcast(x) => x,
            Self::Reduce(x) => x,
            Self::ElementWise(x) => x,
            Self::Dot(x) => x,
            Self::Full(x) => x,
            Self::Arange(x) => x,
            Self::For(x) => x,
            Self::If(x) => x,
            Self::FunctionOp(x) => x,
            Self::Assign(x) => x,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssignOp {
    pub lhs: Value,
    pub rhs: Value,
}


#[derive(Debug, Clone)]
pub struct ForOp {
    pub induction_var: Value,
    pub start: Value,
    pub end: Value,
    pub step: Value,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct IfOp {
    pub cond: Value,
    pub then: Block,
    pub else_: Block,
}

#[derive(Debug, Clone)]
pub struct FunctionOp {
    pub name: String,
    pub body: Block
}

#[derive(Debug, Clone)]
pub struct ProgramIDOp {
    pub output: Value,
}

impl ProgramIDOp {
    pub fn build() -> Self {
        let val = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        ProgramIDOp { output: val.clone() }
    }

    
}

#[derive(Debug, Clone)]
pub struct LoadOp {
    pub ptr: Value,
    pub mask: Option<Value>,
    pub value: Option<Value>,
    pub output: Value,
}

impl LoadOp {
    pub fn build(ptr: &Value, mask: Option<&Value>, value: Option<&Value>) -> Result<Self> {
        let val = Value::new(ptr.type_of().to_value());
        Ok(LoadOp {
            ptr: ptr.clone(),
            mask: mask.cloned(),
            value: value.cloned(),
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct StoreOp {
    pub ptr: Value,
    pub value: Value,
    pub mask: Option<Value>,
}

impl StoreOp {
    pub fn build(ptr: &Value, value: &Value, mask: Option<&Value>) -> Result<Self> {
        Ok(StoreOp {
            ptr: ptr.clone(),
            value: value.clone(),
            mask: mask.cloned(),
        })
    }

}

#[derive(Debug, Clone)]
pub struct ReshapeOp {
    pub input: Value,
    pub shape: Vec<usize>,
    pub output: Value,
}

impl ReshapeOp {
    fn get_reshape_output_shape(ishape: &[usize], shape: &[i32]) -> Result<Vec<usize>> {
        if shape.iter().filter(|x| **x < -1 || **x == 0).count() > 0 {
            return Err(Error::msg(
                "shape cannot contain any negative values (other than -1) or zeros",
            ));
        }
        let neg_count = shape.iter().filter(|x| **x == -1).count();
        if neg_count > 1 {
            return Err(Error::msg("shape cannot contain more than one -1"));
        }
        let prod_wo_neg = shape
            .iter()
            .filter(|x| **x > 0)
            .fold(1, |x, y| x * (*y) as usize);
        let prod_in = ishape.iter().fold(1, |x, y| x * *y);
        if (neg_count == 0 && prod_in != prod_wo_neg) || prod_in % prod_wo_neg != 0 {
            return Err(Error::msg(format!(
                "cannot reshape tensor of size {:?} into shape {:?}",
                ishape, shape
            )));
        }
        let oshape = shape
            .iter()
            .map(|x| {
                if *x == -1 {
                    prod_in / prod_wo_neg
                } else {
                    *x as usize
                }
            })
            .collect::<Vec<_>>();
        Ok(oshape)
    }

    pub fn build(input: &Value, shape: &[i32]) -> Result<Self> {
        let oshape = ReshapeOp::get_reshape_output_shape(input.type_of().shape(), shape)?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        Ok(ReshapeOp {
            input: input.clone(),
            shape: oshape,
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PermuteOp {
    pub input: Value,
    pub permutation: Vec<usize>,
    pub output: Value,
}

impl PermuteOp {
    fn get_permute_output_shape(ishape: &[usize], permute: &[u32]) -> Result<Vec<usize>> {
        let mut perm: Vec<_> = permute.into();
        perm.sort();
        for (i, p) in (0..ishape.len()).zip(perm) {
            if i != p as usize {
                return Err(Error::msg(format!(
                    "Invalid permutation indicies, got {:?}",
                    permute
                )));
            }
        }
        let out = permute.iter().map(|x| ishape[*x as usize]).collect();
        Ok(out)
    }

    pub fn build(input: &Value, permutation: &[u32]) -> Result<Self> {
        let oshape =
            PermuteOp::get_permute_output_shape(input.type_of().shape(), permutation)?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        Ok(PermuteOp {
            input: input.clone(),
            permutation: permutation.iter().map(|x| *x as usize).collect(),
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SliceOp {
    pub input: Value,
    pub slices: Vec<Range<usize>>,
    pub output: Value,
}

impl SliceOp {
    fn get_slice_output_shape(
        ishape: &[usize],
        slices: &[Range<i32>],
    ) -> Result<(Vec<usize>, Vec<Range<usize>>)> {
        if slices.len() > ishape.len() {
            return Err(Error::msg(
                "Number of slice dimensions must be equal or smaller 
                than the number of actual dimensions"));
        }
        let mut oshape = vec![];
        let mut oslice = vec![];
        for (os, slice) in ishape.iter().zip(slices) {
            if slice.end + (*os as i32) <= 0 || slice.start < 0 {
                return Err(Error::msg("slice error"));
            }
            let upper = if slice.end < 0 {
                ((*os) as i32 + slice.end) as usize
            } else {
                slice.end as usize
            };
            let lower = slice.start as usize;
            if upper < lower {
                return Err(Error::msg("slice error"));
            }
            let new_shape = upper - lower;
            if new_shape > 0 {
                oshape.push(new_shape)
            }
            oslice.push(lower..upper);
        }
        Ok((oshape, oslice))
    }

    /// Range: a..b
    /// RangeTo: 0..b
    /// RangeFrom: a..-1
    /// RangeFull: 0..-1
    /// Index and remove dim: a..a
    pub fn build(input: &Value, slices: &[Range<i32>]) -> Result<Self> {
        let (oshape, oslice) =
            SliceOp::get_slice_output_shape(input.type_of().shape(), slices)?;

        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        Ok(SliceOp {
            input: input.clone(),
            slices: oslice,
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExpandOp {
    pub input: Value,
    pub dim: usize,
    pub output: Value,
}

impl ExpandOp {
    pub fn build(input: &Value, dim: i32) -> Result<Self> {
        let res_dim = if dim < 0 {
            input.type_of().rank() as i32 + dim + 1
        } else {
            dim
        };
        if res_dim < 0 || res_dim > input.type_of().rank() as i32 {
            return Err(Error::msg(format!(
                "Invalid expand dimension, got {}",
                dim
            )));
        }
        let mut oshape = input.type_of().shape().to_vec();
        oshape.insert(res_dim as usize, 1);
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        Ok(ExpandOp {
            input: input.clone(),
            dim: res_dim as usize,
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastOp {
    pub input: Value,
    pub other: Value,
    pub output: Value,
}

impl BroadcastOp {
    fn broad_cast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
        if b.len() > a.len() {
            return Self::broad_cast_shape(b, a);
        }
        let mut oshape = vec![];
        for (x, y) in a.iter().zip(b.iter()) {
            if *x == *y {
                oshape.push(*x);
            } else if *x == 1 {
                oshape.push(*y);
            } else if *y == 1 {
                oshape.push(*x);
            } else {
                return Err(Error::msg(format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    a, b
                )));
            }
        }
        for x in a.iter().skip(b.len()) {
            oshape.push(*x);
        }
        Ok(oshape)
    }

    pub fn build(input: &Value, other: &Value) -> Result<Self> {
        let shape = BroadcastOp::broad_cast_shape(
            input.type_of().shape(),
            other.type_of().shape(),
        )?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &shape));
        Ok(BroadcastOp {
            input: input.clone(),
            other: other.clone(),
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ReduceOp {
    pub input: Value,
    pub dim: usize,
    pub op: ReduceOpOption,
    pub output: Value,
}

impl ReduceOp {
    pub fn build(input: &Value, dim: i32, op: ReduceOpOption) -> Result<Self> {
        let reduce_dim = if dim < 0 {
            input.type_of().rank() as i32 + dim
        } else {
            dim
        };
        if reduce_dim < 0 || reduce_dim >= input.type_of().rank() as i32 {
            return Err(Error::msg(format!(
                "Invalid reduce dimension, got {} with input shape {:?}",
                dim, input.type_of().shape()
            )));
        }
        let val = Value::new(Type::tensor(
            input.type_of().eltype.clone(),
            &input
                .type_of()
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != dim as usize)
                .map(|(_, x)| *x)
                .collect::<Vec<_>>(),
        ));
        Ok(ReduceOp {
            input: input.clone(),
            dim: dim as usize,
            op,
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct DotOp {
    pub a: Value,
    pub b: Value,
    pub output: Value,
}

impl DotOp {
    pub fn build(a: &Value, b: &Value) -> Result<Self> {
        if a.type_of().rank() != 2 || b.type_of().rank() != 2 {
            return Err(Error::msg(format!(
                "Dot product requires both inputs to be matrices, got {:?} and {:?}",
                a.type_of(),
                b.type_of()
            )));
        }
        if a.type_of().eltype != b.type_of().eltype {
            return Err(Error::msg(format!(
                "Dot product requires both inputs to have the same element type, got {:?} and {:?}",
                a.type_of(),
                b.type_of()
            )));
        }
        let val = Value::new(Type::tensor(
            a.type_of().eltype.clone(),
            &[a.type_of().shape()[0], b.type_of().shape()[1]],
        ));
        Ok(DotOp {
            a: a.clone(),
            b: b.clone(),
            output: val.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ElementWiseOp {
    pub name: ElementwiseFnOption,
    pub args: Vec<Value>,
    pub output: Value,
}

#[derive(Debug, Clone)]
pub enum ElementwiseFnOption {
    Intrinsic(IntrinsicElementwise),
    Extern(String, ElType),
}

#[derive(Debug, Clone)]
pub enum IntrinsicElementwise {
    Add, Sub, Mul, Div, Neg, Rem,
    Pow, Exp, Log, Sqrt,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or, Xor, Not,
    LogicalAnd, LogicalOr, // short circuiting
    Shr, Shl,
    Ceil, Floor, Round,
    Max, Min,
    Cast(Type),
    Where,
}

impl IntrinsicElementwise {
    pub fn num_operands(&self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem | Self::Pow => 2,
            Self::Neg | Self::Not => 1,
            Self::Exp | Self::Log | Self::Sqrt => 1,
            Self::Eq | Self::Ne | Self::Lt | Self::Gt | Self::Le | Self::Ge => 2,
            Self::And | Self::Or | Self::Xor => 2,
            Self::LogicalAnd | Self::LogicalOr => 2,
            Self::Shr | Self::Shl => 2,
            Self::Ceil | Self::Floor | Self::Round => 1,
            Self::Max | Self::Min => 2,
            Self::Cast(_) => 1,
            Self::Where => 3,
        }
    }

    pub fn verify_type(&self, vals: &[Value]) -> Result<()> {
        if self.num_operands() != vals.len() {
            return Err(Error::msg(format!(
                "Intrinsic {:?} requires {} operands, got {}",
                self,
                self.num_operands(),
                vals.len()
            )));
        }
        let a = vals[0].type_of();
        if vals.iter().map(|x| x.type_of().shape() == a.shape()).all(|x| x) {
            return Err(Error::msg(format!(
                "Intrinsic {:?} requires all operands to have the same shape, got {:?}",
                self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
            )));
        }

        if self.num_operands() == 2 {
            let b = vals[1].type_of();
            let all_same = a == b;
            match self {
                Self::Add | Self::Sub => {
                    if !((a.is_float() && b.is_float()) || (a.is_int() && b.is_int()) || (a.is_ptr() && b.is_int()) || (a.is_int() && b.is_ptr())) {
                        return Err(Error::msg(format!(
                            "+ requires either floating + floating, int + int, int + ptr or ptr + int, got {:?}",
                            vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Mul | Self::Div | Self::Max | Self::Min => {
                    if !all_same && !(a.is_float() || a.is_int()) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and either floating or integral, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Rem => {
                    if !all_same && !a.is_int() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and integral, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Pow => {
                    if !all_same && !a.is_float() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and floating, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Eq | Self::Ne | Self::Lt | Self::Gt | Self::Le | Self::Ge => {
                    if !all_same {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::And | Self::Or | Self::Xor => {
                    if !(all_same && (a.is_int() || a.is_bool())) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and either integral or boolean, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::LogicalAnd | Self::LogicalOr => {
                    if !(all_same && a.is_bool()) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and boolean, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Shl | Self::Shr => {
                    if !(b.is_int() && (a.is_int() || a.is_ptr() || a.is_bool())) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires first operand to be either integral boolean or pointer and the second to be integral, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                _ => {}
            }
        } else if self.num_operands() == 3 {
            if let Self::Where = self {
                let b = vals[1].type_of();
                let c = vals[2].type_of();
                if !(a.is_bool() && b == c) {
                    return Err(Error::msg(format!(
                        "Intrinsic {:?} requires first operand to be boolean and the other two to have the same element type, got {:?}",
                        self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                    )));
                }
            }
        } else {
            match self {
                Self::Neg => {
                    if !a.is_float() && !a.is_int() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have either floating or integral element type, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Not => {
                    if !a.is_bool() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have boolean element type, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                },
                Self::Exp | Self::Log | Self::Sqrt | Self::Ceil | Self::Floor | Self::Round => {
                    if !a.is_float() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have floating element type, got {:?}",
                            self, vals.iter().map(|x| x.type_of()).collect::<Vec<_>>()
                        )));
                    }
                },
                Self::Cast(_) => {}
                _ => {}
            }
        }
        Ok(())
    }

    fn binary_upcast<'a>(a: &'a Type, b: &'a Type) -> &'a Type {
        if a.is_ptr() {
            a
        } else if b.is_ptr() {
            return b;
        } else if a.is_float() {
            return a;
        } else if b.is_float() {
            return b;
        } else if a.is_int() {
            return a;
        } else if b.is_int() {
            return b;
        } else {
            return a;
        }
    }

    pub fn return_type<'a>(&self, vals: &'a [Value]) -> &'a Type {
        if self.num_operands() == 2 {
            return Self::binary_upcast(vals[0].type_of(), vals[1].type_of());
        } else if self.num_operands() == 3 {
            if let Self::Where = self {
                return vals[1].type_of();
            }
            unreachable!();
        } else if self.num_operands() == 1 {
            return vals[0].type_of();
        }
        unreachable!()
    }
}

impl ElementWiseOp {
    pub fn build(f: ElementwiseFnOption, args: &[Value]) -> Result<Self> {
        let repr_arg = args.first().ok_or(Error::msg("args cannot be empty"))?;
        let oshape = repr_arg.type_of().shape().to_vec();
        if let ElementwiseFnOption::Intrinsic(op) = &f {
            op.verify_type(args)?;
        } // extern functions are not verified

        let return_type = match &f {
            ElementwiseFnOption::Intrinsic(a) => a.return_type(args).eltype.clone(),
            ElementwiseFnOption::Extern(_, a) => a.clone(),
        };

        let val = Value::new(Type::tensor(
            return_type,
            &oshape,
        ));
        Ok(ElementWiseOp {
            name: f,
            args: args.into(),
            output: val.clone(),
        })
    }
    
}

#[derive(Debug, Clone)]
pub struct FullOp {
    pub const_value: Constant,
    pub output: Value,
}

impl FullOp {
    pub fn build(value: impl Into<Constant>, shape: &[usize]) -> Self {
        let c: Constant = value.into();
        let ctype = Type::tensor(ElType::Val(c.dtype()), shape);
        let val = Value::new(ctype);
        FullOp {
            const_value: c,
            output: val.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArangeOp {
    pub begin: i32,
    pub end: i32,
    pub output: Value,
}

impl ArangeOp {
    pub fn build(begin: i32, end: i32) -> Result<Self> {
        if begin >= end {
            return Err(Error::msg(format!(
                "begin must be less than end, got {} and {}",
                begin, end
            )));
        }
        let val = Value::new(Type::tensor(ElType::Val(Dtype::I32), &[(end - begin) as usize]));
        Ok(ArangeOp {
            begin,
            end,
            output: val.clone(),
        })
    }
}


#[derive(Debug, Clone)]
pub enum ReduceOpOption {
    Sum,
    Prod,
    Min,
    Max,
    And,
    Or,
    Xor,
}

