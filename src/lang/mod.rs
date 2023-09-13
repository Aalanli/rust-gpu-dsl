use core::panic;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, Range};
use std::rc::Rc;

use anyhow::{Context, Error, Result};
use super::utils::Doc;

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

/// Type has value semantics, and are equal if they have the same data
/// Everying is a tensor type, scalars are represented by tensors with rank 0
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Type {
    eltype: ElType,
    shape: Vec<usize>, 
    // todo: Encoding
}

impl Type {
    pub fn scalar(eltype: ElType) -> Self {
        Type {
            eltype,
            shape: vec![],
        }
    }

    pub fn i32_scalar() -> Self {
        Type::scalar(ElType::Val(Dtype::I32))
    }

    pub fn f32_scalar() -> Self {
        Type::scalar(ElType::Val(Dtype::F32))
    }

    pub fn i32_ptr() -> Self {
        Type::scalar(ElType::Ptr(Dtype::I32))
    }

    pub fn f32_ptr() -> Self {
        Type::scalar(ElType::Ptr(Dtype::F32))
    }

    pub fn is_bool(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::I1) => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::F32) => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::I32) => true,
            _ => false,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn is_ptr(&self) -> bool {
        match &self.eltype {
            ElType::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn tensor(eltype: ElType, shape: &[usize]) -> Self {
        Type {
            eltype,
            shape: shape.into(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn to_pointer(&self) -> Self {
        let dtype = if let ElType::Val(dtype) = &self.eltype {
            ElType::Ptr(dtype.clone())
        } else {
            self.eltype.clone()
        };
        Type {
            eltype: dtype,
            shape: self.shape.clone(),
        }
    }

    pub fn to_value(&self) -> Self {
        let dtype = if let ElType::Ptr(dtype) = &self.eltype {
            ElType::Val(dtype.clone())
        } else {
            self.eltype.clone()
        };
        Type {
            eltype: dtype,
            shape: self.shape.clone(),
        }
    }

    pub fn eltype(&self) -> &ElType {
        &self.eltype
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ElType {
    Ptr(Dtype),
    Val(Dtype),
}

impl ElType {
    pub fn bits(&self) -> usize {
        match self {
            ElType::Ptr(_) => 64,
            ElType::Val(dtype) => match dtype {
                Dtype::I1 => 8,
                Dtype::I32 => 32,
                Dtype::F32 => 32,
            },
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Dtype {
    I1,
    I32,
    F32,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Constant {
    I32(i32),
    F32(f32),
    Bool(bool),
}

impl Constant {
    pub fn dtype(&self) -> Dtype {
        match self {
            Constant::I32(_) => Dtype::I32,
            Constant::F32(_) => Dtype::F32,
            Constant::Bool(_) => Dtype::I1,
        }
    }

    pub fn type_of(&self) -> Type {
        Type::scalar(ElType::Val(self.dtype()))
    }
}

impl From<i32> for Constant {
    fn from(val: i32) -> Self {
        Constant::I32(val)
    }
}

impl From<f32> for Constant {
    fn from(val: f32) -> Self {
        Constant::F32(val)
    }
}

impl From<bool> for Constant {
    fn from(val: bool) -> Self {
        Constant::Bool(val)
    }
}


// pub type Op = Rc<Operation>;

#[derive(Debug, Clone)]
pub struct Op(Rc<Operation>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(usize);


#[derive(Debug, Clone)]
pub struct Block(Rc<BlockImpl>);

impl Deref for Block {
    type Target = BlockImpl;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Block {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Block {}

#[derive(Debug, Clone)]
pub struct BlockImpl {
    pub args: Vec<Value>,
    pub body: Vec<Op>,
}


impl Op {
    pub fn new(op: OpEnum, location: Location) -> Self {
        Op(Rc::new(Operation::new(op, location)))
    }
}

impl Deref for Op {
    type Target = Operation;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Op {}

#[derive(Debug)]
pub struct Operation {
    location: Location,
    op: OpEnum,
}

impl Operation {
    pub fn new(op: OpEnum, location: Location) -> Self {
        Operation { op, location }
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct AssignOp {
    pub lhs: Value,
    pub rhs: Value,
}


#[derive(Debug)]
pub struct ForOp {
    pub induction_var: Value,
    pub start: Value,
    pub end: Value,
    pub step: Value,
    pub body: Block,
}

#[derive(Debug)]
pub struct IfOp {
    pub cond: Value,
    pub then: Block,
    pub else_: Block,
}

#[derive(Debug)]
pub struct FunctionOp {
    pub name: String,
    pub body: Block
}

#[derive(Debug)]
pub struct ProgramIDOp {
    pub output: Value,
}

impl ProgramIDOp {
    pub fn build() -> Self {
        let val = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        ProgramIDOp { output: val.clone() }
    }

    
}

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub struct PermuteOp {
    input: Value,
    permutation: Vec<usize>,
    output: Value,
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

#[derive(Debug)]
pub struct SliceOp {
    input: Value,
    slices: Vec<Range<usize>>,
    output: Value,
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

#[derive(Debug)]
pub struct ExpandOp {
    input: Value,
    dim: usize,
    output: Value,
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

#[derive(Debug)]
pub struct BroadcastOp {
    input: Value,
    other: Value,
    output: Value,
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

#[derive(Debug)]
pub struct ReduceOp {
    input: Value,
    dim: usize,
    op: ReduceOpOption,
    output: Value,
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

#[derive(Debug)]
pub struct DotOp {
    a: Value,
    b: Value,
    output: Value,
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
            &vec![a.type_of().shape()[0], b.type_of().shape()[1]],
        ));
        Ok(DotOp {
            a: a.clone(),
            b: b.clone(),
            output: val.clone(),
        })
    }
}

#[derive(Debug)]
pub struct ElementWiseOp {
    pub name: ElementwiseFnOption,
    pub args: Vec<Value>,
    pub output: Value,
}

#[derive(Debug)]
pub enum ElementwiseFnOption {
    Intrinsic(IntrinsicElementwise),
    Extern(String, ElType),
}

#[derive(Debug)]
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
            return a;
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
            return Self::binary_upcast(&vals[0].type_of(), &vals[1].type_of());
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

#[derive(Debug)]
pub struct FullOp {
    const_value: Constant,
    output: Value,
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

#[derive(Debug)]
pub struct ArangeOp {
    begin: i32,
    end: i32,
    output: Value,
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


#[derive(Debug)]
pub enum ReduceOpOption {
    Sum,
    Prod,
    Min,
    Max,
    And,
    Or,
    Xor,
}



#[derive(Clone, Debug)]
pub struct Value(Rc<ValueImpl>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl Value {
    fn new(type_of: Type) -> Self {
        Value(Rc::new(ValueImpl {
            type_of,
            name_hint: RefCell::new(None),
        }))
    }

    pub fn type_of(&self) -> &Type {
        &self.0.type_of
    }

    pub fn name_hint(self, name: impl Into<String>) -> Self {
        self.0.name_hint.replace(Some(name.into()));
        self
    }

    pub fn name(&self) -> Option<String> {
        self.0.name_hint.borrow().clone()
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}

#[derive(Debug)]
struct ValueImpl {
    type_of: Type,
    name_hint: RefCell<Option<String>>,
}

#[derive(Debug)]
pub struct Function {
    name: String,
    args: Vec<Value>,
    returns: Vec<Type>,
    body: Vec<AST>,
}

pub struct FunctionBuilder {
    name: String,
    args: Vec<Value>,
    scope: Vec<Vec<AST>>,
}

macro_rules! binary_builder {
    ($fn_name:ident, $enum_name:ident) => {
        pub fn $fn_name(&mut self, a: &Value, b: &Value) -> Result<Value> {
            let loc = std::panic::Location::caller().into();
            let openum = OpEnum::ElementWise(ElementWiseOp::build(
                ElementwiseFnOption::Intrinsic(IntrinsicElementwise::$enum_name),
                &[a.clone(), b.clone()],
            )?);
            let val = openum.outputs().pop().unwrap().clone();
            let op = AST::Op(openum, loc);
            self.push_node(op)?;
            Ok(val)
        }
    };
}

macro_rules! unary_builder {
    ($fn_name:ident, $enum_name:ident) => {
        pub fn $fn_name(&mut self, a: &Value) -> Result<Value> {
            let loc = std::panic::Location::caller().into();
            let openum = OpEnum::ElementWise(ElementWiseOp::build(
                ElementwiseFnOption::Intrinsic(IntrinsicElementwise::$enum_name),
                &[a.clone()],
            )?);
            let val = openum.outputs().pop().unwrap().clone();
            let op = AST::Op(openum, loc);
            self.push_node(op)?;
            Ok(val)
        }
    };
}

impl FunctionBuilder {
    binary_builder!(add, Add);
    binary_builder!(sub, Sub);
    binary_builder!(mul, Mul);
    binary_builder!(div, Div);
    binary_builder!(rem, Rem);
    binary_builder!(pow, Pow);
    binary_builder!(eq, Eq);
    binary_builder!(ne, Ne);
    binary_builder!(lt, Lt);
    binary_builder!(gt, Gt);
    binary_builder!(le, Le);
    binary_builder!(ge, Ge);
    binary_builder!(and, And);
    binary_builder!(or, Or);
    binary_builder!(logical_and, LogicalAnd);
    binary_builder!(logical_or, LogicalOr);
    binary_builder!(xor, Xor);
    binary_builder!(shr, Shr);
    binary_builder!(shl, Shl);
    binary_builder!(max, Max);
    binary_builder!(min, Min);

    unary_builder!(neg, Neg);
    unary_builder!(not, Not);
    unary_builder!(exp, Exp);
    unary_builder!(log, Log);
    unary_builder!(sqrt, Sqrt);
    unary_builder!(ceil, Ceil);
    unary_builder!(floor, Floor);
    unary_builder!(round, Round);

    pub fn where_(&mut self, cond: &Value, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::ElementWise(ElementWiseOp::build(
            ElementwiseFnOption::Intrinsic(IntrinsicElementwise::Where),
            &[cond.clone(), a.clone(), b.clone()],
        )?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn new(name: impl Into<String>) -> Self {
        FunctionBuilder {
            name: name.into(),
            args: vec![],
            scope: vec![vec![]],
        }
    }

    pub fn arg<const N: usize>(&mut self, args: [Type; N]) -> [Value; N] {
        let values: [Value; N] = args.map(|ty| Value::new(ty.clone()));
        for i in &values {
            self.args.push(i.clone());
        }
        values
    }

    fn push_node(&mut self, op: AST) -> Result<()> {
        self.scope.last_mut().ok_or(Error::msg("scope is empty"))?.push(op);
        Ok(())
    }

    pub fn program_id(&mut self) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::ProgramID(ProgramIDOp::build());
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn load(
        &mut self,
        ptr: &Value,
        mask: Option<&Value>,
        value: Option<&Value>,
    ) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Load(LoadOp::build(ptr, mask, value)?);        
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(
            openum,
            loc,
        );
        self.push_node(op)?;
        Ok(val)
    }

    pub fn store(&mut self, ptr: &Value, value: &Value, mask: Option<&Value>) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Store(StoreOp::build(ptr, value, mask)?);
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(())
    }

    pub fn reshape(&mut self, input: &Value, shape: &[i32]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Reshape(ReshapeOp::build(input, shape)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn permute(&mut self, input: &Value, permutation: &[u32]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Permute(PermuteOp::build(input, permutation)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    /// Range: a..b
    /// RangeTo: 0..b
    /// RangeFrom: a..-1
    /// RangeFull: 0..-1
    /// Index and remove dim: a..a
    pub fn slice(&mut self, input: &Value, slices: &[Range<i32>]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Slice(SliceOp::build(input, slices)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn expand(&mut self, input: &Value, dims: i32) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Expand(ExpandOp::build(input, dims)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn broadcast(&mut self, input: &Value, other: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Broadcast(BroadcastOp::build(input, other)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn reduce(&mut self, input: &Value, dim: i32, op: ReduceOpOption) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Reduce(ReduceOp::build(input, dim, op)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn dot(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Dot(DotOp::build(a, b)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn arange(&mut self, begin: i32, end: i32) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Arange(ArangeOp::build(begin, end)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn full(&mut self, value: impl Into<Constant>, shape: &[usize]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::Full(FullOp::build(value, shape));
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn for_(
        &mut self,
        begin: &Value,
        end: &Value,
        step: &Value,
        scope: impl Fn(&mut Self, &Value) -> Result<()>,
    ) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let induction_var = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        self.scope.push(vec![]);
        scope(self, &induction_var)?;
        let for_scope = self.scope.pop().unwrap();
        let op = AST::Stmt(
            Stmt::For(
                induction_var.clone(),
                begin.clone(),
                end.clone(),
                step.clone(),
                for_scope,
            ),
            loc,
        );
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn while_(&mut self, cond: impl FnOnce(&mut Self) -> Result<Value>, body: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        self.scope.push(vec![]);
        let cond = cond(self)?;
        let cond_block = self.scope.pop().unwrap();
        self.scope.push(vec![]);
        body(self)?;
        let while_block = self.scope.pop().unwrap();
        let op = AST::Stmt(Stmt::While(cond_block, cond.clone(), while_block), loc);
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn return_(&mut self, values: &[Value]) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Return(values.into()), loc);
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn if_(
        &mut self,
        cond: &Value,
        then: impl FnOnce(&mut Self) -> Result<()>,
        else_: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        self.scope.push(vec![]);
        then(self)?;
        let then_scope = self.scope.pop().unwrap();
        self.scope.push(vec![]);
        else_(self)?;
        let else_scope = self.scope.pop().unwrap();
        let op = AST::Stmt(Stmt::If(cond.clone(), then_scope, else_scope), loc);
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn break_(&mut self) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Break, loc);
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn assign(&mut self, lhs: &Value, rhs: &Value) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Assign(lhs.clone(), rhs.clone()), loc);
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn extern_elementwise(&mut self, args: &[Value], op_code: &str, return_type: ElType) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::ElementWise(ElementWiseOp::build(ElementwiseFnOption::Extern(op_code.into(), return_type), args)?);
        let val = openum.outputs().pop().unwrap().clone();
        let op = AST::Op(openum, loc);
        self.push_node(op)?;
        Ok(val)
    }

    pub fn visit_terminals(ast: &AST, f: &mut impl FnMut(&AST)) { 
        // if not &mut then f will raise a recursion error in memorization
        // as inner recursive functions needs to be &mut f
        match ast {
            AST::Op(_, _) => f(ast),
            AST::Stmt(stmt, _) => match stmt {
                Stmt::For(_, _, _, _, body) => {
                    for ast in body {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                }
                Stmt::While(cond_block, _, body) => {
                    for ast in body {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                    for ast in cond_block {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                }
                Stmt::If(_, then, else_) => {
                    for ast in then {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                    for ast in else_ {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                }
                Stmt::Assign(_, _) => f(ast),
                Stmt::Return(_) => f(ast),
                Stmt::Break => f(ast),
                Stmt::Continue => f(ast),
                Stmt::Function(func) => {
                    for ast in &func.body {
                        FunctionBuilder::visit_terminals(ast, f);
                    }
                },
            },
        }
    }

    fn verify_function_return(body: &Vec<AST>) -> Result<Vec<Type>> {
        let mut returns = Ok(vec![]);
        let mut seen = false;
        for node in body {
            Self::visit_terminals(node, &mut |node| {
                if let AST::Stmt(Stmt::Return(values), _) = node {
                    if !seen && returns.is_ok() {
                        let types = values.iter().map(|x| x.type_of().clone()).collect();
                        returns = Ok(types);
                        seen = true;
                    } else if seen && returns.is_ok() {
                        let types: Vec<_> = values.iter().map(|x| x.type_of().clone()).collect();
                        if returns.as_ref().unwrap() != &types {
                            returns = Err(Error::msg(format!(
                                "Return types mismatch, got {:?} and {:?}",
                                returns.as_ref().unwrap(),
                                types
                            )));
                        }
                    }
                }
            });
        }
        returns
    }

    pub fn build(mut self) -> Result<Function> {
        if self.scope.len() != 1 {
            return Err(Error::msg("Internal error, scope should only have one element"));
        }
        
        Ok(Function {
            name: self.name,
            args: self.args,
            returns: Self::verify_function_return(self.scope.last().unwrap())?,
            body: self.scope.pop().unwrap(),
        })
    }

    pub fn build_body(mut self) -> Result<Vec<AST>> {
        if self.scope.len() != 1 {
            return Err(Error::msg("Internal error, scope should only have one element"));
        }
        Ok(self.scope.pop().unwrap())
    }

    pub fn insert_block(&mut self, body: Vec<AST>) {
        self.scope.last_mut().unwrap().extend(body);
    }

    pub fn insert_node(&mut self, node: AST) {
        self.scope.last_mut().unwrap().push(node);
    }

    pub fn pop_node(&mut self) -> Option<AST> {
        self.scope.last_mut().unwrap().pop()
    }

    pub fn set_last_loc(&mut self, loc: Location) {
        if let Some(node) = self.scope.last_mut().unwrap().last_mut() {
            match node {
                AST::Op(_, loc_) => *loc_ = loc,
                AST::Stmt(_, loc_) => *loc_ = loc,
            }
        }
    }
}

fn replace_all_uses_with(nodes: &mut [AST], from: &Value, to: &Value) {
    for node in nodes {
        match node {
            AST::Op(op, _) => {
                for input in op.inputs_mut() {
                    if input == from {
                        *input = to.clone();
                    }
                }
            }
            AST::Stmt(stmt, _) => {
                match stmt {
                    Stmt::For(a, b, c, d, body) => {
                        for node in [a, b, c, d] {
                            if node == from {
                                *node = to.clone();
                            }
                        }
                        replace_all_uses_with(body, from, to);
                    }
                    Stmt::While(cond_block, cond, body) => {
                        if cond == from {
                            *cond = to.clone();
                        }
                        replace_all_uses_with(cond_block, from, to);
                        replace_all_uses_with(body, from, to);
                    }
                    Stmt::If(cond, then, else_) => {
                        if cond == from {
                            *cond = to.clone();
                        }
                        replace_all_uses_with(then, from, to);
                        replace_all_uses_with(else_, from, to);
                    }
                    Stmt::Assign(lhs, rhs) => {
                        if lhs == from {
                            *lhs = to.clone();
                        }
                        if rhs == from {
                            *rhs = to.clone();
                        }
                    }
                    Stmt::Return(values) => {
                        for value in values {
                            if value == from {
                                *value = to.clone();
                            }
                        }
                    }
                    Stmt::Break => {}
                    Stmt::Continue => {}
                    Stmt::Function(func) => {
                        replace_all_uses_with(&mut func.body, from, to);
                    }
                }
            }
        }
    }
}


fn remove_predicated_break_continue(
    stmt: &mut Stmt,
    break_cond_var: &Value, 
    continue_cond_var: &Value
) -> (bool, bool) { // (has_break, has_continue)

    let mut has_break = false;
    let mut has_continue = false;
    match stmt {
        Stmt::If(cond, then, else_) => {
            let mut i = 0;
            while i < then.len() {
                if let Some(Stmt::Break) = then[i].get_stmt_ref() {
                    then[i] = AST::Stmt(Stmt::Assign(break_cond_var.clone(), cond.clone()), then[i].loc().clone());
                    has_break = true;
                    break;
                } else if let Some(Stmt::Continue) = then[i].get_stmt_ref() {
                    then[i] = AST::Stmt(Stmt::Assign(continue_cond_var.clone(), cond.clone()), then[i].loc().clone());
                    has_continue = true;
                    break;
                }
                i += 1;
            }
            then.truncate(i + 1);
            let (has_break_, has_continue_) = hoist_break_continue(then, Some(break_cond_var), Some(continue_cond_var));
            has_break |= has_break_;
            has_continue |= has_continue_;

            i = 0;
            while i < else_.len() {
                let loc = else_[i].loc();
                if let Some(Stmt::Break) = else_[i].get_stmt_ref() {
                    let mut builder = FunctionBuilder::new("break");
                    let not_val = builder.not(cond).unwrap();
                    builder.set_last_loc(loc.clone());
                    builder.assign(break_cond_var, &not_val).unwrap();
                    builder.set_last_loc(loc.clone());
                    else_.splice(i..i+1, builder.build_body().unwrap());
                    has_break = true;
                    break;
                } else if let Some(Stmt::Continue) = else_[i].get_stmt_ref() {
                    let mut builder = FunctionBuilder::new("break");
                    let not_val = builder.not(cond).unwrap();
                    builder.set_last_loc(loc.clone());
                    builder.assign(continue_cond_var, &not_val).unwrap();
                    builder.set_last_loc(loc.clone());
                    else_.splice(i..i+1, builder.build_body().unwrap());
                    has_continue = true;
                    break;
                }
                i += 1;
            }

            else_.truncate(i + 1);
            let (has_break_, has_continue_) = hoist_break_continue(then, Some(break_cond_var), Some(continue_cond_var));
            has_break |= has_break_;
            has_continue |= has_continue_;
        }
        _ => {}
    }
    (has_break, has_continue)
}

fn protect_block_break_continue(
    nodes: &mut Vec<AST>,
    break_cond_var: &Value,
    continue_cond_var: &Value,
) -> (bool, bool) { // (has_break, has_continue)
    let mut has_break = false;
    let mut has_continue = false;
    let mut i = 0;
    while i < nodes.len() {
        let loc = nodes[i].loc().clone();
        if let Some(stmt) = nodes[i].get_stmt_mut() {
            if let Stmt::If(_, _, _) = stmt {
                let (has_break_, has_continue_) = remove_predicated_break_continue(stmt, break_cond_var, continue_cond_var);
                has_break |= has_break_;
                has_continue |= has_continue_;
            } else if let Stmt::Break = stmt {
                let const_true = OpEnum::Full(FullOp::build(true, &[]));
                let const_true_val = const_true.outputs()[0].clone();
                nodes.insert(i, AST::Op(const_true, loc.clone()));
                nodes[i + 1] = AST::Stmt(Stmt::Assign(break_cond_var.clone(), const_true_val), loc.clone()); 
                i += 1;
                has_break = true;
            } else if let Stmt::Continue = stmt {
                let const_true = OpEnum::Full(FullOp::build(true, &[]));
                let const_true_val = const_true.outputs()[0].clone();
                nodes.insert(i, AST::Op(const_true, loc.clone()));
                nodes[i + 1] = AST::Stmt(Stmt::Assign(continue_cond_var.clone(), const_true_val), loc.clone()); 
                i += 1;
                has_continue = true;
            }
        }

        if has_break || has_continue {
            let mut builder = FunctionBuilder::new("break");
            let mut rest_body: Vec<_> = nodes.drain(i..).collect();
            let (has_break_, has_continue_) = hoist_break_continue(&mut rest_body, Some(break_cond_var), Some(continue_cond_var));
            has_break |= has_break_;
            has_continue |= has_continue_;
            
            let has_break_or_continue = builder.or(break_cond_var, continue_cond_var).unwrap();
            builder.set_last_loc(loc.clone());
            let no_break_or_continue = builder.not(&has_break_or_continue).unwrap();
            builder.set_last_loc(loc.clone());

            builder.if_(&no_break_or_continue, |b| {
                for node in rest_body {
                    b.insert_node(node);
                }
                Ok(())
            }, |_| {Ok(())}).unwrap();
            builder.set_last_loc(loc.clone());

            nodes.extend(builder.build_body().unwrap());
            break;
        }
        i += 1;
    }
    (has_break, has_continue)

}

fn hoist_break_continue(
    nodes: &mut Vec<AST>, 
    break_var: Option<&Value>,
    continue_var: Option<&Value>,
) -> (bool, bool) { // (has_break, has_continue)
    let mut has_break = false;
    let mut has_continue = false;
    let mut i = 0;
    while i < nodes.len() {
        match &mut nodes[i] {
            AST::Op(_, _) => {},
            AST::Stmt(stmt, _) => {
                match stmt {
                    Stmt::For(ind_var, start, end, step, body) => {
                        let mut builder = FunctionBuilder::new("hoist_break_continue");
                        let break_cond_var = builder.full(false, &[]).unwrap();
                        let continue_cond_var = builder.full(false, &[]).unwrap();
                        let (inner_break, _inner_continue) = hoist_break_continue(body, Some(&break_cond_var), Some(&continue_cond_var));
                        if inner_break {
                            let mut new_body = vec![];
                            // steal body
                            std::mem::swap(&mut new_body, body);
                            let new_ind_var = builder.full(0, &[]).unwrap();
                            builder.assign(&new_ind_var, start).unwrap();
                            replace_all_uses_with(&mut new_body, &ind_var, &new_ind_var);
                            // let ind_cond = builder.elementwise(&[new_ind_var.clone(), end.clone()], "lt").unwrap();
                            // let cond_var = builder.elementwise(&[ind_cond, break_cond_var.clone()], "and").unwrap();
                            // let cond_var = builder.elementwise(&[cond_var, continue_cond_var.clone()], "and").unwrap();
                            // builder.while_(&cond_var, |builder| {
                            //     for node in new_body {
                            //         builder.insert_node(node);
                            //     }
                            //     let new_ind_var_ = builder.elementwise(&[new_ind_var.clone(), step.clone()], "add").unwrap();
                            //     builder.assign(&new_ind_var, &new_ind_var_).unwrap();
                            //     let ind_cond = builder.elementwise(&[new_ind_var.clone(), end.clone()], "lt").unwrap();
                            //     let cond_var_ = builder.elementwise(&[ind_cond, break_cond_var], "and").unwrap();
                            //     let cond_var_ = builder.elementwise(&[cond_var_, continue_cond_var], "and").unwrap();
                            //     builder.assign(&cond_var, &cond_var_).unwrap();
                            //     Ok(())
                            // }).unwrap();
                        }


                    }
                    _ => {}
                }            
            }
        }
    }
    if break_var.is_some() && continue_var.is_some() {
        let (has_break_, has_continue_) = protect_block_break_continue(nodes, break_var.unwrap(), continue_var.unwrap());
        has_break |= has_break_;
        has_continue |= has_continue_;
    }

    (has_break, has_continue)
}

fn hoist_break(mut f: Function) -> Result<Function> {
    for node in f.body.iter_mut() {
        match node {
            AST::Op(_, _) => {},
            AST::Stmt(stmt, _) => {
                match stmt {
                    Stmt::For(_, _, _, _, body) => {
                        for i in 0..body.len() {
                            if let AST::Stmt(Stmt::Break, _) = body[i] {
                                body.truncate(i);
                                break;
                            }
                            
                        }

                    }
                    _ => {}
                }
            }
        }
    }
    todo!()
}

fn collect_outside_uses(f: &AST, uses: &mut HashSet<Value>, internal: &mut HashSet<Value>) {
    let mut add_use = |v| {
            if !internal.contains(v) {
                uses.insert(v.clone());
            }
        };
    match f {
        AST::Op(op, _) => {
            for input in op.inputs() {
                add_use(input);
            }
            for output in op.outputs() {
                internal.insert(output.clone());
            }
        }
        AST::Stmt(stmt, _) => {
            match stmt {
                Stmt::For(ind_var, start, end, step, body) => {
                    add_use(start);
                    add_use(end);
                    add_use(step);
                    internal.insert(ind_var.clone());
                    for node in body {
                        collect_outside_uses(node, uses, internal);
                    }
                }
                Stmt::While(cond, cond_res, body) => {
                    internal.insert(cond_res.clone());
                    for node in cond {
                        collect_outside_uses(node, uses, internal);
                    }
                    for node in body {
                        collect_outside_uses(node, uses, internal);
                    }
                }
                Stmt::If(cond, then, else_) => {
                    add_use(cond);
                    for node in then {
                        collect_outside_uses(node, uses, internal);
                    }
                    for node in else_ {
                        collect_outside_uses(node, uses, internal);
                    }
                }
                Stmt::Assign(lhs, rhs) => {
                    add_use(lhs);
                    add_use(rhs);
                }
                _ => {}
            }
        }
    }
}

fn convert_to_operation(mut f: AST, remap: &mut HashMap<Value, Value>) -> Result<Op> {
    let mut get_remap = |v| {
        if let Some(v) = remap.get(&v) {
            v.clone()
        } else {
            v
        }
    };

    match f {
        AST::Op(op, loc) => { todo!() },
        AST::Stmt(stmt, loc) => {
            match stmt {
                Stmt::For(ind_var, start, end, step, body) => {
                    let mut builder = FunctionBuilder::new("for");

                },
                _ => {}
                // Stmt::While(_, _, _) => todo!(),
                // Stmt::If(_, _, _) => todo!(),
                // Stmt::Assign(_, _) => todo!(),
                // Stmt::Return(_) => todo!(),
                // Stmt::Break => todo!(),
                // Stmt::Continue => todo!(),
                // Stmt::Function(_) => todo!(),
            }
        }
    }

    todo!()
}


pub struct FlowAnalysis {
    pub source: HashMap<Value, Op>,
    pub uses: HashMap<Value, Vec<Op>>,
    pub parent_block: HashMap<Op, Block>,
    pub block_parent: HashMap<Block, Op>,
}

impl FlowAnalysis {
    fn visit(&mut self, op: &Op, parent_block: Option<&Block>) {
        for outputs in op.outputs() {
            self.source.insert(outputs.clone(), op.clone());
        }
        for input in op.inputs() {
            if let Some(o) = self.source.get(input) {
                self.uses.entry(input.clone()).or_default().push(o.clone());
            }
        }

        if let Some(parent_block) = parent_block {
            self.parent_block.insert(op.clone(), parent_block.clone());
        }

        for block in op.blocks() {
            self.block_parent.insert(block.clone(), op.clone());
            for op in block.body.iter() {
                self.visit(op, Some(block));
            }
        }
    }

    pub fn new(op: &Op) -> Self {
        let mut s = Self {
            source: HashMap::new(),
            uses: HashMap::new(),
            parent_block: HashMap::new(),
            block_parent: HashMap::new(),
        };
        s.visit(op, None);
        s
    }

    pub fn val_source(&self, val: &Value) -> Option<&Op> {
        self.source.get(val)
    }

    pub fn val_uses(&self, val: &Value) -> Option<&Vec<Op>> {
        self.uses.get(val)
    }
}

pub fn collect(op: &Op, f: impl Fn(&Op) -> bool) -> Vec<Op> {
    let mut ops = vec![];
    let mut worklist = vec![op];
    while let Some(op) = worklist.pop() {
        if f(op) {
            ops.push(op.clone());
            worklist.extend(op.blocks().iter().flat_map(|x| x.body.iter()));
        }
    }
    ops
}

pub struct DCE {
    pub value_live: HashSet<Value>,
    pub op_live: HashSet<Op>,
}

impl DCE {

    pub fn is_live(&self, val: &Value) -> bool {
        self.value_live.contains(val)
    }

    pub fn mark_live(&mut self, val: &Value) {
        self.value_live.insert(val.clone());
    }


    pub fn on_op(&mut self, op: &Op, flow: &FlowAnalysis) {
        match &op.op {
            OpEnum::Store(_) => {
                for i in op.inputs() {
                    self.mark_live(i);
                }
            }
            OpEnum::If(if_op) => {
                for i in if_op.yields.iter() {
                    if self.is_live(i) {
                        let yarg = &if_op.yield_to_term_arg[i];
                        self.mark_live(yarg);
                    }
                }
                self.on_block(&if_op.then, flow);
                self.on_block(&if_op.else_, flow);
                for i in if_op.then.args.iter().chain(if_op.else_.args.iter()) {
                    if self.is_live(i) {
                        let yarg = &if_op.block_arg_to_carry[i];
                        self.mark_live(yarg);
                    }
                }
                if if_op.carries.iter().any(|x| self.is_live(x)) {
                    self.mark_live(&if_op.cond);
                }
            }
            OpEnum::Loop(loop_op) => {
                if let Terminator::LoopYield(loop_term) = &loop_op.body.terminator {
                    for (i, v) in loop_op.yields.iter().enumerate() {
                        if self.is_live(v) {
                            self.mark_live(&loop_term.rest[i]);
                        }
                    }
                    self.on_block(&loop_op.body, flow);
                    for (i, v) in loop_op.body.args.iter().enumerate() {
                        if self.is_live(v) {
                            self.mark_live(&loop_op.uses[i]);
                        }
                    }
                } else {
                    panic!();
                }
            }
            _ => {
                if op.outputs().into_iter().any(|x| self.is_live(x)) {
                    self.op_live.insert(op.clone());
                    for input in op.inputs() {
                        self.mark_live(input);
                    }
                }
            }
        }
    }

    pub fn on_block(&mut self, block: &Block, flow: &FlowAnalysis) {
        match &block.terminator {
            Terminator::LoopYield(LoopYield { continue_cond, rest }) => {
                self.mark_live(continue_cond);
                loop {
                    let yield_live = rest.iter().map(|x| self.is_live(x));
                    let arg_live = block.args.iter().map(|x| self.is_live(x));
                    let changed = yield_live.zip(arg_live).any(|(x, y)| x != y);
                    if !changed {
                        break;
                    }
                    
                    for op in block.body.iter().rev() {
                        self.on_op(op, flow);
                    }

                }
            }
            Terminator::Yield(Yield { values }) => {
            }
        }
    }

    // pub fn run(op: &Op, flow: &FlowAnalysis) -> Self {
    //     let mut v_live = HashSet::new();
    //     let mut op_live = HashSet::new();


    //     let mut worklist = collect(op, |op| {
    //         if let OpEnum::Store(_) = op.op {
    //             true
    //         } else {
    //             false
    //         }
    //     });

    //     while let Some(op) = worklist.pop() {
    //         if !op_live.contains(&op) {
    //             op_live.insert(op.clone());
    //             for input in op.inputs() {
    //                 v_live.insert(input.clone());
    //                 if let Some(o) = flow.val_source(input) {
    //                     worklist.push(o.clone());
    //                 }
    //             }

    //             if let Some(par_block) = flow.parent_block.get(&op) {
    //                 if let Some(par_op) = flow.block_parent.get(par_block) {
    //                     worklist.push(par_op.clone());
    //                 }
    //             }
    //         }

    //     }
        

    //     Self { value_live: v_live, op_live }
    // }
}

pub enum Encoding {
    Shared(SharedEncoding),
    Local(LocalEncoding),
}

pub struct SharedEncoding {}
pub struct LocalEncoding {}


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
