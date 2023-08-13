use std::cell::RefCell;
use std::ops::{Deref, DerefMut, Range};
use std::rc::Rc;

use anyhow::{Context, Error, Result};

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
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ElType {
    Ptr(Dtype),
    Val(Dtype),
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

#[derive(Debug)]
pub enum AST {
    Op(OpEnum, Location),
    Stmt(Stmt, Location),
}

impl AST {
    pub fn is_stmt(&self) -> bool {
        match self {
            Self::Stmt(_, _) => true,
            _ => false,
        }
    }

    pub fn is_op(&self) -> bool {
        match self {
            Self::Op(_, _) => true,
            _ => false,
        }
    }

    pub fn get_op(self) -> Result<OpEnum> {
        match self {
            Self::Op(op, _) => Ok(op),
            _ => Err(Error::msg("AST is not an Op")),
        }
    }

    pub fn get_stmt(self) -> Result<Stmt> {
        match self {
            Self::Stmt(stmt, _) => Ok(stmt),
            _ => Err(Error::msg("AST is not a Stmt")),
        }
    }

    pub fn get_stmt_ref(&self) -> Option<&Stmt> {
        match self {
            Self::Stmt(stmt, _) => Some(stmt),
            _ => None,
        }
    }

    pub fn get_stmt_mut(&mut self) -> Option<&mut Stmt> {
        match self {
            Self::Stmt(stmt, _) => Some(stmt),
            _ => None,
        }
    }

    pub fn loc(&self) -> &Location {
        match self {
            Self::Op(_, loc) => loc,
            Self::Stmt(_, loc) => loc,
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    Function(Function),
    For(Value, Value, Value, Value, Vec<AST>), // induction_var, start, end, step, body
    While(Value, Vec<AST>),                    // condition, body
    If(Value, Vec<AST>, Vec<AST>),             // condition, then, else
    Assign(Value, Value),                      // lhs, rhs
    Return(Vec<Value>),
    Break,
    Continue
}

impl Stmt {
    
}

#[derive(Debug)]
pub enum OpEnum {
    ProgramID(ProgramIDOp), // output

    Load(LoadOp), // ptr, mask, value, output
    Store(StoreOp),               // ptr, value, mask

    Reshape(ReshapeOp),      // input, shape, output
    Permute(PermuteOp),      // input, permutation, output
    Slice(SliceOp), // input, (begin, end), output
    Expand(ExpandOp),              // input, dim, output
    Broadcast(BroadcastOp),         // input, other, output

    Reduce(ReduceOp), // input, dims, op, output
    ElementWise(ElementWiseOp),          // for extensibility reasons
    Dot(DotOp),            // a @ b = c

    Full(FullOp),   // const_value, output
    Arange(ArangeOp), // begin, end, output
}

impl OpEnum {
    pub fn inputs(&self) -> Vec<&Value> {
        match self {
            Self::ProgramID(op) => vec![],
            Self::Load(op) => { 
                let mut output = vec![&op.ptr];
                if let Some(mask) = &op.mask {
                    output.push(mask);
                }
                if let Some(value) = &op.value {
                    output.push(value);
                }
                output
            },

            Self::Store(op) => {
                let mut output = vec![&op.ptr, &op.value];
                if let Some(mask) = &op.mask {
                    output.push(mask);
                }
                output
            },

            Self::Reshape(op) => vec![&op.input],
            Self::Permute(op) => vec![&op.input],
            Self::Slice(op) => vec![&op.input],
            Self::Expand(op) => vec![&op.input],
            Self::Broadcast(op) => vec![&op.input, &op.other],
            Self::Reduce(op) => vec![&op.input],
            Self::ElementWise(op) => op.args.iter().collect(),
            Self::Dot(op) => vec![&op.a, &op.b],
            Self::Full(op) => vec![],
            Self::Arange(op) => vec![],
        }
    }

    pub fn inputs_mut(&mut self) -> Vec<&mut Value> {
        match self {
            Self::ProgramID(op) => vec![],
            Self::Load(op) => { 
                let mut output = vec![&mut op.ptr];
                if let Some(mask) = &mut op.mask {
                    output.push(mask);
                }
                if let Some(value) = &mut op.value {
                    output.push(value);
                }
                output
            },

            Self::Store(op) => {
                let mut output = vec![&mut op.ptr, &mut op.value];
                if let Some(mask) = &mut op.mask {
                    output.push(mask);
                }
                output
            },

            Self::Reshape(op) => vec![&mut op.input],
            Self::Permute(op) => vec![&mut op.input],
            Self::Slice(op) => vec![&mut op.input],
            Self::Expand(op) => vec![&mut op.input],
            Self::Broadcast(op) => vec![&mut op.input, &mut op.other],
            Self::Reduce(op) => vec![&mut op.input],
            Self::ElementWise(op) => op.args.iter_mut().collect(),
            Self::Dot(op) => vec![&mut op.a, &mut op.b],
            Self::Full(op) => vec![],
            Self::Arange(op) => vec![],
        }
    }

    pub fn outputs(&self) -> Vec<&Value> {
        match self {
            Self::ProgramID(op) => vec![&op.output],
            Self::Load(op) => vec![&op.output],
            Self::Store(op) => vec![],
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
        }
    }
}

#[derive(Debug)]
pub struct ProgramIDOp {
    output: Value,
}

impl ProgramIDOp {
    pub fn build() -> Self {
        let val = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        ProgramIDOp { output: val.clone() }
    }
}

#[derive(Debug)]
pub struct LoadOp {
    ptr: Value,
    mask: Option<Value>,
    value: Option<Value>,
    output: Value,
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
    ptr: Value,
    value: Value,
    mask: Option<Value>,
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
    input: Value,
    shape: Vec<usize>,
    output: Value,
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
    pub name: String,
    pub args: Vec<Value>,
    pub output: Value,
}

impl ElementWiseOp {
    pub fn build(name: impl Into<String>, args: &[Value]) -> Result<Self> {
        let repr_arg = args.first().ok_or(Error::msg("args cannot be empty"))?;
        let oshape = repr_arg.type_of().shape().to_vec();
        for s in args.iter().skip(1).map(|x| x.type_of().shape()) {
            if s != oshape {
                return Err(Error::msg(format!(
                    "Elementwise operation requires all arguments to have the same shape, got {:?} and {:?}",
                    oshape, s
                )));
            }
        }
        let val = Value::new(Type::tensor(
            repr_arg.type_of().eltype.clone(),
            &oshape,
        ));
        Ok(ElementWiseOp {
            name: name.into(),
            args: args.into(),
            output: val.clone(),
        })
    }

    // arith
    pub fn add(a: &Value, b: &Value) -> Result<Self> {
        if a.type_of() != b.type_of() && !(a.type_of().is_int() || a.type_of().is_float()) {
            return Err(Error::msg(format!(
                "add requires both inputs to have the same element type and either floating or integral, got {:?} and {:?}",
                a.type_of(),
                b.type_of()
            )));
        }
        ElementWiseOp::build("add", &[a.clone(), b.clone()])
    }

    // TODO:
    // arith: sub, mul, div, neg, rem
    // floating: pow, exp, log, sqrt, etc.
    // comparison: eq, ne, lt, gt, le, ge
    // logical: and, or, xor, not
    
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

#[derive(Debug)]
pub struct ElementWiseFn {
    name: String,
    args: Vec<Value>,
    output: Value,
}

#[derive(Clone, Debug)]
pub struct Value(Rc<ValueImpl>);
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
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
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

impl FunctionBuilder {
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

    pub fn while_(&mut self, cond: &Value, scope: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        self.scope.push(vec![]);
        scope(self)?;
        let while_scope = self.scope.pop().unwrap();
        let op = AST::Stmt(Stmt::While(cond.clone(), while_scope), loc);
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
        then: impl Fn(&mut Self) -> Result<()>,
        else_: impl Fn(&mut Self) -> Result<()>,
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

    pub fn elementwise(&mut self, args: &[Value], op_code: &str) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let openum = OpEnum::ElementWise(ElementWiseOp::build(op_code, args)?);
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
                Stmt::While(_, body) => {
                    for ast in body {
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
                    Stmt::While(cond, body) => {
                        if cond == from {
                            *cond = to.clone();
                        }
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

enum Cond {
    AND(Vec<Cond>),
    OR(Vec<Cond>),
    NOT(Box<Cond>),
    Val(Value),
}

fn has_pred_break(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::If(_, then, else_) => {
            then.iter().filter_map(|x| x.get_stmt_ref()).map(|x| has_pred_break(x)).any(|x| x)
            || else_.iter().filter_map(|x| x.get_stmt_ref()).map(|x| has_pred_break(x)).any(|x| x)
        }
        _ => false
    }
}

fn remove_unecessary_loop(stmts: &mut Vec<AST>) {
    let mut i = 0;
    while i < stmts.len() {
        let mut body_stmts = vec![];
        if let Some(stmt) = stmts[i].get_stmt_mut() {
            match stmt {
                Stmt::For(ind_var, start, _, _, body) => {
                    for j in 0..body.len() {
                        if let AST::Stmt(Stmt::Break, _) = body[j] {
                            body.truncate(j);
                            replace_all_uses_with(body, ind_var, start);
                            remove_unecessary_loop(body);
                            std::mem::swap(&mut body_stmts, body);
                            break;
                        }
                    }
                },
                Stmt::While(_, body) => {
                    for j in 0..body.len() {
                        if let AST::Stmt(Stmt::Break, _) = body[j] {
                            body.truncate(j);
                            remove_unecessary_loop(body);
                            std::mem::swap(&mut body_stmts, body);
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
        if body_stmts.len() > 0 {
            let old_len = body_stmts.len();
            stmts.splice(i..i+1, body_stmts);
            i += old_len - 1;
        }
        i += 1;
    }
}

// fn protect_block_break(nodes: &mut Vec<AST>, break_cond_var: &Value) -> bool {
//     let mut i = 0;
//     let mut has_break = false;
//     while i < nodes.len() {
//         if let Some(stmt) = nodes[i].get_stmt_mut() {
//             if let Stmt::If(_, _, _) = stmt {
//                 has_break = remove_pred_break(stmt, break_cond_var);
//             }
//         }
//         if has_break {
//             let mut rest: Vec<_> = nodes.drain(i+1..).collect();
//             protect_block_break(&mut rest, break_cond_var);
//             let if_stmt = Stmt::If(break_cond_var.clone(), rest, vec![]);
//             nodes.push(AST::Stmt(if_stmt, nodes[i].loc().clone()));
//             break;
//         }
//         i += 1;
//     }
//     has_break
// }

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
                if let Some(Stmt::Break) = else_[i].get_stmt_ref() {
                    let not_op = OpEnum::ElementWise(ElementWiseOp::build("not",&[cond.clone()]).unwrap());                    
                    let not_val = not_op.outputs().pop().unwrap().clone();
                    else_.insert(i, AST::Op(not_op, else_[i].loc().clone()));
                    i += 1;
                    else_[i] = AST::Stmt(Stmt::Assign(break_cond_var.clone(), not_val), else_[i].loc().clone());
                    has_break = true;
                    break;
                } else if let Some(Stmt::Continue) = else_[i].get_stmt_ref() {
                    let not_op = OpEnum::ElementWise(ElementWiseOp::build("not",&[cond.clone()]).unwrap());                    
                    let not_val = not_op.outputs().pop().unwrap().clone();
                    else_.insert(i, AST::Op(not_op, else_[i].loc().clone()));
                    i += 1;
                    else_[i] = AST::Stmt(Stmt::Assign(continue_cond_var.clone(), not_val), else_[i].loc().clone());
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
            let not_execute_rest = OpEnum::ElementWise(ElementWiseOp::build("or", &[break_cond_var.clone(), continue_cond_var.clone()]).unwrap());
            let not_execute_continue_val = not_execute_rest.outputs()[0].clone();
            let execute_rest = OpEnum::ElementWise(ElementWiseOp::build("not", &[not_execute_continue_val.clone()]).unwrap());
            let execute_rest_val = execute_rest.outputs()[0].clone();
            nodes.splice(i+1..i+1, [AST::Op(not_execute_rest, loc.clone()), AST::Op(execute_rest, loc.clone())]);
            i += 2;

            let mut rest: Vec<_> = nodes.drain(i+1..).collect();
            let (has_break_, has_continue_) = hoist_break_continue(&mut rest, Some(break_cond_var), Some(continue_cond_var));
            has_break |= has_break_;
            has_continue |= has_continue_;
            let if_stmt = Stmt::If(execute_rest_val, rest, vec![]);
            nodes.push(AST::Stmt(if_stmt, nodes[i].loc().clone()));
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
                            let ind_cond = builder.elementwise(&[new_ind_var.clone(), end.clone()], "lt").unwrap();
                            let cond_var = builder.elementwise(&[ind_cond, break_cond_var.clone()], "and").unwrap();
                            let cond_var = builder.elementwise(&[cond_var, continue_cond_var.clone()], "and").unwrap();
                            builder.while_(&cond_var, |builder| {
                                for node in new_body {
                                    builder.insert_node(node);
                                }
                                let new_ind_var_ = builder.elementwise(&[new_ind_var.clone(), step.clone()], "add").unwrap();
                                builder.assign(&new_ind_var, &new_ind_var_).unwrap();
                                let ind_cond = builder.elementwise(&[new_ind_var.clone(), end.clone()], "lt").unwrap();
                                let cond_var_ = builder.elementwise(&[ind_cond, break_cond_var], "and").unwrap();
                                let cond_var_ = builder.elementwise(&[cond_var_, continue_cond_var], "and").unwrap();
                                builder.assign(&cond_var, &cond_var_).unwrap();
                                Ok(())
                            }).unwrap();
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
