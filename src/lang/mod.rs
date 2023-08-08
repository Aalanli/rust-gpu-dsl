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
    Op(Ops, Location),
    Stmt(Stmt, Location),
}

impl AST {
    pub fn verify(&self) -> Result<()> {
        match self {
            AST::Op(op, loc) => op
                .verify()
                .context(format!("at {}:{}:{}", loc.file, loc.row, loc.col)),
            AST::Stmt(stmt, loc) => stmt
                .verify()
                .context(format!("at {}:{}:{}", loc.file, loc.row, loc.col)),
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
    pub fn verify(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub enum Ops {
    ProgramID(Value), // output

    Load(Value, Option<Value>, Option<Value>, Value), // ptr, mask, value, output
    Store(Value, Value, Option<Value>),               // ptr, value, mask

    Reshape(Value, Vec<usize>, Value),      // input, shape, output
    Permute(Value, Vec<usize>, Value),      // input, permutation, output
    Slice(Value, Vec<Range<usize>>, Value), // input, (begin, end), output
    Expand(Value, usize, Value),              // input, dim, output
    Broadcast(Value, Value, Value),         // input, other, output

    Reduce(Value, usize, ReduceOpOption, Value), // input, dims, op, output
    ElementWise(ElementWiseFn),          // for extensibility reasons
    Dot(Value, Value, Value),            // a @ b = c

    Full(Constant, Value),   // const_value, output
    Arange(i32, i32, Value), // begin, end, output
}

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

pub struct BroadcastOp {
    input: Value,
    other: Value,
    output: Value,
}

impl BroadcastOp {
    fn broad_cast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
        if b.len() > a.len() {
            return FunctionBuilder::broad_cast_shape(b, a);
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

impl Ops {
    pub fn verify(&self) -> Result<()> {
        Ok(())
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

    pub fn program_id(&mut self) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let val = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        let op = AST::Op(Ops::ProgramID(val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn load(
        &mut self,
        ptr: &Value,
        mask: Option<&Value>,
        value: Option<&Value>,
    ) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let val = Value::new(ptr.type_of().clone());
        let op = AST::Op(
            Ops::Load(ptr.clone(), mask.cloned(), value.cloned(), val.clone()),
            loc,
        );
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn store(&mut self, ptr: &Value, value: &Value, mask: Option<&Value>) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Op(Ops::Store(ptr.clone(), value.clone(), mask.cloned()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

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

    pub fn reshape(&mut self, input: &Value, shape: &[i32]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();

        let oshape = FunctionBuilder::get_reshape_output_shape(input.type_of().shape(), shape)?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        let op = AST::Op(Ops::Reshape(input.clone(), oshape, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

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

    pub fn permute(&mut self, input: &Value, permutation: &[u32]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let oshape =
            FunctionBuilder::get_permute_output_shape(input.type_of().shape(), permutation)?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        let op = AST::Op(
            Ops::Permute(
                input.clone(),
                permutation.iter().map(|x| *x as usize).collect(),
                val.clone(),
            ),
            loc,
        );
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

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
    pub fn slice(&mut self, input: &Value, slices: &[Range<i32>]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let (oshape, oslice) =
            FunctionBuilder::get_slice_output_shape(input.type_of().shape(), slices)?;

        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        let op = AST::Op(Ops::Slice(input.clone(), oslice, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn expand(&mut self, input: &Value, dims: i32) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let res_dim = if dims < 0 {
            input.type_of().rank() as i32 + dims + 1
        } else {
            dims
        };
        if res_dim < 0 || res_dim > input.type_of().rank() as i32 {
            return Err(Error::msg(format!(
                "Invalid expand dimension, got {}",
                dims
            )));
        }
        let mut oshape = input.type_of().shape().to_vec();
        oshape.insert(res_dim as usize, 1);
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &oshape));
        let op = AST::Op(Ops::Expand(input.clone(), res_dim as usize, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    fn broad_cast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
        if b.len() > a.len() {
            return FunctionBuilder::broad_cast_shape(b, a);
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

    pub fn broadcast(&mut self, input: &Value, other: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let shape = FunctionBuilder::broad_cast_shape(
            input.type_of().shape(),
            other.type_of().shape(),
        )?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &shape));
        let op = AST::Op(Ops::Broadcast(input.clone(), other.clone(), val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn reduce(&mut self, input: &Value, dim: i32, op: ReduceOpOption) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
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
            input
                .type_of()
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != reduce_dim as usize)
                .map(|(_, x)| *x)
                .collect::<Vec<_>>()
                .as_slice(),
        ));
        let op = AST::Op(Ops::Reduce(input.clone(), reduce_dim as usize, op, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn dot(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let ashape = a.type_of().shape();
        let bshape = b.type_of().shape();
        if ashape.len() != 2 || bshape.len() != 2 || ashape[1] != bshape[0] {
            return Err(Error::msg(format!(
                "Invalid dot input shapes {:?} and {:?}",
                ashape, bshape
            )));
        }
        let oshape = vec![ashape[0], bshape[1]];
        let val = Value::new(Type::tensor(a.type_of().eltype.clone(), &oshape));
        let op = AST::Op(Ops::Dot(a.clone(), b.clone(), val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn extern_elementwise(&mut self, name: impl Into<String>, args: &[Value]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, name, args)
    }

    fn elementwise_builder(
        &mut self,
        loc: Location,
        name: impl Into<String>,
        args: &[Value],
    ) -> Result<Value> {
        let repr_arg = args.first().ok_or(Error::msg("args cannot be empty"))?;
        let mut oshape = repr_arg.type_of().shape().to_vec();
        for s in args.iter().skip(1).map(|x| x.type_of().shape()) {
            oshape = FunctionBuilder::broad_cast_shape(&oshape, s)?;
        }

        let val = Value::new(Type::tensor(
            repr_arg.type_of().eltype.clone(),
            &oshape,
        ));
        let op = AST::Op(
            Ops::ElementWise(ElementWiseFn {
                name: name.into(),
                args: args.into(),
                output: val.clone(),
            }),
            loc,
        );
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    // arith
    pub fn add(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "add", &[a.clone(), b.clone()])
    }

    pub fn sub(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "sub", &[a.clone(), b.clone()])
    }

    pub fn mul(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "mul", &[a.clone(), b.clone()])
    }

    pub fn div(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "div", &[a.clone(), b.clone()])
    }

    pub fn neg(&mut self, a: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "neg", &[a.clone()])
    }

    // integer
    pub fn rem(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "rem", &[a.clone(), b.clone()])
    }

    // floating
    pub fn pow(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "pow", &[a.clone(), b.clone()])
    }

    pub fn exp(&mut self, a: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "exp", &[a.clone()])
    }

    // logical
    pub fn eq(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "eq", &[a.clone(), b.clone()])
    }

    pub fn leq(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "leq", &[a.clone(), b.clone()])
    }

    pub fn lt(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "lt", &[a.clone(), b.clone()])
    }

    pub fn shr(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "shr", &[a.clone(), b.clone()])
    }

    pub fn shl(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "shl", &[a.clone(), b.clone()])
    }

    pub fn and(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "and", &[a.clone(), b.clone()])
    }

    pub fn or(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "or", &[a.clone(), b.clone()])
    }

    pub fn xor(&mut self, a: &Value, b: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "xor", &[a.clone(), b.clone()])
    }

    pub fn not(&mut self, a: &Value) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, "not", &[a.clone()])
    }

    pub fn arange(&mut self, begin: i32, end: i32) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let val = Value::new(Type::tensor(
            ElType::Val(Dtype::I32),
            &[end as usize - begin as usize],
        ));
        let op = AST::Op(Ops::Arange(begin, end, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(val)
    }

    pub fn full(&mut self, value: impl Into<Constant>, shape: &[usize]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let c: Constant = value.into();
        let ctype = Type::tensor(ElType::Val(c.dtype()), shape);
        let val = Value::new(ctype);
        let op = AST::Op(Ops::Full(c, val.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
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
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn while_(&mut self, cond: &Value, scope: impl Fn(&mut Self) -> Result<()>) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        self.scope.push(vec![]);
        scope(self)?;
        let while_scope = self.scope.pop().unwrap();
        let op = AST::Stmt(Stmt::While(cond.clone(), while_scope), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn return_(&mut self, values: &[Value]) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Return(values.into()), loc);
        op.verify()?;
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
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn break_(&mut self) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Break, loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn assign(&mut self, lhs: &Value, rhs: &Value) -> Result<()> {
        let loc = std::panic::Location::caller().into();
        let op = AST::Stmt(Stmt::Assign(lhs.clone(), rhs.clone()), loc);
        op.verify()?;
        self.scope.last_mut().unwrap().push(op);
        Ok(())
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
}



#[test]
fn test_softmax() -> Result<()> {
    let mut builder = FunctionBuilder::new("softmax_kernel");
    let [x_ptr, y_ptr, row_shape] =
        builder.arg([Type::f32_ptr(), Type::f32_ptr(), Type::i32_scalar()]);

    let tid = builder.program_id()?;
    let idx = builder.arange(0, 512)?;
    let mask = builder.lt(&idx, &row_shape)?;
    let offset = builder.mul(&tid, &row_shape)?;
    let idx = builder.add(&idx, &offset)?;

    let load_ptr = builder.add(&x_ptr, &idx)?;

    let x = builder.load(&load_ptr, Some(&mask), None)?;
    let x = builder.exp(&x)?;
    let sum = builder.reduce(&x, 0, ReduceOpOption::Sum)?;
    let x = builder.div(&x, &sum)?;

    let write_ptr = builder.add(&y_ptr, &idx)?;
    builder.store(&write_ptr, &x, Some(&mask))?;

    let softmax_kernel = builder.build()?;
    println!("{:#?}", softmax_kernel);
    Ok(())
}
