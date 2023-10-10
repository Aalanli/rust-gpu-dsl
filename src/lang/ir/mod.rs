use anyhow::{Error, Result};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Range};
use std::rc::Rc;
use super::Location;

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

    pub fn inner_dtype(&self) -> &Dtype {
        match self {
            ElType::Ptr(dtype) => dtype,
            ElType::Val(dtype) => dtype,
        }
    }

    pub fn i32() -> Self {
        ElType::Val(Dtype::I32)
    }

    pub fn f32() -> Self {
        ElType::Val(Dtype::F32)
    }

    pub fn bool() -> Self {
        ElType::Val(Dtype::I1)
    }

    pub fn ptr_i32() -> Self {
        ElType::Ptr(Dtype::I32)
    }

    pub fn ptr_f32() -> Self {
        ElType::Ptr(Dtype::F32)
    }

    pub fn ptr_bool() -> Self {
        ElType::Ptr(Dtype::I1)
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



#[derive(Clone, Debug)]
pub struct Value(Rc<ValueImpl>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl Value {
    pub fn new(type_of: Type) -> Self {
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


// pub type Op = Rc<Operation>;

#[derive(Debug, Clone)]
pub struct Op(Rc<Operation>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(usize);


#[derive(Debug, Clone)]
pub struct Block(Rc<BlockImpl>);

impl Block {
    pub fn new(args: Vec<Value>, body: Vec<Op>) -> Self {
        Block(Rc::new(BlockImpl { args, body }))
    }

    pub fn args(&self) -> &[Value] {
        &self.0.args
    }

    pub fn body(&self) -> &[Op] {
        &self.0.body
    }
}

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

    pub fn location(&self) -> &Location {
        &self.0.location
    }

    pub fn inputs(&self) -> Vec<&Value> {
        self.0.op.inputs()
    }

    pub fn outputs(&self) -> Vec<&Value> {
        self.0.op.outputs()
    }

    pub fn blocks(&self) -> Vec<&Block> {
        self.0.op.blocks()
    }

    pub fn internal_as_any(&self) -> &dyn std::any::Any {
        self.0.op.internal_as_any()
    }

    pub fn isa<T: 'static>(&self) -> bool {
        self.internal_as_any().is::<T>()
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.internal_as_any().downcast_ref::<T>()
    }

    pub fn name(&self) -> &str {
        self.0.op.name()
    }

    pub fn get_inner(&self) -> &OpEnum {
        &self.0.op
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

struct BuilderCTX {
    ops: RefCell<Vec<Vec<Op>>>,
}

impl BuilderCTX {
    fn new() -> Self {
        BuilderCTX { ops: RefCell::new(vec![]) }
    }

    fn new_block(&self) {
        self.ops.borrow_mut().push(vec![]);
    }

    fn push(&self, op: Op) {
        self.ops.borrow_mut().last_mut().unwrap().push(op);
    }

    fn pop_block(&self) -> Vec<Op> {
        let bk = self.ops.borrow_mut().pop().unwrap();
        bk
    }
}

thread_local! {
    static BUILDER_CTX: BuilderCTX = BuilderCTX::new(); 
}

pub fn ctx_push_block() {
    BUILDER_CTX.with(|ctx| {
        ctx.new_block();
    });
}

pub fn ctx_pop_block() -> Vec<Op> {
    BUILDER_CTX.with(|ctx| {
        ctx.pop_block()
    })
}

pub fn ctx_push(op: &Op) {
    BUILDER_CTX.with(|ctx| {
        ctx.push(op.clone());
    });
}




#[derive(Debug, Clone)]
pub enum OpEnum { // could be a trait, once we have a better idea of the functions
    ProgramID(ProgramIDOp),

    Load(LoadOp),
    Store(StoreOp),

    // Reshape(ReshapeOp),
    // Permute(PermuteOp),
    // Slice(SliceOp),
    Expand(ExpandOp),
    Broadcast(BroadcastOp),

    Reduce(ReduceOp),
    ElementWise(ElementWiseOp),
    Dot(DotOp),

    Full(FullOp),
    Constant(ConstantOp),
    Arange(ArangeOp),

    For(ForOp),
    SCFFOR(SCFForOp),
    // If(IfOp),
    FunctionOp(FunctionOp),
    Assign(AssignOp),
}


impl OpEnum {
    pub fn inputs(&self) -> Vec<&Value> {
        match &self {
            Self::ProgramID(op) => {vec![]},
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
            // Self::Reshape(op) => vec![&op.input],
            // Self::Permute(op) => vec![&op.input],
            // Self::Slice(op) => vec![&op.input],
            Self::Expand(op) => vec![&op.input],
            Self::Broadcast(op) => vec![&op.input],
            Self::Reduce(op) => vec![&op.input],
            Self::ElementWise(op) => op.args.iter().collect(),
            Self::Dot(op) => vec![&op.a, &op.b],
            Self::Full(_op) => vec![],
            Self::Constant(_op) => vec![],
            Self::Arange(_op) => vec![],
            Self::For(op) => vec![&op.start, &op.end, &op.step],
            Self::SCFFOR(op) => vec![&op.start, &op.end, &op.step].into_iter().chain(op.carries.iter()).collect(),
            // Self::If(op) => vec![&op.cond],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(op) => vec![&op.lhs, &op.rhs],
        }
    }

    pub fn outputs(&self) -> Vec<&Value> {
        match &self {
            Self::ProgramID(op) => {vec![&op.output]},
            Self::Load(op) => vec![&op.output],
            Self::Store(_op) => vec![],
            // Self::Reshape(op) => vec![&op.output],
            // Self::Permute(op) => vec![&op.output],
            // Self::Slice(op) => vec![&op.output],
            Self::Expand(op) => vec![&op.output],
            Self::Broadcast(op) => vec![&op.output],
            Self::Reduce(op) => vec![&op.output],
            Self::ElementWise(op) => vec![&op.output],
            Self::Dot(op) => vec![&op.output],
            Self::Full(op) => vec![&op.output],
            Self::Constant(op) => vec![&op.output],
            Self::Arange(op) => vec![&op.output],
            Self::For(_op) => vec![],
            Self::SCFFOR(op) => op.redefines.iter().collect(),
            // Self::If(_op) => vec![],
            Self::FunctionOp(_op) => vec![],
            Self::Assign(_op) => vec![],
        }
    }

    pub fn blocks(&self) -> Vec<&Block> {
        match &self {
            // Self::If(op) => {vec![&op.then, &op.else_]},
            Self::For(op) => {vec![&op.body]},
            Self::SCFFOR(op) => {vec![&op.body]},
            Self::FunctionOp(op) => {vec![&op.body]},
            _ => vec![],
        }
    }

    pub fn internal_as_any(&self) -> &dyn std::any::Any {
        match self {
            Self::ProgramID(x) => x,
            Self::Load(x) => x,
            Self::Store(x) => x,
            // Self::Reshape(x) => x,
            // Self::Permute(x) => x,
            // Self::Slice(x) => x,
            Self::Expand(x) => x,
            Self::Broadcast(x) => x,
            Self::Reduce(x) => x,
            Self::ElementWise(x) => x,
            Self::Dot(x) => x,
            Self::Full(x) => x,
            Self::Constant(x) => x,
            Self::Arange(x) => x,
            Self::For(x) => x,
            Self::SCFFOR(x) => x,
            // Self::If(x) => x,
            Self::FunctionOp(x) => x,
            Self::Assign(x) => x,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::ProgramID(_) => "ProgramID",
            Self::Load(_) => "Load",
            Self::Store(_) => "Store",
            // Self::Reshape(_) => "Reshape",
            // Self::Permute(_) => "Permute",
            // Self::Slice(_) => "Slice",
            Self::Expand(_) => "Expand",
            Self::Broadcast(_) => "Broadcast",
            Self::Reduce(_) => "Reduce",
            Self::ElementWise(_) => "ElementWise",
            Self::Dot(_) => "Dot",
            Self::Full(_) => "Full",
            Self::Constant(_) => "Constant",
            Self::Arange(_) => "Arange",
            Self::For(_) => "For",
            Self::SCFFOR(_) => "SCFFOR",
            // Self::If(_) => "If",
            Self::FunctionOp(_) => "FunctionOp",
            Self::Assign(_) => "Assign",
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
pub struct SCFForOp {
    pub induction_var: Value,
    pub start: Value,
    pub end: Value,
    pub step: Value,
    pub body: Block,
    pub carries: Vec<Value>,
    pub redefines: Vec<Value>,
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
    pub output: Value,
}

impl BroadcastOp {
    pub fn broad_cast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
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

    pub fn build(input: &Value, shape: &[usize]) -> Result<Self> {
        let shape = BroadcastOp::broad_cast_shape(
            input.type_of().shape(),
            shape
        )?;
        let val = Value::new(Type::tensor(input.type_of().eltype.clone(), &shape));
        Ok(BroadcastOp {
            input: input.clone(),
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
        if !vals.iter().all(|x| x.type_of().shape() == a.shape()) {
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
pub struct ConstantOp {
    pub const_value: Constant,
    pub output: Value,
}

impl ConstantOp {
    pub fn build(value: impl Into<Constant>) -> Self {
        let c: Constant = value.into();
        let ctype = Type::scalar(ElType::Val(c.dtype()));
        let val = Value::new(ctype);
        ConstantOp {
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

