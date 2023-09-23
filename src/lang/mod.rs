use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Range};
use std::rc::Rc;

use anyhow::{Error, Result};
use crate::AsAny;

mod ir;
use ir::OpEnum;

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




fn test(a: Result<()>) {
    match a {
        Ok(_) => {}
        Err(e) => {
            let t = e.as_any();
            println!("{:#?}", t.type_id());
            println!("{}", e);
        }
    }
}

#[test]
fn test1() {
    let a = Err(Error::msg("test"));
    test(a);
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
