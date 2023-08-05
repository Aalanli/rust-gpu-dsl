use std::rc::Rc;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut, Range};

use anyhow::{Result, Error, Context};

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
    shape: Vec<usize>
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

    pub fn tensor(eltype: ElType, shape: &[usize]) -> Self {
        Type {
            eltype,
            shape: shape.into(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
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
            AST::Op(op, loc) => op.verify().context(format!("at {}:{}:{}", loc.file, loc.row, loc.col)),
            AST::Stmt(stmt, loc) => stmt.verify().context(format!("at {}:{}:{}", loc.file, loc.row, loc.col)),
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    For(Value, Value, Value, Value, Vec<AST>), // induction_var, start, end, step, body
    While(Value, Vec<AST>), // condition, body
    If(Value, Vec<AST>, Vec<AST>), // condition, then, else
    Assign(Value, Value), // lhs, rhs
    Return(Vec<Value>),
    Break,
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
    Store(Value, Value, Option<Value>), // ptr, value, mask
    
    Reshape(Value, Vec<i32>, Value), // input, shape, output
    Permute(Value, Vec<u32>, Value), // input, permutation, output
    Slice(Value, Vec<(i32, i32)>, Value), // input, (begin, end), output
    Expand(Value, i32, Value), // input, dims, output
    BroadCast(Value, Value, Value), // input, other, output
    
    Reduce(Value, i32, ReduceOp, Value), // input, dims, op, output
    ElementWise(ElementWiseFn), // for extensibility reasons
    Dot(Value, Value, Value), // a @ b = c

    Full(Constant, Value), // const_value, output
    Arange(i32, i32, Value) // begin, end, output
}

impl Ops {
    pub fn verify(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    And,
    Or,
    Xor
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
    name_hint: RefCell<Option<String>>
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

    pub fn load(&mut self, ptr: &Value, mask: Option<&Value>, value: Option<&Value>) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        let val = Value::new(ptr.type_of().clone());
        let op = AST::Op(Ops::Load(ptr.clone(), mask.cloned(), value.cloned(), val.clone()), loc);
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

    pub fn reshape(&mut self, input: &Value, shape: &[i32]) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Reshape(input.clone(), shape.into(), val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn permute(&mut self, input: &Value, permutation: &[u32]) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Permute(input.clone(), permutation.into(), val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    /// Range: a..b
    /// RangeTo: 0..b
    /// RangeFrom: a..-1
    /// RangeFull: 0..-1
    pub fn slice(&mut self, input: &Value, slices: &[Range<i32>]) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Slice(input.clone(), slices.into(), val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn expand(&mut self, input: &Value, dims: i32) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Expand(input.clone(), dims, val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn broadcast(&mut self, input: &Value, other: &Value) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Broadcast(input.clone(), other.clone(), val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn reduce(&mut self, input: &Value, dims: i32, op: ReduceOp) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Reduce(input.clone(), dims, op, val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn dot(&mut self, a: &Value, b: &Value) -> Result<Value> {
        // let loc = std::panic::Location::caller().into();
        // let val = Value::new(Type::tensor(input.type_of().eltype.clone(), shape));
        // let op = AST::Op(Ops::Dot(a.clone(), b.clone(), val.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // Ok(val)
        todo!()
    }

    pub fn extern_elementwise(&mut self, name: impl Into<String>, args: &[Value]) -> Result<Value> {
        let loc = std::panic::Location::caller().into();
        self.elementwise_builder(loc, name, args)
    }

    fn elementwise_builder(&mut self, loc: Location, name: impl Into<String>, args: &[Value]) -> Result<Value> {
        let repr_arg = args.first().ok_or(Error::msg("args cannot be empty"))?;
        let val = Value::new(Type::tensor(
            repr_arg.type_of().eltype.clone(), repr_arg.type_of().shape()));
        let op = AST::Op(Ops::ElementWise(ElementWiseFn {
            name: name.into(),
            args: args.into(),
            output: val.clone(),
        }), loc);
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
        let val = Value::new(Type::tensor(ElType::Val(Dtype::I32), &[end as usize - begin as usize]));
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

    pub fn for_(&mut self, begin: &Value, end: &Value, step: &Value, scope: impl Fn(&mut Self) -> Result<()>) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let induction_var = Value::new(Type::scalar(ElType::Val(Dtype::I32)));
        // let op = AST::Stmt(Stmt::For(induction_var.clone(), begin.clone(), end.clone(), step.clone(), vec![]), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // self.scope.push(vec![]);
        // scope(self)?;
        // self.scope.pop();
        Ok(())
    }

    pub fn while_(&mut self, cond: &Value, scope: impl Fn(&mut Self) -> Result<()>) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let op = AST::Stmt(Stmt::While(cond.clone(), vec![]), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // self.scope.push(vec![]);
        // scope(self)?;
        // self.scope.pop();
        Ok(())
    }

    pub fn return_(&mut self, values: &[Value]) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let op = AST::Stmt(Stmt::Return(values.into()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn if_(&mut self, cond: &Value, then: impl Fn(&mut Self) -> Result<()>, else_: impl Fn(&mut Self) -> Result<()>) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let op = AST::Stmt(Stmt::If(cond.clone(), vec![], vec![]), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        // self.scope.push(vec![]);
        // then(self)?;
        // self.scope.pop();
        // self.scope.push(vec![]);
        // else_(self)?;
        // self.scope.pop();
        Ok(())
    }

    pub fn break_(&mut self) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let op = AST::Stmt(Stmt::Break, loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn assign(&mut self, lhs: &Value, rhs: &Value) -> Result<()> {
        // let loc = std::panic::Location::caller().into();
        // let op = AST::Stmt(Stmt::Assign(lhs.clone(), rhs.clone()), loc);
        // op.verify()?;
        // self.scope.last_mut().unwrap().push(op);
        Ok(())
    }

    pub fn build(self) -> Result<Function> {
        todo!()
    }
}


#[test]
fn test_softmax() -> Result<()> {
    let mut builder = FunctionBuilder::new("softmax_kernel");
    let [x_ptr, y_ptr, row_shape] = builder.arg([
        Type::f32_ptr(), Type::f32_ptr(), Type::i32_scalar()
    ]);

    let tid = builder.program_id()?;
    let idx = builder.arange(0, 512)?;
    let mask = builder.lt(&idx, &row_shape)?;
    let offset = builder.mul(&tid, &row_shape)?;
    let idx = builder.add(&idx, &offset)?;

    let load_ptr = builder.add(&x_ptr, &idx)?;

    let x = builder.load(&load_ptr, Some(&mask), None)?;
    let x = builder.exp(&x)?;
    let sum = builder.reduce(&x, 0, ReduceOp::Sum)?;
    let x = builder.div(&x, &sum)?;

    let write_ptr = builder.add(&y_ptr, &idx)?;
    builder.store(&write_ptr, &x, Some(&mask))?;

    let softmax_kernel = builder.build()?;
    println!("{:#?}", softmax_kernel);
    Ok(())
}