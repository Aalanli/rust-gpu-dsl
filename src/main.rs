use anyhow::{Error, Result};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(PartialEq, Eq, Clone)]
struct Location {
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

pub struct ModuleBuilder {
    name: Option<String>,
    body: Vec<Op>,
    block_stack: Vec<Block>,
}

impl ModuleBuilder {
    thread_local! {
        static MODULE_BUILDER: RefCell<Option<ModuleBuilder>> = RefCell::new(None);
    }

    thread_local! {
        static INTRINSIC_FUNC: RefCell<HashMap<&'static str, FuncOp>> = RefCell::new(HashMap::new());
    }

    pub fn new() -> Self {
        ModuleBuilder {
            name: None,
            body: vec![],
            block_stack: vec![],
        }
    }

    pub fn set_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub fn main_func(mut self, scope: impl Fn()) -> Result<ModuleOp> {
        self.block_stack.push(Block::new());
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_some() {
                return Err(Error::msg("ModuleBuilder already exists"));
            }
            s.borrow_mut().replace(self);
            Ok(())
        })?;
        scope();
        let mut builder = Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            Ok(s.borrow_mut().take().unwrap())
        })?;
        let block = builder.block_stack.pop().ok_or(Error::msg("No block"))?;
        let func_ptr_val = Value::new(None, Type(Rc::new(TypeInternal::Fn(vec![], vec![]))));
        let func_op = FuncOp {
            name: "main".to_string(),
            args: vec![],
            body: block,
            val: func_ptr_val,
            returns: vec![],
            kind: FuncKind::Def,
        };
        let op = Op(Rc::new(OpInternal::FuncOp(func_op)));
        builder.body.push(op);

        Ok(ModuleOp {
            name: builder.name,
            body: Block { ops: builder.body },
        })
    }

    pub fn build_fn(
        name: &str,
        args: Vec<(Option<String>, Type)>,
        returns: Vec<(Option<String>, Type)>,
        scope: impl Fn(&Vec<Value>),
    ) -> Result<Value> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            for opi in s.borrow().as_ref().unwrap().body.iter() {
                if let OpInternal::FuncOp(func_op) = &*opi.0 {
                    if func_op.name == name {
                        return Err(Error::msg("Function already exists"));
                    }
                }
            }
            s.borrow_mut()
                .as_mut()
                .unwrap()
                .block_stack
                .push(Block::new());
            Ok(())
        })?;
        let arguments = args
            .iter()
            .map(|(name, _type)| Value::new(name.clone(), _type.clone()))
            .collect();
        let returns: Vec<Value> = returns
            .iter()
            .map(|(name, _type)| Value::new(name.clone(), _type.clone()))
            .collect();
        scope(&arguments);
        let block_body = Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            Ok(s.borrow_mut()
                .as_mut()
                .unwrap()
                .block_stack
                .pop()
                .ok_or(Error::msg("No block"))?)
        })?;
        let func_ptr_val = Value::new(
            None,
            Type(Rc::new(TypeInternal::Fn(
                arguments.iter().map(|v| v._type().clone()).collect(),
                returns.iter().map(|v| v._type().clone()).collect(),
            ))),
        );
        let func_op = FuncOp {
            name: name.to_string(),
            args: arguments,
            body: block_body,
            val: func_ptr_val.clone(),
            returns,
            kind: FuncKind::Def,
        };
        let op = Op(Rc::new(OpInternal::FuncOp(func_op)));
        Self::MODULE_BUILDER.with(|s| {
            s.borrow_mut().as_mut().unwrap().body.push(op);
        });
        Ok(func_ptr_val)
    }

    pub fn build_intrinsic_fn(
        name: &'static str,
        args: Vec<(Option<String>, Type)>,
        returns: Vec<(Option<String>, Type)>,
    ) -> Result<Value> {
        Self::INTRINSIC_FUNC.with(|s| {
            if s.borrow().contains_key(name) {
                return Err(Error::msg("Function already exists"));
            }
            let arguments: Vec<Value> = args
                .iter()
                .map(|(name, _type)| Value::new(name.clone(), _type.clone()))
                .collect();
            let returns: Vec<Value> = returns
                .iter()
                .map(|(name, _type)| Value::new(name.clone(), _type.clone()))
                .collect();
            let func_ptr_val = Value::new(
                None,
                Type(Rc::new(TypeInternal::Fn(
                    arguments.iter().map(|v| v._type().clone()).collect(),
                    returns.iter().map(|v| v._type().clone()).collect(),
                ))),
            );
            let func_op = FuncOp {
                name: name.to_string(),
                args: arguments,
                body: Block::new(),
                val: func_ptr_val.clone(),
                returns,
                kind: FuncKind::Intrinsic,
            };
            s.borrow_mut().insert(name, func_op);
            Ok(func_ptr_val)
        })
    }

    pub fn get_intrinsic_fn(name: &'static str) -> Option<Value> {
        Self::INTRINSIC_FUNC.with(|s| Some(s.borrow().get(name)?.val.clone()))
    }

    pub fn get_fn(name: &str) -> Option<Value> {
        Self::MODULE_BUILDER.with(|s| {
            for opi in s.borrow().as_ref().unwrap().body.iter() {
                if let OpInternal::FuncOp(func_op) = &*opi.0 {
                    if func_op.name == name {
                        return Some(func_op.val.clone());
                    }
                }
            }
            None
        })
    }

    pub fn insert_op(op: Op) -> Result<()> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            s.borrow_mut()
                .as_mut()
                .unwrap()
                .block_stack
                .last_mut()
                .ok_or(Error::msg("No block"))?
                .ops
                .push(op);
            Ok(())
        })
    }
}

#[derive(Debug, Clone)]
pub struct Value(Rc<ValueInternal>);
impl Value {
    fn is(&self, other: &Value) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    fn new(name: Option<String>, _type: Type) -> Self {
        Value(Rc::new(ValueInternal { name, _type }))
    }
}
#[derive(Debug)]
struct ValueInternal {
    name: Option<String>,
    _type: Type,
}

impl Value {
    fn _type(&self) -> &Type {
        &self.0._type
    }
}

#[derive(Debug)]
pub struct Op(Rc<OpInternal>);
#[derive(Debug)]
pub enum OpInternal {
    Constant(ConstantOp),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    If(IfOp),
    While(WhileOp),
    For(ForOp),
    Assign(AssignOp),
    FuncOp(FuncOp),
    Call(CallOp),
    Cast(CastOp),
    Module(ModuleOp),
}

#[derive(Debug)]
pub struct ModuleOp {
    name: Option<String>,
    body: Block,
}

#[derive(Debug)]
pub struct ConstantOp {
    value: Constant,
    result: Value,
}

#[derive(Debug)]
pub struct IfOp {
    cond: Value,
    then_block: Block,
    else_block: Block,
}

#[derive(Debug)]
pub struct WhileOp {
    cond: Value,
    body: Block,
}
#[derive(Debug)]
pub struct ForOp {
    init: Value,
    fin: Value,
    step: Value,
    body: Block,
}

#[derive(Debug)]
pub struct AssignOp {
    lhs: Value,
    rhs: Value,
}

#[derive(Debug)]
pub struct FuncOp {
    name: String,
    args: Vec<Value>,
    body: Block,
    returns: Vec<Value>,
    val: Value,
    kind: FuncKind,
}

#[derive(Debug)]
enum FuncKind {
    Def,
    Intrinsic,
}

#[derive(Debug)]
pub struct YieldOp {
    value: Vec<Value>,
}

#[derive(Debug)]
pub struct CallOp {
    func: Value,
    args: Vec<Value>,
    returns: Vec<Value>,
}

#[derive(Debug)]
pub struct CastOp {
    value: Value,
    result: Value,
    _type: Type,
}

#[derive(Debug)]
struct Block {
    ops: Vec<Op>,
}

impl Block {
    fn new() -> Self {
        Block { ops: vec![] }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Type(Rc<TypeInternal>);

impl Type {
    fn is_integer(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Int) => true,
            _ => false,
        }
    }

    fn is_float(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Float) => true,
            _ => false,
        }
    }

    fn is_bool(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Bool) => true,
            _ => false,
        }
    }
}
#[derive(Debug, Eq, PartialEq)]
enum TypeInternal {
    Val(Dtype),
    Ptr(Dtype),
    Fn(Vec<Type>, Vec<Type>),
}
#[derive(Debug)]
enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Constant {
    fn get_type(&self) -> Type {
        match self {
            Constant::Int(_) => Type(Rc::new(TypeInternal::Val(Dtype::Int))),
            Constant::Float(_) => Type(Rc::new(TypeInternal::Val(Dtype::Float))),
            Constant::Bool(_) => Type(Rc::new(TypeInternal::Val(Dtype::Bool))),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Dtype {
    Int,
    Float,
    Bool,
}

impl Dtype {
    fn bit_width(&self) -> u64 {
        match self {
            Dtype::Int => 64,
            Dtype::Float => 64,
            Dtype::Bool => 8,
        }
    }
}

impl From<i64> for Constant {
    fn from(value: i64) -> Self {
        Constant::Int(value)
    }
}

impl From<f64> for Constant {
    fn from(value: f64) -> Self {
        Constant::Float(value)
    }
}

impl From<bool> for Constant {
    fn from(value: bool) -> Self {
        Constant::Bool(value)
    }
}

#[derive(Debug)]
pub struct BinaryOp {
    lhs: Value,
    rhs: Value,
    result: Value,
    op: BinaryOps,
}

#[derive(Debug, Clone)]
enum BinaryOps {
    // (lhs, rhs, result)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Eq,
    Lt,
    Leq,
    Shr,
    Shl,
}

#[derive(Debug)]
pub struct UnaryOp {
    val: Value,
    result: Value,
    op: UnaryOps,
}

#[derive(Debug, Clone)]
enum UnaryOps {
    // (val, result)
    Neg,
    Not,
}

fn constant<T: Into<Constant>>(val: T, name: Option<&str>) -> Result<Value> {
    let constant: Constant = val.into();
    let result = Value::new(name.map(|s| s.to_string()), constant.get_type());

    let constant_op = ConstantOp {
        value: constant,
        result: result.clone(),
    };
    ModuleBuilder::insert_op(Op(Rc::new(OpInternal::Constant(constant_op))))?;
    Ok(result)
}

// fn assign(lhs: &Value, rhs: &Value) -> Result<()> {
//     if lhs.0._type != rhs.0._type {
//         return Err(Error::msg("Type mismatch"));
//     }
//     let assign_op = AssignOp {
//         lhs: lhs.clone(),
//         rhs: rhs.clone()
//     };
//     insert_op(Op(Rc::new(OpInternal::Assign(assign_op))))?;
//     Ok(())
// }

// fn binary_op(lhs: &Value, rhs: &Value, op: BinaryOps) -> Result<Value> {
//     let out_type = match op {
//         BinaryOps::Add | BinaryOps::Sub | BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow => {
//             if lhs.0._type != rhs.0._type {
//                 return Err(Error::msg("Type mismatch"));
//             }
//             if !(lhs.0._type.is_integer() || lhs.0._type.is_float()) {
//                 return Err(Error::msg("Must be integer or float"));
//             }
//             lhs.0._type.clone()
//         },
//         BinaryOps::And | BinaryOps::Or | BinaryOps::Lt | BinaryOps::Leq | BinaryOps::Eq => {
//             if lhs.0._type != rhs.0._type {
//                 return Err(Error::msg("Type mismatch"));
//             }
//             if !lhs.0._type.is_bool() {
//                 return Err(Error::msg("Must be bool"));
//             }
//             Type(Rc::new(TypeInternal::Val(Dtype::Bool)))
//         },
//         BinaryOps::Mod | BinaryOps::Shr | BinaryOps::Shl => {
//             if lhs.0._type != rhs.0._type {
//                 return Err(Error::msg("Type mismatch"));
//             }
//             if !lhs.0._type.is_integer() {
//                 return Err(Error::msg("Must be integer"));
//             }
//             lhs.0._type.clone()
//         },
//     };

//     let result = Value::new(None, out_type);
//     let binary_op = BinaryOp {
//         lhs: lhs.clone(),
//         rhs: rhs.clone(),
//         result: result.clone(),
//         op
//     };
//     insert_op(Op(Rc::new(OpInternal::BinaryOp(binary_op))))?;
//     Ok(result)
// }

// fn unary_op(val: &Value, op: UnaryOps) -> Result<Value> {
//     let out_type = match op {
//         UnaryOps::Neg => {
//             if !val.0._type.is_integer() && !val.0._type.is_float() {
//                 return Err(Error::msg("Must be integer or float"));
//             }
//             val.0._type.clone()
//         },
//         UnaryOps::Not => {
//             if !val.0._type.is_bool() {
//                 return Err(Error::msg("Must be bool"));
//             }
//             Type(Rc::new(TypeInternal::Val(Dtype::Bool)))
//         },
//     };

//     let result = Value::new(None, out_type);
//     let unary_op = UnaryOp {
//         val: val.clone(),
//         result: result.clone(),
//         op
//     };
//     insert_op(Op(Rc::new(OpInternal::UnaryOp(unary_op))))?;
//     Ok(result)
// }

// fn for_op(init: &Value, fin: &Value, step: &Value, body: impl Fn(&Value) -> Result<()>) -> Result<()> {
//     let induction_var = Value::new(Some("i".to_string()), Type(Rc::new(TypeInternal::Val(Dtype::Int))));

//     let body_block = with_scope(|| {
//         body(&induction_var)
//     })?;

//     let for_op = ForOp {
//         init: init.clone(),
//         fin: fin.clone(),
//         step: step.clone(),
//         body: body_block
//     };

//     let op = Op(Rc::new(OpInternal::For(for_op)));
//     insert_op(op)
// }

// fn if_op(cond: &Value, then_block: impl Fn() -> Result<()>, else_block: Option<impl Fn() -> Result<()>>) -> Result<()> {
//     let then_block = with_scope(|| {
//         then_block()
//     })?;

//     let else_block = with_scope(|| {
//         if let Some(else_block) = &else_block {
//             else_block()?;
//         }
//         Ok(())
//     })?;

//     let if_op = IfOp {
//         cond: cond.clone(),
//         then_block,
//         else_block
//     };
//     let op = Op(Rc::new(OpInternal::If(if_op)));
//     insert_op(op)
// }

// fn while_op(cond: &Value, body: impl Fn() -> Result<()>) -> Result<()> {
//     let body = with_scope(|| {
//         body()
//     })?;

//     let while_op = WhileOp {
//         cond: cond.clone(),
//         body: body
//     };
//     insert_op(Op(Rc::new(OpInternal::While(while_op))))
// }

/*

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::cell::{RefCell, Ref};
use std::thread::Scope;

use anyhow::{Error, Result};

struct TraitRegistry {
    trait_converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn Any>>,
}

impl TraitRegistry {
    fn get_trait<'a, T: 'static, F: 'static + ?Sized>(&self, a: &'a T) -> Option<&'a F> {
        use std::any::TypeId;
        let f = self
            .trait_converters
            .get(&(TypeId::of::<T>(), TypeId::of::<F>()))?
            .downcast_ref::<fn(&T) -> &F>()?;
        println!("{:?}", f.type_id());
        Some(f(a))
    }

    fn register_trait<T: 'static, F: 'static + ?Sized>(&mut self, f: fn(&T) -> &F) {
        use std::any::TypeId;
        println!("{:?}", TypeId::of::<fn(&T) -> &F>());

        self.trait_converters
            .insert((TypeId::of::<T>(), TypeId::of::<F>()), Box::new(f));
    }
}


#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    DType(Dtype),
    FnPtr(Vec<Type>, Vec<Type>),
    Ptr(Dtype)
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Dtype {
    Float,
    Int,
    Bool
}

#[derive(Debug, PartialEq)]
pub enum ConstValue {
    Float(f64),
    Int(i64),
    Bool(bool)
}

#[derive(Debug, Clone)]
pub struct Value(Arc<ValueInternal>);

impl Value {
    pub fn new(_type: Type, name: Option<String>) -> Self {
        Value(Arc::new(
            ValueInternal { name: name.map(|x| x.to_string()), _type }
        ))
    }

    pub fn _type(&self) -> &Type {
        &self.0._type
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Debug)]
struct ValueInternal {
    name: Option<String>,
    _type: Type,
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct Operand(Value);

impl Operand {
    fn new(val: Value) -> Self {
        Operand(val)
    }

    fn new_from(val: &Value) -> Self {
        Operand(Value::new(val._type().clone(), val.0.name.clone()))
    }
}


#[derive(Debug)]
pub struct Block {
    body: Vec<Op>
}

impl Block {
    pub fn new() -> Self {
        Block { body: vec![] }
    }

    pub fn insert_op(&mut self, i: usize, op: Op) {
        self.body.insert(i, op);
    }
}

#[derive(Debug)]
pub struct OperationBase {
    source: HashMap<Operand, Value>, // the source value
    users: HashMap<Value, Vec<Operand>>, // the users of the source value
    val_parent: HashMap<Value, (usize, usize)>, // the source op
    oper_parent: HashMap<Operand, (usize, usize)>, // the op that the operand belongs to
    operands: Vec<Operand>,
    values: Vec<Value>,
    blocks: Vec<Block>,
    loc: Option<Location>,
}

impl OperationBase {
    pub fn new(loc: Location) -> Self {
        OperationBase {
            source: HashMap::new(),
            users: HashMap::new(),
            val_parent: HashMap::new(),
            oper_parent: HashMap::new(),
            operands: vec![],
            values: vec![],
            blocks: vec![],
            loc: Some(loc),
        }
    }
}

impl Default for OperationBase {
    fn default() -> Self {
        OperationBase {
            source: HashMap::new(),
            users: HashMap::new(),
            val_parent: HashMap::new(),
            oper_parent: HashMap::new(),
            operands: vec![],
            values: vec![],
            blocks: vec![],
            loc: None,
        }
    }
}

#[derive(Debug)]
pub enum Op {
    Constant(ConstantOp),
    If(IfOp),
    For(ForOp),
    While(WhileOp),
    DeclareFn(DeclareFnOp),
    Call(CallOp),
    Yield(YieldOp),
    Module(ModuleOp),
}

impl Op {
    fn get_op_base(&self) -> &OperationBase {
        match self {
            Op::Constant(op) => &op.base,
            Op::If(op) => &op.base,
            Op::For(op) => &op.base,
            Op::While(op) => &op.base,
            Op::DeclareFn(op) => &op.base,
            Op::Call(op) => &op.base,
            Op::Yield(op) => &op.base,
            Op::Module(op) => &op.base,
        }
    }

    fn operands(&self) -> &[Operand] {
        &self.get_op_base().operands
    }

    fn values(&self) -> &[Value] {
        &self.get_op_base().values
    }

    fn blocks(&self) -> &[Block] {
        &self.get_op_base().blocks
    }
}

#[derive(Debug)]
pub struct ConstantOp {
    value: ConstValue,
    base: OperationBase
}

#[derive(Debug)]
pub struct IfOp {
    cond_operand: Operand,
    cond_val: Value,
    base: OperationBase
}

#[derive(Debug)]
pub struct ForOp {
    init_operand: Operand,
    end_operand: Operand,
    step_operand: Operand,
    ind_var: Value,
    base: OperationBase
}

#[derive(Debug)]
pub struct WhileOp {
    init_cond: Operand,
    cond_val: Value,
    body: Block,
    base: OperationBase
}

#[derive(Debug)]
pub struct DeclareFnOp {
    name: String,
    fn_type: FnType,
    fn_kind: FnKind,
    base: OperationBase
}

#[derive(Debug)]
pub enum FnType {
    Intrinsic,
    Defined
}

#[derive(Debug)]
pub enum FnKind {
    Device,
    Host
}

#[derive(Debug)]
pub struct CallOp {
    func: String,
    base: OperationBase
}

#[derive(Debug)]
pub struct YieldOp {
    base: OperationBase
}


#[derive(Debug)]
pub struct ModuleOp {
    name: Option<String>,
    base: OperationBase
}



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


pub struct ModuleBuilder {
    name: Option<String>,
    body: Vec<Op>,
    op_stack: Vec<ScopeStack>,
    remap: HashMap<Value, Value>
}

struct ScopeStack {
    stack: Vec<Op>,
    source: HashMap<Operand, Value>,
}

impl ScopeStack {
    fn connect(&mut self, operand: &Operand, value: &Value) {
        self.source.insert(operand.clone(), value.clone());
    }
}

impl ModuleBuilder {
    thread_local! {
        static MODULE_BUILDER: RefCell<Option<ModuleBuilder>> = RefCell::new(None);
    }

    thread_local! {
        static INTRINSIC_FUNC: RefCell<HashMap<&'static str, Arc<DeclareFnOp>>> = RefCell::new(HashMap::new());
    }

    pub fn new() -> Self {
        ModuleBuilder {
            name: None,
            body: vec![],
            op_stack: vec![],
            remap: HashMap::new(),
        }
    }

    pub fn set_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    /*
    pub fn main_func(mut self, scope: impl Fn()) -> Result<ModuleOp> {
        self.block_stack.push(Block::new());
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_some() {
                return Err(Error::msg("ModuleBuilder already exists"));
            }
            s.borrow_mut().replace(self);
            Ok(())
        })?;
        scope();
        let mut builder = Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            Ok(s.borrow_mut().take().unwrap())
        })?;
        let block = builder.block_stack.pop().ok_or(Error::msg("No block"))?;
        let func_ptr_val = Value::new(None, Type(Rc::new(TypeInternal::Fn(vec![], vec![]))));
        let func_op = FuncOp {
            name: "main".to_string(),
            args: vec![],
            body: block,
            val: func_ptr_val,
            returns: vec![],
            kind: FuncKind::Def,
        };
        let op = Op(Rc::new(OpInternal::FuncOp(func_op)));
        builder.body.push(op);

        Ok(ModuleOp { name: builder.name, body: Block { ops: builder.body } })
    }*/
    /*
    pub fn declare_fn(name: &str, args: Vec<(Option<String>, Type)>, returns: Vec<(Option<String>, Type)>, scope: impl Fn(&Vec<Value>)) -> Result<()> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            for opi in s.borrow().as_ref().unwrap().body.iter() {
                if let OpEnum::DeclareFn(func_op) = &*opi.0 {
                    if func_op.name == name {
                        return Err(Error::msg("Function already exists"));
                    }
                }
            }
            s.borrow_mut().as_mut().unwrap().block_stack.push(Block::new());
            Ok(())
        })?;
        let arguments = args.iter().map(|(name, _type)| Value::new(name.clone(), _type.clone())).collect();
        let returns: Vec<Value> = returns.iter().map(|(name, _type)| Value::new(name.clone(), _type.clone())).collect();
        scope(&arguments);
        let block_body = Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            Ok(s.borrow_mut().as_mut().unwrap().block_stack.pop().ok_or(Error::msg("No block"))?)
        })?;
        let func_ptr_val = Value::new(None, Type(Rc::new(TypeInternal::Fn(arguments.iter().map(|v| v._type().clone()).collect(), returns.iter().map(|v| v._type().clone()).collect()))));
        let func_op = FuncOp {
            name: name.to_string(),
            args: arguments,
            body: block_body,
            val: func_ptr_val.clone(),
            returns,
            kind: FuncKind::Def,
        };
        let op = Op(Rc::new(OpInternal::FuncOp(func_op)));
        Self::MODULE_BUILDER.with(|s| {
            s.borrow_mut().as_mut().unwrap().body.push(op);
        });
        Ok(func_ptr_val)
    }
    */

    fn new_scope() {
        Self::MODULE_BUILDER.with(|s| {
            s.borrow_mut().as_mut().expect("No Module Builder present").op_stack.push(ScopeStack {
                stack: vec![],
                source: HashMap::new(),
            });
        });
    }

    fn with_scope<R>(mut f: impl FnMut(ScopeStack) -> R) -> R {
        Self::MODULE_BUILDER.with(|s| {
            f(s.borrow_mut().as_mut().expect("No Module Builder present").op_stack.pop().expect("No block present"))
        })
    }

    // fn build_intrinsic_fn(name: &'static str, args: Vec<(Option<String>, Type)>, returns: Vec<(Option<String>, Type)>) -> Result<()> {
    //     Self::INTRINSIC_FUNC.with(|s| {
    //         if s.borrow().contains_key(name) {
    //             return Err(Error::msg("Function already exists"));
    //         }
    //         let arguments: Vec<Value> = args.iter().map(|(name, _type)| Value::new(_type.clone(), name.clone())).collect();
    //         let returns: Vec<Value> = returns.iter().map(|(name, _type)| Value::new(_type.clone(), name.clone())).collect();
    //         let func_op = DeclareFnOp {
    //             name: name.to_string(),
    //             body: Block::new(),
    //             fn_type: FnType::Intrinsic,
    //             fn_kind: FnKind::Host,
    //         };
    //         s.borrow_mut().insert(name, Arc::new(func_op));
    //         Ok(())
    //     })
    // }

    // fn get_intrinsic_fn(name: &'static str) -> Option<Arc<DeclareFnOp>> {
    //     Self::INTRINSIC_FUNC.with(|s| {
    //         Some(s.borrow().get(name)?.clone())
    //     })
    // }

    // fn has_intrinsic_fn(name: &'static str) -> bool {
    //     Self::INTRINSIC_FUNC.with(|s| {
    //         s.borrow().contains_key(name)
    //     })
    // }

    fn insert_op(op: Op) -> Result<()> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            s.borrow_mut().as_mut().unwrap().op_stack.last_mut().ok_or(Error::msg("No block"))?.stack.push(op);
            Ok(())
        })
    }

    fn link_operand(operand: &Operand, value: &Value) -> Result<()> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            s.borrow_mut().as_mut().unwrap().op_stack.last_mut().ok_or(Error::msg("No Scope present"))?.connect(operand, value);
            Ok(())
        })
    }

    fn remap(old: &Value, new: &Value) {
        Self::MODULE_BUILDER.with(|s| {
            s.borrow_mut().as_mut().unwrap().remap.insert(old.clone(), new.clone());
        })
    }

    fn should_remap(value: &Value) -> Option<Value> {
        Self::MODULE_BUILDER.with(|s| {
            s.borrow().as_ref().unwrap().remap.get(value).cloned()
        })
    }

    fn has_remap(value: &Value) -> bool {
        Self::MODULE_BUILDER.with(|s| {
            s.borrow().as_ref().unwrap().remap.contains_key(value)
        })
    }
}

pub enum OpSource {
    SELF,
    BLOCK(usize, usize)
}

pub fn constant<T: Into<ConstValue>>(a: T, name: Option<&str>) -> Value {
    let val = a.into();
    let value_type = match val {
        ConstValue::Float(_) => Type::DType(Dtype::Float),
        ConstValue::Int(_) => Type::DType(Dtype::Int),
        ConstValue::Bool(_) => Type::DType(Dtype::Bool),
    };
    let loc = Location::from(std::panic::Location::caller().clone());
    let value = Value::new(value_type, name.map(|x| x.to_string()));
    let mut base = OperationBase::new(loc);
    base.values.push(value.clone());
    let op = Op::Constant(ConstantOp { value: val, base });
    ModuleBuilder::insert_op(op).expect("failed to insert op");
    value
}


fn lift_values_to_boundary(scope_stack: ScopeStack, should_lift: impl Fn(&Value) -> bool) -> OperationBase {
    let ScopeStack {
        stack, // The stack of ops defined by the inner scope
        source: sources // The source value of operands in the inner scope
    } = scope_stack;

    // We lift the values depended by the inner ops to the op boundary, filtered by the predicate should_lift

    // source maps operands of ops of the inner scope to values defined in the inner scope or
    //  the value created by an operand
    let mut source = HashMap::<Operand, Value>::new();
    // users maps values defined in the inner scope or the values created by operands to the list of operands
    //  that are part of inner ops
    let mut users = HashMap::<Value, Vec<Operand>>::new();
    // oper_parent maps operands of inner ops to the index of the inner op
    let mut oper_parent = HashMap::<Operand, (usize, usize)>::new();
    // val_parent maps values defined in the inner scope to its parent
    let mut val_parent = HashMap::<Value, (usize, usize)>::new();
    let mut operands = vec![];
    let mut values = vec![];

    let mut block = Block::new();
    let mut temp_remap = HashMap::new();

    for (i, op) in stack.into_iter().enumerate() {
        for oper in op.operands() {
            let v = sources.get(oper).expect("Failed to retrieve operand, graph is disconnected");

            if !should_lift(v) {
                continue;
            }

            let remapped_v = ModuleBuilder::should_remap(v).unwrap_or_else(|| v.clone());
            if !temp_remap.contains_key(v) {
                let new_oper = Operand::new_from(&remapped_v);
                users.insert(new_oper.0.clone(), vec![]);
                ModuleBuilder::link_operand(&new_oper, &remapped_v).unwrap();
                temp_remap.insert(v, new_oper.0.clone());
                values.push(new_oper.0.clone());
                operands.push(new_oper);
            }

            source.insert(oper.clone(), temp_remap.get(v).unwrap().clone());
            oper_parent.insert(oper.clone(), (0, i));
        }
        for val in op.values() {
            val_parent.insert(val.clone(), (0, i));
        }
        block.insert_op(i, op);
    }

    for (k, v) in temp_remap {
        ModuleBuilder::remap(k, &v);
    }


    todo!()
}

pub fn for_op(init: &Value, end: &Value, step: &Value, scope: impl Fn(&Value)) {
    let loc = Location::from(std::panic::Location::caller().clone());
    assert!(init._type() == step._type() && step._type() == end._type());
    assert!(init._type() == &Type::DType(Dtype::Int));
    let ind_var = Value::new(init._type().clone(), Some("i".to_string()));
    ModuleBuilder::new_scope();
    scope(&ind_var);
    ModuleBuilder::with_scope(|stack| {
        let ScopeStack {
            stack, // The stack of ops defined by the inner scope
            source: sources // The source value of operands in the inner scope
        } = stack;

        // We lift the values depended by the inner ops to the for op boundary
        // source maps operands of ops of the inner scope to values either defined in the inner scope or the induction variable, or
        //  the value created by an operand
        let mut source = HashMap::<Operand, Value>::new();
        // users maps values defined in the inner scope, the induction variable or the values created by operands to the list of operands
        //  that are part of inner operations
        let mut users = HashMap::<Value, Vec<Operand>>::new();
        // oper_parent maps operands of inner ops to the index of the inner op
        let mut oper_parent = HashMap::<Operand, (usize, usize)>::new();
        // val_parent maps values defined in the inner scope to its parent
        let mut val_parent = HashMap::<Value, (usize, usize)>::new();

        let mut operands = vec![Operand::new_from(init), Operand::new_from(end), Operand::new_from(step)];
        ModuleBuilder::link_operand(&operands[0], init).unwrap();
        ModuleBuilder::link_operand(&operands[1], end).unwrap();
        ModuleBuilder::link_operand(&operands[2], step).unwrap();

        let mut values = vec![];
        let mut block = Block::new();
        let mut temp_remap = HashMap::new();

        for (i, op) in stack.into_iter().enumerate() {
            for oper in op.operands() {
                let v = sources.get(oper).expect("Failed to retrieve operand, graph is disconnected");

                if v == &ind_var { // the induction variable is never remapped since its defined by the for_op itself
                    if !users.contains_key(&ind_var) {
                        users.insert(ind_var.clone(), vec![]);
                    }
                    users.get_mut(&ind_var).unwrap().push(oper.clone());
                    continue;
                }

                let remapped_v = ModuleBuilder::should_remap(v).unwrap_or_else(|| v.clone());
                if !temp_remap.contains_key(v) {
                    let new_oper = Operand::new_from(&remapped_v);
                    users.insert(new_oper.0.clone(), vec![]);
                    ModuleBuilder::link_operand(&new_oper, &remapped_v).unwrap();
                    temp_remap.insert(v, new_oper.0.clone());
                    values.push(new_oper.0.clone());
                    operands.push(new_oper);
                }

                source.insert(oper.clone(), temp_remap.get(v).unwrap().clone());
                oper_parent.insert(oper.clone(), (0, i));
            }
            for val in op.values() {
                val_parent.insert(val.clone(), (0, i));
            }
            block.insert_op(i, op);
        }

        for (k, v) in temp_remap {
            ModuleBuilder::remap(k, &v);
        }

        let op = ForOp {
            init_operand: operands[0].clone(),
            end_operand: operands[1].clone(),
            step_operand: operands[2].clone(),
            ind_var: ind_var.clone(),
            base: OperationBase {
                source,
                users,
                val_parent,
                oper_parent,
                operands,
                values,
                blocks: vec![block],
                loc: Some(loc.clone()),
            }
        };
        let op = Op::For(op);
        ModuleBuilder::insert_op(op).expect("failed to insert op");
    });
}

*/

fn main() {
    // let block = with_scope(|| {
    //     let a = constant(1, Some("a"))?;
    //     let b = constant(10, Some("b"))?;
    //     let c = binary_op(&a, &b, BinaryOps::Add)?;
    //     for_op(&constant(0, None)?, &b, &constant(1, None)?, |i| {
    //         assign(&c, &binary_op(&a, &i, BinaryOps::Add)?)
    //     })?;
    //     Ok(())
    // });

    // println!("{:#?}", block);
}
