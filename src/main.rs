use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::cell::{RefCell, Ref};
use std::collections::HashMap;
use anyhow::{Result, Error};


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

        Ok(ModuleOp { name: builder.name, body: Block { ops: builder.body } })
    }

    pub fn build_fn(name: &str, args: Vec<(Option<String>, Type)>, returns: Vec<(Option<String>, Type)>, scope: impl Fn(&Vec<Value>)) -> Result<Value> {
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
    
    pub fn build_intrinsic_fn(name: &'static str, args: Vec<(Option<String>, Type)>, returns: Vec<(Option<String>, Type)>) -> Result<Value> {
        Self::INTRINSIC_FUNC.with(|s| {
            if s.borrow().contains_key(name) {
                return Err(Error::msg("Function already exists"));
            }
            let arguments: Vec<Value> = args.iter().map(|(name, _type)| Value::new(name.clone(), _type.clone())).collect();
            let returns: Vec<Value> = returns.iter().map(|(name, _type)| Value::new(name.clone(), _type.clone())).collect();
            let func_ptr_val = Value::new(None, Type(Rc::new(TypeInternal::Fn(arguments.iter().map(|v| v._type().clone()).collect(), returns.iter().map(|v| v._type().clone()).collect()))));
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
        Self::INTRINSIC_FUNC.with(|s| {
            Some(s.borrow().get(name)?.val.clone())
        })
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
            s.borrow_mut().as_mut().unwrap().block_stack.last_mut().ok_or(Error::msg("No block"))?.ops.push(op);
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
        Value(Rc::new(ValueInternal {
            name,
            _type,
        }))
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
    Module(ModuleOp)
}


#[derive(Debug)]
pub struct ModuleOp {
    name: Option<String>,
    body: Block,
}

#[derive(Debug)]
pub struct ConstantOp {
    value: Constant,
    result: Value
}

#[derive(Debug)]
pub struct IfOp {
    cond: Value,
    then_block: Block,
    else_block: Block
}

#[derive(Debug)]
pub struct WhileOp {
    cond: Value,
    body: Block
}
#[derive(Debug)]
pub struct ForOp {
    init: Value,
    fin: Value,
    step: Value,
    body: Block
}

#[derive(Debug)]
pub struct AssignOp {
    lhs: Value,
    rhs: Value
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
    Intrinsic
}

#[derive(Debug)]
pub struct YieldOp {
    value: Vec<Value>
}

#[derive(Debug)]
pub struct CallOp {
    func: Value,
    args: Vec<Value>,
    returns: Vec<Value>
}

#[derive(Debug)]
pub struct CastOp {
    value: Value,
    result: Value,
    _type: Type,
}


#[derive(Debug)]
struct Block {
    ops: Vec<Op>
}

impl Block {
    fn new() -> Self {
        Block {
            ops: vec![]
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Type(Rc<TypeInternal>);

impl Type {
    fn is_integer(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Int) => true,
            _ => false
        }
    }

    fn is_float(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Float) => true,
            _ => false
        }
    }

    fn is_bool(&self) -> bool {
        match &*self.0 {
            TypeInternal::Val(Dtype::Bool) => true,
            _ => false
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
    Bool
}

impl Dtype {
    fn bit_width(&self) -> u64 {
        match self {
            Dtype::Int => 64,
            Dtype::Float => 64,
            Dtype::Bool => 8
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
    op: BinaryOps
}

#[derive(Debug, Clone)]
enum BinaryOps { // (lhs, rhs, result)
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
    op: UnaryOps
}

#[derive(Debug, Clone)]
enum UnaryOps { // (val, result)
    Neg,
    Not,
}


fn constant<T: Into<Constant>>(val: T, name: Option<&str>) -> Result<Value> {
    let constant: Constant = val.into();
    let result = Value::new(name.map(|s| s.to_string()), constant.get_type());

    let constant_op = ConstantOp {
        value: constant,
        result: result.clone()
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