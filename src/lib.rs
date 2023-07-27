use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::collections::HashMap;
use std::cell::{RefCell, Ref};

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


#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    DType(Dtype),
    FnPtr(Vec<Type>, Vec<Type>),
    Ptr(Dtype)
}

#[derive(Debug, PartialEq, Eq)]
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
    pub fn new(_type: Type, name: Option<&str>) -> Self {
        Value(Arc::new(
            ValueInternal { name: name.map(|x| x.to_string()), _type }
        ))
    }
}

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

#[derive(Debug)]
pub struct Operand {
    id: Arc<()>
}

impl Operand {
    fn new() -> Self {
        Operand { id: Arc::new(()) }
    }
}

#[derive(Debug)]
pub struct Op {
    parent_def: HashMap<Operand, Value>,
    users: HashMap<Value, Vec<Operand>>,
    loc: Location,
    op: Arc<OpEnum>
}

#[derive(Debug)]
pub enum OpEnum {
    Constant(ConstantOp),
    If(IfOp),
    For(ForOp),
    While(WhileOp),
    DeclareFn(DeclareFnOp),
    Call(CallOp),
    Yield(YieldOp),
    Module(ModuleOp),
}

#[derive(Debug)]
pub struct ConstantOp {
    value: ConstValue,
    ret: Value,
}

#[derive(Debug)]
pub struct IfOp {
    cond_operand: Operand,
    cond_val: Value,
    operands: Vec<Operand>,
    ret: Vec<Value>,
    then_body: Block,
    else_body: Block,
}

#[derive(Debug)]
pub struct ForOp {
    init_operand: Operand,
    end_operand: Operand,
    step_operand: Operand,
    ind_var: Value,
    operands: Vec<Operand>,
    ret: Vec<Value>,
    body: Block,
}

#[derive(Debug)]
pub struct WhileOp {
    init_cond: Operand,
    cond_val: Value,
    operands: Vec<Operand>,
    ret: Vec<Value>,
    body: Block,
}

#[derive(Debug)]
pub struct DeclareFnOp {
    name: String,
    body: Block,
    ret: Value,
    fn_type: FnType
}

#[derive(Debug)]
pub enum FnType {
    Intrinsic,
    Defined
}

#[derive(Debug)]
pub struct CallOp {
    fn_arg: Operand,
    args: Vec<Operand>,
    ret: Vec<Value>,
}

#[derive(Debug)]
pub struct YieldOp { // terminator
    operand: Operand
}

#[derive(Debug)]
pub struct ModuleOp {
    name: Option<String>,
    body: Vec<Op>
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

#[derive(Debug)]
pub struct Block {
    values: Vec<Value>,
    body: Vec<Op>
}

pub struct ModuleBuilder {
    name: Option<String>,
    body: Vec<Op>,
    block_stack: Vec<Block>
}

/*
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
*/
#[test]
fn test_arc() {
    let a = Arc::new(());
    let b = Arc::new(());
    println!("{:#?}", Arc::as_ptr(&a));
    println!("{:#?}", Arc::as_ptr(&b));
}