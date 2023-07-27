use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::collections::HashMap;
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
    vparent_op: HashMap<Value, Op>,
    oper_parent_op: HashMap<Operand, Op>,
    loc: Option<Location>,
    op: OpEnum
}

impl Op {
    pub fn new(op: OpEnum, loc: Option<Location>) -> Self {
        Op {
            parent_def: HashMap::new(),
            users: HashMap::new(),
            vparent_op: HashMap::new(),
            oper_parent_op: HashMap::new(),
            loc,
            op: op
        }
    }
}

#[derive(Debug)]
pub enum OpEnum {
    Constant(Arc<ConstantOp>),
    If(Arc<IfOp>),
    For(Arc<ForOp>),
    While(Arc<WhileOp>),
    DeclareFn(Arc<DeclareFnOp>),
    Call(Arc<CallOp>),
    Yield(Arc<YieldOp>),
    Module(Arc<ModuleOp>),
}

impl Op {
    pub fn operands(&self) -> Option<&[Operand]> {
        match &self.op {
            OpEnum::Constant(_) => None,
            OpEnum::If(op) => Some(&op.operands),
            OpEnum::For(op) => Some(&op.operands),
            OpEnum::While(op) => Some(&op.operands),
            OpEnum::DeclareFn(_) => None,
            OpEnum::Call(op) => Some(&op.args),
            OpEnum::Yield(op) => Some(&op.operand),
            OpEnum::Module(_) => None,
        }
    }

    pub fn values(&self) -> Option<&[Value]> {
        match &self.op {
            OpEnum::Constant(op) => Some(&op.ret),
            OpEnum::If(op) => Some(&op.ret),
            OpEnum::For(op) => Some(&op.ret),
            OpEnum::While(op) => Some(&op.ret),
            OpEnum::DeclareFn(op) => Some(&op.ret),
            OpEnum::Call(op) => Some(&op.ret),
            OpEnum::Yield(_) => None,
            OpEnum::Module(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct ConstantOp {
    value: ConstValue,
    ret: Vec<Value>,
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
    args: Vec<Value>,
    ret: Vec<Value>,
    fn_type: FnType,
    fn_kind: FnKind,
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
    func: Op,
    args: Vec<Operand>,
    ret: Vec<Value>,
}

#[derive(Debug)]
pub struct YieldOp { // terminator
    operand: Vec<Operand>
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
    op_stack: Vec<ScopeStack>
}

struct ScopeStack {
    stack: Vec<Op>,
    oper_refer: HashMap<Operand, Value>,
}

impl ScopeStack {
    fn connect(&mut self, operand: &Operand, value: &Value) {
        self.oper_refer.insert(operand.clone(), value.clone());
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
                oper_refer: HashMap::new(),
            });
        });
    }

    fn with_scope<R>(mut f: impl FnMut(ScopeStack) -> R) -> R {
        Self::MODULE_BUILDER.with(|s| {
            f(s.borrow_mut().as_mut().expect("No Module Builder present").op_stack.pop().expect("No block present"))
        })
    }

    fn build_intrinsic_fn(name: &'static str, args: Vec<(Option<String>, Type)>, returns: Vec<(Option<String>, Type)>) -> Result<()> {
        Self::INTRINSIC_FUNC.with(|s| {
            if s.borrow().contains_key(name) {
                return Err(Error::msg("Function already exists"));
            }
            let arguments: Vec<Value> = args.iter().map(|(name, _type)| Value::new(_type.clone(), name.clone())).collect();
            let returns: Vec<Value> = returns.iter().map(|(name, _type)| Value::new(_type.clone(), name.clone())).collect();
            let func_op = DeclareFnOp {
                name: name.to_string(),
                args: arguments,
                body: Block { values: vec![], body: vec![] },
                ret: returns,
                fn_type: FnType::Intrinsic,
                fn_kind: FnKind::Host,
            };
            s.borrow_mut().insert(name, Arc::new(func_op));
            Ok(())
        })
    }

    fn get_intrinsic_fn(name: &'static str) -> Option<Arc<DeclareFnOp>> {
        Self::INTRINSIC_FUNC.with(|s| {
            Some(s.borrow().get(name)?.clone())
        })
    }

    fn has_intrinsic_fn(name: &'static str) -> bool {
        Self::INTRINSIC_FUNC.with(|s| {
            s.borrow().contains_key(name)
        })
    }

    fn insert_op(op: Op) -> Result<()> {
        Self::MODULE_BUILDER.with(|s| {
            if s.borrow().is_none() {
                return Err(Error::msg("ModuleBuilder does not exist"));
            }
            s.borrow_mut().as_mut().unwrap().op_stack.last_mut().ok_or(Error::msg("No block"))?.stack.push(op);
            Ok(())
        })
    }
}

pub fn constant<T: Into<ConstValue>>(a: T, name: Option<&str>) -> Value {
    let val = a.into();
    let value_type = match val {
        ConstValue::Float(_) => Type::DType(Dtype::Float),
        ConstValue::Int(_) => Type::DType(Dtype::Int),
        ConstValue::Bool(_) => Type::DType(Dtype::Bool),
    };
    let value = Value::new(value_type, name.map(|x| x.to_string()));
    let op = Op::new(OpEnum::Constant(Arc::new(ConstantOp { value: val, ret: vec![value.clone()] })), None);
    ModuleBuilder::insert_op(op).expect("failed to insert op");
    value
}


pub fn for_op(init: &Value, step: &Value, end: &Value, scope: impl Fn()) {
    ModuleBuilder::new_scope();
    scope();
    let op = ModuleBuilder::with_scope(|stack| {
        let ScopeStack { mut stack, mut oper_refer } = stack;

        let mut users = HashMap::<Value, Vec<Operand>>::new();
        let mut oper_parent = HashMap::<Operand, Op>::new();
        let mut val_parent = HashMap::<Value, Op>::new();

        for i in stack {
            if let Some(oper) = i.operands() {
                // add users to source value
                for p in oper {
                    let v = oper_refer.get(p).expect("Failed to retrieve operand, graph is disconnected");
                    if !users.contains_key(v) {
                        users.insert(v.clone(), vec![]);
                    }
                    users.get_mut(v).unwrap().push(p.clone());
                }
                // add parent op to operand
                // for p in oper {
                //     oper_parent.insert(p.clone(), i.clone());
                // }
            }
            if let Some(v) = i.values() {
                for vi in v {
                    
                }

            }
            println!("{:?}", i);
        }
    });
}

#[test]
fn test_arc() {
    let a = Arc::new(());
    let b = Arc::new(());
    println!("{:#?}", Arc::as_ptr(&a));
    println!("{:#?}", Arc::as_ptr(&b));
}