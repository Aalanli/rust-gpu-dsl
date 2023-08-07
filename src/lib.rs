use std::any::Any;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread::Scope;

use anyhow::{Error, Result};

pub mod lang;
mod linear_ir;

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

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static + Sized> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait Operation: AsAny {
    fn uses(&self) -> i32;
}

struct OpTraitRegistry {
    converters: HashMap<(std::any::TypeId, std::any::TypeId), Box<dyn Any>>,
}

impl OpTraitRegistry {
    fn new() -> Self {
        OpTraitRegistry {
            converters: HashMap::new(),
        }
    }

    fn register_trait<OP: Operation + 'static, TRAIT: 'static + ?Sized>(
        &mut self,
        f: fn(&dyn Operation) -> Option<&TRAIT>,
    ) -> Result<()> {
        let id = std::any::TypeId::of::<OP>();
        let trait_id = std::any::TypeId::of::<TRAIT>();
        println!("{:?} {:?}", id, trait_id);
        // let fbox: Box<dyn Fn(&'a dyn Operation) -> Option<&'a TRAIT> + 'static> = Box::new(f1);

        if self.converters.contains_key(&(id, trait_id)) {
            return Err(Error::msg("Converter already registered"));
        }
        self.converters.insert((id, trait_id), Box::new(f));
        Ok(())
    }

    fn get_trait<'a, TRAIT: 'static + ?Sized>(&self, op: &'a dyn Operation) -> Option<&'a TRAIT> {
        let id = op.as_any().type_id();
        let trait_id = std::any::TypeId::of::<TRAIT>();
        println!("{:?} {:?}", id, trait_id);
        let f = self
            .converters
            .get(&(id, trait_id))?
            .downcast_ref::<fn(&dyn Operation) -> Option<&TRAIT>>()?;
        f(op)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    DType(Dtype),
    Fn(Vec<Type>, Vec<Type>),
    Ptr(Dtype),
    Tensor(TensorType),
}

impl Type {
    pub fn is_ptr(&self) -> bool {
        match self {
            Type::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn is_dtype(&self) -> bool {
        match self {
            Type::DType(_) => true,
            _ => false,
        }
    }

    pub fn is_fn_ptr(&self) -> bool {
        match self {
            Type::Fn(_, _) => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Type::DType(Dtype::Int) => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Type::DType(Dtype::Float) => true,
            _ => false,
        }
    }

    pub fn is_bool(&self) -> bool {
        match self {
            Type::DType(Dtype::Bool) => true,
            _ => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Dtype {
    Float,
    Int,
    Bool,
}

mod mutable_ir;

#[derive(Debug, PartialEq, Clone)]
pub enum ConstValue {
    Float(f64),
    Int(i64),
    Bool(bool),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorType {
    dtype: Dtype,
    shape: Vec<usize>,
}

impl From<f64> for ConstValue {
    fn from(value: f64) -> Self {
        ConstValue::Float(value)
    }
}

impl From<i64> for ConstValue {
    fn from(value: i64) -> Self {
        ConstValue::Int(value)
    }
}

impl From<bool> for ConstValue {
    fn from(value: bool) -> Self {
        ConstValue::Bool(value)
    }
}

#[derive(Debug, Clone)]
pub struct Value(Arc<ValueInternal>);

impl Value {
    pub fn new(_type: Type, name: Option<String>) -> Self {
        Value(Arc::new(ValueInternal {
            name: name.map(|x| x.to_string()),
            _type,
        }))
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
    pub fn new_from(val: &Value) -> Self {
        Operand(val.clone())
    }

    pub fn source(&self) -> &Value {
        &self.0
    }
}

#[derive(Debug)]
pub struct Block {
    body: Vec<Op>,
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
pub struct Op {
    op: Box<OpEnum>,
    location: Location,
}

impl Op {
    pub fn id(&self) -> u64 {
        &*self.op as *const OpEnum as u64
    }
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
    Assign(AssignOp),
    Full,
    Load,
    Store,
    Reinterpret,
    Cast,

    Reshape,
    Transpose,
    Concat,
    BroadCast,
    Slice,
    Reduce,
    ElementWise,
    Dot,
}

#[derive(Debug)]
pub struct AssignOp {
    lhs: Operand,
    rhs: Operand,
}

#[derive(Debug)]
pub struct FullOp {
    shape: Vec<usize>,
    value: Operand,
    ret: Value,
}

#[derive(Debug)]
pub struct LoadOp {
    ptr: Operand,
    mask: Operand,
    ret: Operand,
}

#[derive(Debug)]
pub struct StoreOp {
    ptr: Operand,
    mask: Operand,
    ret: Operand,
}

#[derive(Debug)]
pub struct ReinterpretOp {
    value: Operand,
    _type: Type,
    ret: Value,
}

#[derive(Debug)]
pub struct CastOp {
    value: Operand,
    _type: Type,
    ret: Value,
}

#[derive(Debug)]
pub struct ReshapeOp {
    value: Operand,
    shape: Vec<usize>,
    ret: Value,
}

#[derive(Debug)]
pub struct TransposeOp {
    value: Operand,
    axes: Vec<usize>,
    ret: Value,
}

#[derive(Debug)]
pub struct ConcatOp {
    values: Vec<Operand>,
    axis: usize,
    ret: Value,
}

#[derive(Debug)]
pub struct BroadCastOp {
    value: Operand,
    shape: Vec<usize>,
    ret: Value,
}

#[derive(Debug)]
pub struct SliceOp {
    value: Operand,
    start: Vec<usize>,
    end: Vec<usize>,
    ret: Value,
}

#[derive(Debug)]
pub struct ReduceOp {
    value: Operand,
    axes: Vec<usize>,
    keep_dim: bool,
    ret: Value,
}

// #[derive(Debug)]
// pub struct ElementWiseOp {
//     values: Vec<Operand>,
//     ret: Value
// }

#[derive(Debug)]
pub struct DotOp {
    lhs: Operand,
    rhs: Operand,
    ret: Value,
}

#[derive(Debug)]
pub struct ConstantOp {
    value: ConstValue,
    ret: Value,
}

#[derive(Debug)]
pub struct IfOp {
    cond_operand: Operand,
    then_block: Block,
    else_block: Block,
}

#[derive(Debug)]
pub struct ForOp {
    init_operand: Operand,
    end_operand: Operand,
    step_operand: Operand,
    ind_var: Value,
    body: Block,
}

#[derive(Debug)]
pub struct WhileOp {
    init_cond: Operand,
    cond_val: Value,
    body: Block,
}

#[derive(Debug)]
pub struct DeclareFnOp {
    name: String,
    fn_type: FnType,
    body: Block,
    fn_val: Value,
}

#[derive(Debug)]
pub enum FnType {
    Intrinsic(&'static str),
    Defined(Value),
}

#[derive(Debug)]
pub enum FnKind {
    Device,
    Host,
}

#[derive(Debug)]
pub struct CallOp {
    fn_type: FnType,
    args: Vec<Operand>,
    ret: Vec<Value>,
}

#[derive(Debug)]
pub struct YieldOp {
    base: Vec<Operand>,
}

#[derive(Debug)]
pub struct ModuleOp {
    name: Option<String>,
    body: Block,
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

impl<'a, 'b> From<&'b std::panic::Location<'a>> for Location {
    fn from(value: &'b std::panic::Location<'a>) -> Self {
        Location {
            row: value.line(),
            col: value.column(),
            file: value.file().to_string(),
        }
    }
}

pub struct ModuleBuilder {
    name: Option<String>,
    op_stack: Vec<Vec<Op>>,
}

impl ModuleBuilder {
    thread_local! {
        static MODULE_BUILDER: RefCell<ModuleBuilder> = RefCell::new(ModuleBuilder::new());
    }

    pub fn new() -> Self {
        ModuleBuilder {
            name: None,
            op_stack: vec![],
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn push_op(op: Op) -> Result<()> {
        Self::MODULE_BUILDER.with(|builder| {
            builder
                .borrow_mut()
                .op_stack
                .last_mut()
                .ok_or(Error::msg("No block to push op to"))?
                .push(op);
            Ok(())
        })
    }

    pub fn with_scope(mut scope: impl FnMut() -> Result<()>) -> Result<Block> {
        Self::MODULE_BUILDER.with(|builder| {
            builder.borrow_mut().op_stack.push(vec![]);
            scope()?;
            let block = Block {
                body: builder
                    .borrow_mut()
                    .op_stack
                    .pop()
                    .ok_or(Error::msg("failed to pop block"))?,
            };
            Ok(block)
        })
    }

    pub fn build_module(mut self, mut scope: impl FnMut() -> Result<()>) -> Result<ModuleOp> {
        Self::MODULE_BUILDER.with(|builder| {
            self.op_stack.push(vec![]);
            let old = builder.replace(self);
            scope()?;
            let mut orig = builder.replace(old);
            let block = Block {
                body: orig
                    .op_stack
                    .pop()
                    .ok_or(Error::msg("failed to pop block"))?,
            };
            Ok(ModuleOp {
                name: orig.name,
                body: block,
            })
        })
    }
}

fn make_binary_op(loc: Location, op: &'static str, a: &Value, b: &Value) -> Result<Value> {
    let op = CallOp {
        fn_type: FnType::Intrinsic(op),
        args: vec![Operand::new_from(a), Operand::new_from(b)],
        ret: vec![Value::new(a._type().clone(), None)],
    };
    let val = op.ret[0].clone();
    let op = Op {
        op: Box::new(OpEnum::Call(op)),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(val)
}

pub fn cast(a: &Value, _type: &Type) -> Result<Value> {
    match _type {
        Type::Fn(_, _) => {
            return Err(Error::msg("Can't cast to function pointer"));
        }
        Type::Ptr(_) => {
            if !(a._type().is_ptr() || a._type().is_int()) {
                return Err(Error::msg("Can only cast pointer to pointer or int"));
            }
        }
        _ => {}
    }

    let loc: Location = std::panic::Location::caller().into();
    let op = CallOp {
        fn_type: FnType::Intrinsic("cast"),
        args: vec![Operand::new_from(a)],
        ret: vec![Value::new(_type.clone(), None)],
    };
    let val = op.ret[0].clone();
    let op = Op {
        op: Box::new(OpEnum::Call(op)),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(val)
}

pub fn add(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !(a._type().is_float() || a._type().is_int()) {
        return Err(Error::msg("Only Int and Float allowed for add"));
    }
    make_binary_op(loc, "add", a, b)
}

pub fn sub(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !(a._type().is_float() || a._type().is_int()) {
        return Err(Error::msg("Only Int and Float allowed for sub"));
    }
    make_binary_op(loc, "sub", a, b)
}

pub fn mul(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !(a._type().is_float() || a._type().is_int()) {
        return Err(Error::msg("Only Int and Float allowed for mul"));
    }
    make_binary_op(loc, "mul", a, b)
}

pub fn div(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !(a._type().is_float() || a._type().is_int()) {
        return Err(Error::msg("Only Int and Float allowed for div"));
    }
    make_binary_op(loc, "div", a, b)
}

pub fn eq(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !a._type().is_dtype() {
        return Err(Error::msg("Only Dtypes are allowed for eq"));
    }
    let op = CallOp {
        fn_type: FnType::Intrinsic("eq"),
        args: vec![Operand::new_from(a), Operand::new_from(b)],
        ret: vec![Value::new(Type::DType(Dtype::Bool), None)],
    };
    let val = op.ret[0].clone();
    let op = Op {
        op: Box::new(OpEnum::Call(op)),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(val)
}

pub fn gt(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !a._type().is_dtype() {
        return Err(Error::msg("Only Dtypes are allowed for eq"));
    }
    let op = CallOp {
        fn_type: FnType::Intrinsic("gt"),
        args: vec![Operand::new_from(a), Operand::new_from(b)],
        ret: vec![Value::new(Type::DType(Dtype::Bool), None)],
    };
    let val = op.ret[0].clone();
    let op = Op {
        op: Box::new(OpEnum::Call(op)),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(val)
}

pub fn geq(a: &Value, b: &Value) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    if a._type() != b._type() {
        return Err(Error::msg("Type mismatch"));
    } else if !a._type().is_dtype() {
        return Err(Error::msg("Only Dtypes are allowed for geq"));
    }
    let op = CallOp {
        fn_type: FnType::Intrinsic("geq"),
        args: vec![Operand::new_from(a), Operand::new_from(b)],
        ret: vec![Value::new(Type::DType(Dtype::Bool), None)],
    };
    let val = op.ret[0].clone();
    let op = Op {
        op: Box::new(OpEnum::Call(op)),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(val)
}

pub fn constant<T: Into<ConstValue>>(value: T) -> Result<Value> {
    let loc: Location = std::panic::Location::caller().into();
    let value = value.into();
    let ret = Value::new(
        Type::DType(match value {
            ConstValue::Float(_) => Dtype::Float,
            ConstValue::Int(_) => Dtype::Int,
            ConstValue::Bool(_) => Dtype::Bool,
        }),
        None,
    );
    let op = Op {
        op: Box::new(OpEnum::Constant(ConstantOp {
            value,
            ret: ret.clone(),
        })),
        location: loc,
    };
    ModuleBuilder::push_op(op)?;
    Ok(ret)
}

pub fn if_op(
    cond: &Value,
    if_body: impl Fn() -> Result<()>,
    else_body: impl Fn() -> Result<()>,
) -> Result<()> {
    let loc: Location = std::panic::Location::caller().into();
    let if_body = ModuleBuilder::with_scope(if_body)?;
    let else_body = ModuleBuilder::with_scope(else_body)?;
    let op = Op {
        op: Box::new(OpEnum::If(IfOp {
            cond_operand: Operand::new_from(cond),
            then_block: if_body,
            else_block: else_body,
        })),
        location: loc,
    };
    ModuleBuilder::push_op(op)
}

pub fn for_op(
    start: &Value,
    end: &Value,
    step: &Value,
    body: impl Fn(&Value) -> Result<()>,
) -> Result<()> {
    let loc: Location = std::panic::Location::caller().into();
    let ind_var = Value::new(Type::DType(Dtype::Int), None);
    let body = ModuleBuilder::with_scope(|| body(&ind_var))?;
    let op = Op {
        op: Box::new(OpEnum::For(ForOp {
            init_operand: Operand::new_from(start),
            end_operand: Operand::new_from(end),
            step_operand: Operand::new_from(step),
            ind_var,
            body,
        })),
        location: loc,
    };
    ModuleBuilder::push_op(op)
}

#[test]
fn test_construction() {
    let module = ModuleBuilder::new()
        .name("TestModule".to_string())
        .build_module(|| {
            let a = constant(1.0)?;
            let b = constant(2.0)?;
            let c = constant(true)?;
            if_op(
                &c,
                || {
                    for_op(&a, &b, &constant(1)?, |ind_var| {
                        println!("{:?}", ind_var);
                        Ok(())
                    })
                },
                || Ok(()),
            )?;
            Ok(())
        });

    println!("{:#?}", module);
}
