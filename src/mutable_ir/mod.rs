use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::cell::{RefCell, Ref};
use std::hash::{Hash, Hasher};
use anyhow::{Result, Error};

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
            _ => false
        }
    }

    pub fn is_dtype(&self) -> bool {
        match self {
            Type::DType(_) => true,
            _ => false
        }
    }

    pub fn is_fn_ptr(&self) -> bool {
        match self {
            Type::Fn(_, _) => true,
            _ => false
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Type::DType(Dtype::Int) => true,
            _ => false
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Type::DType(Dtype::Float) => true,
            _ => false
        }
    }

    pub fn is_bool(&self) -> bool {
        match self {
            Type::DType(Dtype::Bool) => true,
            _ => false
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
 pub enum Dtype {
    Float,
    Int,
    Bool
}

#[derive(Debug, PartialEq, Clone)]
pub enum ConstValue {
    Float(f64),
    Int(i64),
    Bool(bool)
}

impl ConstValue {
    fn type_of(&self) -> Type {
        match self {
            ConstValue::Float(_) => Type::DType(Dtype::Float),
            ConstValue::Int(_) => Type::DType(Dtype::Int),
            ConstValue::Bool(_) => Type::DType(Dtype::Bool),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorType {
    dtype: Dtype,
    shape: Vec<usize>,
}

#[derive(Clone)]
struct Value(Rc<ValueImpl>);
impl Value {
    fn type_of(&self) -> Type {
        self.0.dtype.clone()
    }
}
struct ValueImpl {
    dtype: Type,
}

impl Value {
    fn new_value_from(operand: &Operand) -> Value {
        Value(Rc::new(ValueImpl {
            dtype: operand.0.parent.0.dtype.clone(),
        }))
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
        Rc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Clone)]
struct Operand(Rc<OperandImpl>);
struct OperandImpl {
    parent: Value
}

impl Operand {
    fn from_value(value: Value) -> Operand {
        Operand(Rc::new(OperandImpl {
            parent: value
        }))
    }

    fn type_of(&self) -> Type {
        self.0.parent.0.dtype.clone()
    }

    fn parent_of(&self) -> &Value {
        &self.0.parent
    }
}

#[derive(Clone)]
struct Op(Rc<RefCell<OpEnum>>);

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Op {}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl Deref for Op {
    type Target = Rc<RefCell<OpEnum>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Op {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

enum OpEnum {
    Const(ConstOp),
    For(ForOp),
    While(WhileOp),
    If(IfOp),
    BinaryElementWise(BinaryElementWise),
    Module(ModuleOp),
    Yield(YieldOp),
}


impl OpEnum {
    fn values(&self) -> Vec<&Value> {
        match &self {
            OpEnum::Const(op) => vec![&op.result],
            OpEnum::For(op) => op.body.arguments.iter().collect(),
            OpEnum::While(op) => op.body.arguments.iter().collect(),
            OpEnum::If(op) => op.then_body.arguments.iter().collect(),
            OpEnum::Module(_) => vec![],
            OpEnum::BinaryElementWise(op) => vec![&op.result],
            OpEnum::Yield(_) => vec![],
        }
    }

    fn operands(&self) -> Vec<&Operand> {
        match &self {
            OpEnum::Const(_) => vec![],
            OpEnum::For(op) => op.carried.iter().collect(),
            OpEnum::While(op) => op.carried.iter().collect(),
            OpEnum::If(op) => op.carried.iter().collect(),
            OpEnum::Module(_) => vec![],
            OpEnum::BinaryElementWise(op) => vec![&op.lhs, &op.rhs],
            OpEnum::Yield(op) => op.values.iter().collect(),
        }
    }

    fn verify(&self) -> Result<()> {
        match &self {
            OpEnum::Const(op) => {
                if op.value.type_of() != op.result.type_of() {
                    return Err(Error::msg("ConstOp: dtype mismatch"));
                }
            },
            OpEnum::For(op) => {
                if op.start.type_of() != Type::DType(Dtype::Int) {
                    return Err(Error::msg("ForOp: start is not int"));
                }
                if op.end.type_of() != Type::DType(Dtype::Int) {
                    return Err(Error::msg("ForOp: end is not int"));
                }
                if op.step.type_of() != Type::DType(Dtype::Int) {
                    return Err(Error::msg("ForOp: step is not int"));
                }
                if op.ind_var.type_of() != Type::DType(Dtype::Int) {
                    return Err(Error::msg("ForOp: ind_var is not int"));
                }
                if op.ind_var != op.body.arguments[0] {
                    return Err(Error::msg("ForOp: ind_var is not first argument"));
                }
                if op.body.arguments.len() != op.carried.len() + 1 {
                    return Err(Error::msg("ForOp: number of arguments mismatch"));
                }
                for (value, operand) in op.body.arguments[1..].iter().zip(op.carried.iter()) {
                    if value.type_of() != operand.type_of() {
                        return Err(Error::msg("ForOp: carried argument type mismatch"));
                    }
                }
                match op.body.body.last() {
                    Some(terminator) => {
                        if let OpEnum::Yield(yield_op) = &*terminator.borrow() {
                            if yield_op.values.len() + 1 != op.body.arguments.len() {
                                return Err(Error::msg("ForOp: number of yielded values mismatch"));
                            }
                            for (value, operand) in yield_op.values.iter().zip(op.body.arguments[1..].iter()) {
                                if value.parent_of() != operand {
                                    return Err(Error::msg("ForOp: yielded value is not the same as block arguments"));
                                }
                            }
                        } else {
                            return Err(Error::msg("ForOp: body does not end with yield"));
                        }
                    },
                    None => {
                        return Err(Error::msg("ForOp: body is empty"));
                    }
                }
                op.body.verify()?;
            },
            // TODO
            _ => {}
        }

        Ok(())
    }
}

fn test() {
    let mut a = 3;
    let b = &mut a;
    t(b);
    t(b);
}

fn t(a: &mut i32) {
    println!("{}", a);
    *a *= 2;
}

struct Block {
    arguments: Vec<Value>,
    body: Vec<Op>
}

impl Block {
    fn verify(&self) -> Result<()> {
        for op in self.body.iter() {
            for operand in op.borrow().operands() {
                if !self.arguments.contains(operand.parent_of()) {
                    return Err(Error::msg("Block: operand not in arguments"));
                }
            }
        }
        for op in self.body.iter() {
            op.borrow().verify()?;
        }
        Ok(())
    }
}

struct ConstOp {
    value: ConstValue,
    result: Value,
}

struct ForOp {
    start: Operand,
    end: Operand,
    step: Operand,
    carried: Vec<Operand>,

    ind_var: Value,
    body: Block,
}


struct WhileOp {
    carried: Vec<Operand>,
    cond_region: Block,
    body: Block
}

struct IfOp {
    cond: Operand,
    carried: Vec<Operand>,
    then_body: Block,
    else_body: Block,
}

struct BinaryElementWise {
    lhs: Operand,
    rhs: Operand,
    result: Value,
}

struct ModuleOp {
    body: Block,
}

struct YieldOp {
    values: Vec<Operand>,
}

struct UsageAnalysis {
    users: HashMap<Value, Vec<Operand>>,
    parent_val: HashMap<Value, Op>,
    parent_oper: HashMap<Operand, Op>,
    parent_op: HashMap<Op, Op>,
}

// impl UsageAnalysis {
//     fn on_operation(op: Op) -> UsageAnalysis {
//         let mut users = HashMap::new();
//         let mut parent = HashMap::new();
//         let mut parent_op = HashMap::new();
//         for use_ in op.borrow().recursive_uses() {
//             users.entry(use_).or_insert_with(|| Vec::new()).push(op.clone());
//         }

//         for values in op.borrow().values() {
//             parent.insert(values.clone(), op.clone());
//         }

//         UsageAnalysis {
//             users,
//             parent,
//             parent_op
//         }
//     }
// }

// fn for_every_backward(op: &Op, mut f: impl FnMut(&Op)) {
//     for block in op.borrow().blocks() {
//         for op in block.iter().rev() {
//             for_every_backward(op, &mut f);
//         }
//     }
//     f(op);
// }



// struct DCE {
//     analysis: UsageAnalysis,
//     alive: HashSet<Value>
// }

// impl DCE {
//     fn run(&mut self, op: &Op) {
//         let mut visited = HashSet::new();
//         let mut to_visit = vec![];
//         for_every_backward(op, |op| {
//             match &*op.borrow() {
//                 OpEnum::Return(rop) => {
//                     for value in rop.values.iter().rev() {
//                         self.alive.insert(value.clone());
//                         visited.insert(op.clone());
//                         to_visit.push(self.analysis.parent_op.get(op).unwrap().clone());
//                     }
                    
//                 },
//                 _ => {}
//             }
//         });

//         while to_visit.len() > 0 {
//             let op = to_visit.pop().unwrap();
//             if visited.contains(&op) {
//                 continue;
//             }
//             match *op.borrow() {
//                 OpEnum::Const(_) => {},
//                 OpEnum::For(_) => {},
//                 OpEnum::While(_) => {},
//                 OpEnum::If(_) => {},
//                 OpEnum::Add(_) => {},
//                 OpEnum::Module(_) => {},
//                 OpEnum::Return(_) => {},
//                 OpEnum::AssignOp(_) => {}
//             }
//             visited.insert(op.clone());
//             for value in op.borrow().values() {
//                 self.alive.insert(value.clone());
//             }
//             for block in op.borrow().blocks() {
//                 for op in block.iter().rev() {
//                     to_visit.push(op.clone());
//                 }
//             }
//         }
//     }
// }