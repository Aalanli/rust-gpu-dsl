use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::cell::{RefCell, Ref};
use std::hash::{Hash, Hasher};

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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TensorType {
    dtype: Dtype,
    shape: Vec<usize>,
}

#[derive(Clone)]
struct Value(Rc<ValueImpl>);
struct ValueImpl {
    dtype: Dtype,
    value: ConstValue
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
    Add(AddOp),
    Module(ModuleOp),
    Return(ReturnOp),
    AssignOp(AssignOp)
}

use std::iter;
impl OpEnum {
    fn values(&self) -> Box<dyn DoubleEndedIterator<Item=&Value> + '_> {
        match &self {
            OpEnum::Const(op) => Box::new(iter::once(&op.result)),
            OpEnum::For(_) => Box::new(iter::empty()),
            OpEnum::While(_) => Box::new(iter::empty()),
            OpEnum::If(_) => Box::new(iter::empty()),
            OpEnum::Add(op) => Box::new(iter::once(&op.result)),
            OpEnum::Module(_) => Box::new(iter::empty()),
            OpEnum::Return(_) => Box::new(iter::empty()),
            OpEnum::AssignOp(_) => Box::new(iter::empty())
        }
    }

    fn values_mut(&mut self) -> Box<dyn DoubleEndedIterator<Item=&mut Value> + '_> {
        match self {
            OpEnum::Const(op) => Box::new(iter::once(&mut op.result)),
            OpEnum::For(op) => Box::new(iter::empty()),
            OpEnum::While(_) => Box::new(iter::empty()),
            OpEnum::If(_) => Box::new(iter::empty()),
            OpEnum::Add(op) => Box::new(iter::once(&mut op.result)),
            OpEnum::Module(_) => Box::new(iter::empty()),
            OpEnum::Return(_) => Box::new(iter::empty()),
            OpEnum::AssignOp(_) => Box::new(iter::empty())

        }
    }

    fn uses(&self) -> Box<dyn DoubleEndedIterator<Item=&Value> + '_> {
        match &self {
            OpEnum::Const(op) => Box::new(iter::once(&op.result)),
            OpEnum::For(op) => Box::new(iter::once(&op.start).chain(iter::once(&op.end)).chain(iter::once(&op.step))),
            OpEnum::While(op) => Box::new(iter::once(&op.cond)),
            OpEnum::If(op) => Box::new(iter::once(&op.cond)),
            OpEnum::Add(op) => Box::new(vec![&op.lhs, &op.rhs].into_iter()),
            OpEnum::Module(_) => Box::new(iter::empty()),
            OpEnum::Return(op) => Box::new(op.values.iter()),
            OpEnum::AssignOp(op) => Box::new(iter::once(&op.lhs).chain(iter::once(&op.rhs)))
        }
    }

    fn uses_mut(&mut self) -> Box<dyn DoubleEndedIterator<Item = &mut Value> + '_> {
        match self {
            OpEnum::Const(op) => Box::new(iter::once(&mut op.result)),
            OpEnum::For(op) => Box::new(iter::once(&mut op.start).chain(iter::once(&mut op.end)).chain(iter::once(&mut op.step))),
            OpEnum::While(op) => Box::new(iter::once(&mut op.cond)),
            OpEnum::If(op) => Box::new(iter::once(&mut op.cond)),
            OpEnum::Add(op) => Box::new(vec![&mut op.lhs, &mut op.rhs].into_iter()),
            OpEnum::Module(_) => Box::new(iter::empty()),
            OpEnum::Return(op) => Box::new(op.values.iter_mut()),
            OpEnum::AssignOp(op) => Box::new(iter::once(&mut op.lhs).chain(iter::once(&mut op.rhs)))
        }
    }

    fn blocks(&self) -> Box<dyn DoubleEndedIterator<Item = &Vec<Op>> + '_> {
        match &self {
            OpEnum::Const(_) => Box::new(iter::empty()),
            OpEnum::For(op) => Box::new(iter::once(&op.body)),
            OpEnum::While(op) => Box::new(iter::once(&op.body)),
            OpEnum::If(op) => Box::new(iter::once(&op.then_body).chain(iter::once(&op.else_body))),
            OpEnum::Add(_) => Box::new(iter::empty()),
            OpEnum::Module(op) => Box::new(iter::once(&op.body)),
            OpEnum::Return(_) => Box::new(iter::empty()),
            OpEnum::AssignOp(_) => Box::new(iter::empty())
        }
    }

    fn blocks_mut(&mut self) -> Box<dyn DoubleEndedIterator<Item = &mut Vec<Op>> + '_> {
        match self {
            OpEnum::Const(_) => Box::new(iter::empty()),
            OpEnum::For(op) => Box::new(iter::once(&mut op.body)),
            OpEnum::While(op) => Box::new(iter::once(&mut op.body)),
            OpEnum::If(op) => Box::new(iter::once(&mut op.then_body).chain(iter::once(&mut op.else_body))),
            OpEnum::Add(_) => Box::new(iter::empty()),
            OpEnum::Module(op) => Box::new(iter::once(&mut op.body)),
            OpEnum::Return(_) => Box::new(iter::empty()),
            OpEnum::AssignOp(_) => Box::new(iter::empty())
        }
    }

    fn recursive_uses_impl(&self, uses: &mut Vec<Value>) {
        for use_ in self.uses() {
            uses.push(use_.clone());
        }
        for block in self.blocks() {
            for op in block {
                op.borrow().recursive_uses_impl(uses);
            }
        }
    }

    fn recursive_uses(&self) -> Vec<Value> {
        let mut uses = Vec::new();
        self.recursive_uses_impl(&mut uses);
        uses
    }
}


struct ConstOp {
    value: ConstValue,
    result: Value,
}

struct ForOp {
    start: Value,
    end: Value,
    step: Value,
    ind_var: Value,
    body: Vec<Op>,
}

struct WhileOp {
    cond: Value,
    body: Vec<Op>,
}

struct IfOp {
    cond: Value,
    then_body: Vec<Op>,
    else_body: Vec<Op>,
}

struct AddOp {
    lhs: Value,
    rhs: Value,
    result: Value,
}

struct ModuleOp {
    body: Vec<Op>,
}

struct ReturnOp {
    values: Vec<Value>,
}

struct AssignOp {
    lhs: Value,
    rhs: Value,
}

struct UsageAnalysis {
    users: HashMap<Value, Vec<Op>>,
    parent: HashMap<Value, Op>,
    parent_op: HashMap<Op, Op>,
}

impl UsageAnalysis {
    fn on_operation(op: Op) -> UsageAnalysis {
        let mut users = HashMap::new();
        let mut parent = HashMap::new();
        let mut parent_op = HashMap::new();
        for use_ in op.borrow().recursive_uses() {
            users.entry(use_).or_insert_with(|| Vec::new()).push(op.clone());
        }

        for values in op.borrow().values() {
            parent.insert(values.clone(), op.clone());
        }

        UsageAnalysis {
            users,
            parent,
            parent_op
        }
    }
}

fn for_every_backward(op: &Op, mut f: impl FnMut(&Op)) {
    for block in op.borrow().blocks() {
        for op in block.iter().rev() {
            for_every_backward(op, &mut f);
        }
    }
    f(op);
}



struct DCE {
    analysis: UsageAnalysis,
    alive: HashSet<Value>
}

impl DCE {
    fn run(&mut self, op: &Op) {
        let mut visited = HashSet::new();
        let mut to_visit = vec![];
        for_every_backward(op, |op| {
            match &*op.borrow() {
                OpEnum::Return(rop) => {
                    for value in rop.values.iter().rev() {
                        self.alive.insert(value.clone());
                        visited.insert(op.clone());
                        to_visit.push(self.analysis.parent_op.get(op).unwrap().clone());
                    }
                    
                },
                _ => {}
            }
        });

        while to_visit.len() > 0 {
            let op = to_visit.pop().unwrap();
            if visited.contains(&op) {
                continue;
            }
            match *op.borrow() {
                OpEnum::Const(_) => {},
                OpEnum::For(_) => {},
                OpEnum::While(_) => {},
                OpEnum::If(_) => {},
                OpEnum::Add(_) => {},
                OpEnum::Module(_) => {},
                OpEnum::Return(_) => {},
                OpEnum::AssignOp(_) => {}
            }
            visited.insert(op.clone());
            for value in op.borrow().values() {
                self.alive.insert(value.clone());
            }
            for block in op.borrow().blocks() {
                for op in block.iter().rev() {
                    to_visit.push(op.clone());
                }
            }
        }
    }
}