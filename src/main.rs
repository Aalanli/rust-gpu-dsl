use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::cell::{RefCell, Ref};

thread_local! {
    static BLOCK_STACK: RefCell<Vec<Block>> = RefCell::new(vec![]);
}

static OBJECT_COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_id() -> u64 {
    OBJECT_COUNTER.fetch_add(1, Ordering::SeqCst)
}

fn insert_block(block: Block) {
    BLOCK_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.push(block);
    })
}

fn pop_block() -> Block {
    BLOCK_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.pop().unwrap()
    })
}

fn insert_op(op: Op) {
    BLOCK_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let block = stack.last_mut().unwrap();
        block.ops.push(op);
    })
}

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



#[derive(Debug)]
struct Value(Rc<ValueInternal>);
impl Value {
    fn is(&self, other: &Value) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    fn set_name(&mut self, name: &str) {
        *self.0.name.borrow_mut() = Some(name.to_string());
    }

    fn add_user(&mut self, op: &Op, operands: &Operand) {
        self.0.users.borrow_mut().push((Op(op.0.clone()), Operand(operands.0.clone())));
    }

    fn users(&self) -> Ref<Vec<(Op, Operand)>> {
        self.0.users.borrow()
    }

    fn get_name(&self) -> Option<String> {
        self.0.name.borrow().clone()
    }

    fn make_operand(&self) -> Operand {
        Operand(Rc::downgrade(&self.0))
    }

}
#[derive(Debug)]
struct ValueInternal {
    name: RefCell<Option<String>>,
    _type: Type,
    users: RefCell<Vec<(Op, Operand)>>
}


#[derive(Debug, Clone)]
struct Operand(Weak<ValueInternal>);


#[derive(Debug)]
struct Op(Rc<OpInternal>);
#[derive(Debug)]
enum OpInternal {
    Constant(ConstantOp),
    If(IfOp),
    While(WhileOp),
    For(ForOp),
    Module(ModuleOp)
}
#[derive(Debug)]
struct ModuleOp {
    blocks: Vec<Block>,
}
#[derive(Debug)]
struct ConstantOp {
    value: Constant,
    result: Value
}
#[derive(Debug)]
struct IfOp {
    cond: Operand,
    then_block: Block,
    else_block: Block
}
#[derive(Debug)]
struct WhileOp {
    cond: Operand,
    body: Block
}
#[derive(Debug)]
struct ForOp {
    init: Operand,
    fin: Operand,
    step: Operand,
    body: Block
}

#[derive(Debug)]
struct AssignOp {
    lhs: Operand,
    rhs: Operand
}

#[derive(Debug)]
struct Block {
    values: Vec<Value>,
    ops: Vec<Op>
}

#[derive(Debug)]
struct Type(Rc<TypeInternal>);
#[derive(Debug)]
enum TypeInternal {
    Val(Dtype),
    Ptr(Dtype)
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

#[derive(Debug)]
enum Dtype {
    Int,
    Float,
    Bool
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

fn constant<T: Into<Constant>>(val: T) -> Value {
    let constant: Constant = val.into();
    let value = ValueInternal {
        name: RefCell::new(None),
        _type: constant.get_type(),
        users: RefCell::new(vec![])
    };
    let value = Value(Rc::new(value));
    let constant_op = ConstantOp {
        value: constant,
        result: Value(value.0.clone())
    };
    insert_op(Op(Rc::new(OpInternal::Constant(constant_op))));
    value
}

fn for_op(init: &mut Value, fin: &mut Value, step: &mut Value, body: impl Fn(&Value)) {
    let induction_var = Value(Rc::new(ValueInternal {
        name: RefCell::new(Some("i".to_string())),
        _type: Type(Rc::new(TypeInternal::Val(Dtype::Int))),
        users: RefCell::new(vec![])
    }));

    let body_block = Block {
        values: vec![Value(induction_var.0.clone())],
        ops: vec![]
    };
    insert_block(body_block);

    body(&induction_var);
    let init_oper = init.make_operand();
    let fin_oper = fin.make_operand();
    let step_oper = step.make_operand();

    let for_op = ForOp {
        init: init_oper.clone(),
        fin: fin_oper.clone(),
        step: step_oper.clone(),
        body: pop_block()
    };
    let op = Op(Rc::new(OpInternal::For(for_op)));
    init.add_user(&op, &init_oper);
    fin.add_user(&op, &fin_oper);
    step.add_user(&op, &step_oper);
    insert_op(op);
}

fn if_op(cond: &mut Value, then_block: impl Fn(), else_block: impl Fn()) {
    let cond_oper = cond.make_operand();

    let then_block_ = Block {
        values: vec![],
        ops: vec![]
    };
    insert_block(then_block_);

    then_block();

    let then_block_ = pop_block();

    let else_block_ = Block {
        values: vec![],
        ops: vec![]
    };
    insert_block(else_block_);

    else_block();

    let else_block_ = pop_block();

    let if_op = IfOp {
        cond: cond_oper.clone(),
        then_block: then_block_,
        else_block: else_block_ 
    };
    let op = Op(Rc::new(OpInternal::If(if_op)));
    cond.add_user(&op, &cond_oper);
    // cond.add_user(Op(Rc::new(OpInternal::If(if_op.clone()))));
    insert_op(op);
}

fn while_op(cond: &Value, body: impl Fn()) {
    let cond_oper = cond.make_operand();

    let body_block = Block {
        values: vec![],
        ops: vec![]
    };
    insert_block(body_block);

    body();

    let while_op = WhileOp {
        cond: cond_oper,
        body: pop_block()
    };
    insert_op(Op(Rc::new(OpInternal::While(while_op))));
}


fn main() {
    let a = constant(1);
    let b = constant(10);

}