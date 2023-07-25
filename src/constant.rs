#[allow(unused_imports, dead_code)]
use std::borrow::BorrowMut;
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

use std::any::Any;
use std::cmp::{Eq, Ord};
use std::sync::{Arc, Mutex};
/// Constant := Naturals
/// Var := Name | Constant
/// Expr := Constant
///       | Var
///       | Call(Function, Expr*)
/// Stmt := Let(Var, Expr)
///       | LetMut(Var, Expr)
///       | For(Var*, )
/// Function := FunctionDef(Name, Var*, *Stmt)
///

// use once_cell::sync::Lazy;
// static ID_COUNTER: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(0));

// thread_local! {
//     static BLOCK_STACK: RefCell<Vec<Block>> = RefCell::new(vec![]);
// }

struct Block {
    name: Option<String>,
    values: Vec<Value>,
    ops: Vec<Ops>,
}

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

struct ConstantOp {
    loc: Location,
    constant: Constant,
    value: Value,
}

#[derive(Debug)]
enum Constant {
    Int(i64, IntegralTypes),
    Float(f64, FloatingTypes),
}

impl Constant {
    fn _type(&self) -> Types {
        match self {
            Constant::Int(_, i) => Types::ValueType(ScalarTypes::Int(*i)),
            Constant::Float(_, f) => Types::Floating(*f),
        }
    }
}

macro_rules! into_constant {
    ($const_type:tt, $enum_type:ident, $($repr:tt)*) => {
        impl From<$const_type> for Constant {
            fn from(i: $const_type) -> Self {
                Constant::$enum_type(i.try_into().unwrap(), $($repr)*)
            }
        }       
    };
}

into_constant!(i64,   Int, IntegralTypes::Int64);
into_constant!(i32,   Int, IntegralTypes::Int32);
into_constant!(i16,   Int, IntegralTypes::Int16);
into_constant!(i8,    Int, IntegralTypes::Int8);
into_constant!(bool,  Int, IntegralTypes::Int1);

into_constant!(u64,   Int, IntegralTypes::UInt64);
into_constant!(u32,   Int, IntegralTypes::UInt32);
into_constant!(u16,   Int, IntegralTypes::UInt16);
into_constant!(u8,    Int, IntegralTypes::UInt8);

into_constant!(f64, Float, FloatingTypes::F64);
into_constant!(f32, Float, FloatingTypes::F32);


// fn constant<T>(val: T) -> Value
// where T: Into<Constant> {
//     let loc: Location = std::panic::Location::caller().clone().into();

//     BLOCK_STACK.with(|x| {
//         let stack = &mut *x.borrow_mut();
//         if stack.len() == 0 {
//             stack.push(Block {
//                 name: None,
//                 values: vec![],
//                 ops: vec![],
//             });
//         }
//         let block = stack.last_mut().unwrap();
//         let constant: Constant = val.into();
        
//         let mut value = ValueBuilder::new()._type(constant.);
//         let op = ConstantOp {
//             loc,
//             constant: val.into(),
//             value: Value::new(),
//         };
    
//     });
//     todo!()
// }

struct ValueBuilder {
    name: Option<String>,
    parent_op: Weak<OpsInternal>,
    consumers: Vec<Weak<OpsInternal>>,
    _type: Option<Types>,
}

impl ValueBuilder {
    fn new() -> ValueBuilder {
        ValueBuilder {
            name: None,
            parent_op: Weak::new(),
            consumers: vec![],
            _type: None,
        }
    }

    fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }


    fn parent_op(mut self, op: &Ops) -> Self {
        self.parent_op = Rc::downgrade(op);
        self
    }

    fn _type(mut self, _type: Types) -> Self {
        self._type = Some(_type);
        self
    }

    fn add_consumer(mut self, consumer: &Ops) -> Self {
        self.consumers.push(Rc::downgrade(consumer));
        self
    }

    fn build(self) -> Value {
        let value_internal = ValueInternal {
            name: self.name,
            op: self.parent_op,
            consumers: self.consumers,
            _type: self._type.unwrap(),
        };
        Value::new(value_internal)
    }
}

type Value = Rc<ValueInternal>;
#[derive(Debug)]
struct ValueInternal {
    name: Option<String>,
    op: Weak<OpsInternal>,
    consumers: Vec<Weak<OpsInternal>>,
    _type: Types,
}


struct Operand {
    value: Weak<ValueInternal>,
}


type Ops = Rc<OpsInternal>;
#[derive(Debug)]
enum OpsInternal {
    For(For),
    While(While),
    Break,
    If(If),
    Call(Call),
    Return(Return),
    Function(Function),
    Constant(Constant),
}

impl OpsInternal {
    fn impure(&self) -> bool {
        todo!()
    }
    fn args(&self) -> Option<&[Value]> {
        match self {
            OpsInternal::For(_) => todo!(),
            OpsInternal::While(_) => todo!(),
            OpsInternal::Break => todo!(),
            OpsInternal::If(_) => todo!(),
            OpsInternal::Call(call) => Some(&call.values),
            OpsInternal::Return(ret) => Some(&ret.values),
            OpsInternal::Function(_) => None,
            OpsInternal::Constant(_) => None,
        }
    }
    fn ret(&self) -> Option<&[Value]> {
        todo!()
    }
    fn args_mut(&mut self) -> &mut Vec<Value> {
        todo!()
    }
    fn ret_mut(&mut self) -> &mut Vec<Value> {
        todo!()
    }
}

// impl Drop for Ops {
//     fn drop(&mut self) {
//         let self_ptr = Rc::as_ptr(self);
//         for arg in self.args_mut() {
//             for i in 0..arg.consumers().len() {
//                 let ptr = Rc::as_ptr(&arg.consumers_mut()[i]);
//                 if self_ptr == ptr {
//                     let i = arg.consumers_mut().remove(i);
//                     std::mem::forget(i);
//                     break;
//                 }
//             }
//         }
//         for ret in self.ret_mut() {
//             for i in 0..ret.parents().len() {
//                 let ptr = ret.parents_mut()[i].0.as_ptr();
//                 if self_ptr == ptr {
//                     let i = ret.parents_mut().remove(i);
//                     std::mem::forget(i);
//                     break;
//                 }
//             }
//         }
//     }
// }

#[derive(Debug)]
struct Call {
    values: Vec<Value>,
    function: Function,
}

#[derive(Debug)]
struct Function {
    name: String,
    res: Vec<Types>,
    args: Vec<Types>,
    body: Vec<Ops>,
}

#[derive(Debug)]
struct For {}

#[derive(Debug)]
struct While {}

#[derive(Debug)]
struct If {}


#[derive(Debug)]
struct Return {
    values: Vec<Value>,
}

/// Types
#[derive(Debug)]
enum Types {
    Ptr(ValueType),
    ValueType(ValueType),
}

#[derive(Debug)]
enum ValueType {
    TensorType(TensorType),
    ScalarType(ScalarTypes),
}

#[derive(Debug)]
struct TensorType {
    datatype: Rc<Types>,
    rank: u64,
    shape: Vec<Value>,
    stride: Vec<Value>,
}

#[derive(Debug)]
enum ScalarTypes {
    Int(IntegralTypes),
    Float(FloatingTypes),
}

#[derive(Debug)]
enum IntegralTypes {
    Int64,
    Int32,
    Int16,
    Int8,
    Int1,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
}

#[derive(Debug)]
enum FloatingTypes {
    F16,
    F32,
    F64,
}

#[derive(Clone, Debug)]
struct A(Rc<RefCell<Vec<A>>>);


#[track_caller]
fn test() {
    let caller = std::panic::Location::caller();
    let line_num = caller.line();
    println!("{:?}", line_num);
    println!("{:?}", caller);
}

fn main() {
    // let mut a = A(Rc::new(RefCell::new(vec![])));
    // let mut b = A(Rc::new(RefCell::new(vec![])));
    // (*a.0).borrow_mut().push(b.clone());
    // (*b.0).borrow_mut().push(a.clone());
    // println!("{:?}", Rc::strong_count(&a.0));
    // println!("{:?}", Rc::strong_count(&b.0));
    // let a = Rc::new(1);
    // println!("{:?}", Rc::downgrade(&a));
    test();
    test();
}
