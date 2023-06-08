use std::rc::Rc;

use std::ops::{
    Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign,
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
    Neg, Not
};

use std::cmp::{Eq, Ord};

/// Constant := Naturals
/// Var := Name | Constant
/// Expr := Constant 
///       | Var
///       | Call(Function, Expr*)
/// Stmt := Let(Var, Expr)
///       | LetMut(Var, Expr)
///       | For(Var*, )
/// Function := FunctionDef(Name, Var*, *Stmt)
struct Function {
    name: String,
    output_type: Types,
    args: Vec<Var>,
    body: Vec<StmtNode>
}

enum StmtNode {
    Let(Var, Expr),
    LetMut(Var, Expr),
    Assign(Var, Expr),
    AddAssign(Var, Expr),
    SubAssign(Var, Expr),
    MulAssign(Var, Expr),
    DivAssign(Var, Expr),
    BitOrAssign(Var, Expr),
    BitAndAssign(Var, Expr),
    BitXorAssign(Var, Expr),    

    For(For),
    While(While),
    If(If),
    Break,
    Return(Expr),
    Function(Function),
}

type Stmt = Rc<StmtNode>;

struct For {
    vars: Vec<Var>,
    inits: Vec<Stmt>,
    iters: Vec<Stmt>,
    cond: Expr,
    body: Vec<Stmt>
}

struct While {
    cond: Expr,
    body: Vec<Stmt>
}

struct If {
    cond: Expr,
    body: Vec<Stmt>,
    orelse: Vec<Stmt>
}


enum ExprNode {
    Constant(Constant),
    Var(Var),
    Binary(BinaryOps),
    Unary(UnaryOps),
    Call(Function, Vec<Expr>),
}



struct Var {
    name: String,
    the_type: Types,
    is_mut: bool,
    val: Option<Expr>
}


enum Types {
    Ptr(ValueType),
    ValueType(ValueType)
}

enum ValueType {
    TensorType(TensorType),
    ScalarType(ScalarTypes)
}

struct TensorType {
    datatype: Rc<Types>,
    rank: u64,
    shape: Vec<Expr>,
    stride: Vec<Expr>
}

enum ScalarTypes {
    Int64,
    Int32,
    Int16,
    Int8,
    UInt64,
    UInt32,
    UInt16,
    UInt8,
    F32,
    F64
}

type Expr = Rc<ExprNode>;

enum BinaryOps {
    Sub(Expr, Expr),
    Add(Expr, Expr),
    Mul(Expr, Expr),
    Div(Expr, Expr),
    Exp(Expr, Expr),

    Mod(Expr, Expr),
    
    LT(Expr, Expr),
    LEQ(Expr, Expr),
    LShift(Expr, Expr),
    RShift(Expr, Expr),
    
    Or(Expr, Expr),
    And(Expr, Expr),
    BitwiseOr(Expr, Expr),
    BitwiseAnd(Expr, Expr),
    Xor(Expr, Expr),

    Index(Expr),
    IndexMut(Expr)
}

enum UnaryOps {
    Neg(Expr),
    Not(Expr),
}

enum Constant {
    Float(FloatConstant),
    Int(IntConstant),
    Bool(bool)
}

enum FloatConstant {
    F32(f32),
    F64(f64),
}

enum IntConstant {
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8)
}

fn main() {
    println!("Hello, world!");
}
