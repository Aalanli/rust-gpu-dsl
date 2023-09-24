
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use anyhow::{Result, Error};
use crate::lang::ir::{self, Value, Op, Block};

pub trait Visitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()>;
    fn should_visit(&self, op: &Op) -> bool;
}

pub struct VisitArgs<F, S>(F, PhantomData<S>);
impl<F, S> VisitArgs<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Value) -> Result<()> {
    pub fn new(f: F) -> Self {
        VisitArgs(f, PhantomData)
    }
}

impl<F, S> Visitor<S> for VisitArgs<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Value) -> Result<()> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        for arg in op.inputs() {
            self.0(state, visitor, arg)?;
        }
        Ok(())
    }

    fn should_visit(&self, _op: &Op) -> bool { true }
}


pub struct VisitReturn<F, S>(F, PhantomData<S>);
impl<F, S> VisitReturn<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Value) -> Result<()> {
    pub fn new(f: F) -> Self {
        VisitReturn(f, PhantomData)
    }
}

impl<F, S> Visitor<S> for VisitReturn<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Value) -> Result<()> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        for arg in op.outputs() {
            self.0(state, visitor, arg)?;
        }
        Ok(())
    }

    fn should_visit(&self, _op: &Op) -> bool { true }
}


pub struct VisitBlock<F, S>(F, PhantomData<S>);
impl<F, S> VisitBlock<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Block) -> Result<()> {
    pub fn new(f: F) -> Self {
        VisitBlock(f, PhantomData)
    }
}

impl<F, S> Visitor<S> for VisitBlock<F, S>
where F: Fn(&mut S, &dyn Visitor<S>, &Block) -> Result<()> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        for arg in op.blocks() {
            self.0(state, visitor, arg)?;
        }
        Ok(())
    }

    fn should_visit(&self, _op: &Op) -> bool { true }
}


struct FnVisitor<F, T, S> {
    f: F,
    _s: std::marker::PhantomData<S>,
    _t: std::marker::PhantomData<T>,
}

impl<F, T, S> Visitor<S> for FnVisitor<F, T, S>
where F: Fn(&mut S, &dyn Visitor<S>, &T) -> Result<()>, T: 'static
{
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        if let Some(t) = op.internal_as_any().downcast_ref::<T>() {
            (self.f)(state, visitor, t)?;
        }
        Ok(())
    }

    fn should_visit(&self, op: &Op) -> bool { 
        op.internal_as_any().is::<T>()
    }
}

impl<F, S> Visitor<S> for F
where F: Fn(&mut S, &dyn Visitor<S>, &Op) -> Result<()>
{
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        self(state, visitor, op)
    }

    fn should_visit(&self, _op: &Op) -> bool { true }
}


struct CachedVisitor<S> {
    visitors: Vec<Box<dyn Visitor<S> + 'static>>,
    cache: RefCell<HashMap<TypeId, usize>>,
}

impl<S: 'static> CachedVisitor<S> {
    pub fn new() -> Self {
        CachedVisitor {
            visitors: vec![],
            cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn add_visitor<V: Visitor<S> + 'static>(mut self, visitor: V) -> Self {
        self.visitors.push(Box::new(visitor));
        self
    }

    pub fn add_typed_visitor<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) -> Result<()> + 'static {
        let f = move |state: &mut S, visitor: &dyn Visitor<S>, op: &Op| {
            if let Some(t) = op.internal_as_any().downcast_ref::<T>() {
                f(state, visitor, t)?;
            }
            Ok(())
        };
        self.visitors.push(Box::new(FnVisitor { f, _s: std::marker::PhantomData, _t: std::marker::PhantomData }));
        self
    }
}

impl<S: 'static> Visitor<S> for CachedVisitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        if let Some(idx) = self.cache.borrow().get(&op.internal_as_any().type_id()) {
            self.visitors[*idx].visit(visitor, state, op)?;
            return Ok(());
        } 
        for (idx, cur_visitor) in self.visitors.iter().enumerate() {
            if cur_visitor.should_visit(op) {
                {
                    self.cache.borrow_mut().insert(op.internal_as_any().type_id(), idx);
                }
                cur_visitor.visit(visitor, state, op)?;
                break;
            }
        }
        Ok(())
    }

    fn should_visit(&self, op: &Op) -> bool {
        self.visitors.iter().any(|v| v.should_visit(op))
    }
}

struct SequentialVisitor<S> {
    visitors: Vec<Box<dyn Visitor<S> + 'static>>,
}

impl<S: 'static> SequentialVisitor<S> {
    pub fn new() -> Self {
        SequentialVisitor {
            visitors: vec![],
        }
    }

    pub fn add_visitor<V: Visitor<S> + 'static>(mut self, visitor: V) -> Self {
        self.visitors.push(Box::new(visitor));
        self
    }

    pub fn add_typed_visitor<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) -> Result<()> + 'static {
        let f = move |state: &mut S, visitor: &dyn Visitor<S>, op: &Op| {
            if let Some(t) = op.internal_as_any().downcast_ref::<T>() {
                f(state, visitor, t)?;
            }
            Ok(())
        };
        self.visitors.push(Box::new(FnVisitor { f, _s: std::marker::PhantomData, _t: std::marker::PhantomData }));
        self
    }
}

impl <S: 'static> Visitor<S> for SequentialVisitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        for cur_visitor in self.visitors.iter() {
            cur_visitor.visit(visitor, state, op)?;
        }
        Ok(())
    }

    fn should_visit(&self, _op: &Op) -> bool { true }
}



pub enum Encoding {
    Shared(SharedEncoding),
    Local(BlockEncoding),
}

pub struct SharedEncoding {
    shape: Vec<u32>,
}
pub struct BlockEncoding {
    shape: Vec<u32>,
    thread: Vec<u32>,
    warp: Vec<u32>,
    block: Vec<u32>,
}

pub struct EncodingState {
    encodings: HashMap<Value, Encoding>,
    num_blocks: u32,
    num_warps: u32,
}


pub struct CorrectnessState {
    defined_values: HashSet<Value>,
}


struct PrinterState {
    indent_size: usize,
    indent: usize,
    buf: String,
    names: HashMap<String, usize>
}

impl PrinterState {
    fn new(indent_size: usize) -> Self {
        PrinterState {
            indent_size,
            indent: 0,
            buf: String::new(),
            names: HashMap::new(),
        }
    }

    fn push_char(&mut self, c: char) {
        self.buf.push(c);
    }

    fn push_token(&mut self, s: impl Into<String>) {
        self.buf.push_str(&s.into());
    }

    fn newline(&mut self) {
        self.push_char('\n');
        for _ in 0..self.indent {
            self.push_char(' ');
        }
    }

    fn indent(&mut self) {
        self.indent += self.indent_size;
    }

    fn unident(&mut self) {
        self.indent -= self.indent_size;
    }

    fn arg_list(&mut self, args: impl Iterator<Item = String>, max_line_size: usize) {
        let max_line_size = max_line_size + self.indent;
        let arg_list = args
            .into_iter()
            .map(|s| s.clone().into())
            .collect::<Vec<String>>();

        let overflow = arg_list.iter().any(|x| x.len() > max_line_size);
        if overflow {
            self.indent();
            self.newline();
            for arg in arg_list.iter().take(arg_list.len() - 1) {
                self.push_token(arg);
                self.push_char(',');
                self.newline();
            }
            if let Some(s) = arg_list.last() {
                self.push_token(s);
                self.newline();
            }
        } else {
            self.push_token(arg_list.join(", "));
        }
    }

    fn get_string(self) -> String { self.buf }

    fn get_unique_name(&mut self, name: impl Into<String>) -> String {
        let mut name = name.into();
        if let Some(i) = self.names.get_mut(&name) {
            name.push_str(&format!("{}", i));
            *i += 1;
        } else {
            self.names.insert(name.clone(), 1);
            name.push_str("0");
        }
        name
    }
}

fn type_repr(ty: &ir::Type) -> String {
    let mut buf = String::new();
    if ty.is_ptr() {
        buf.push_str("&");
    }
    match ty.eltype().inner_dtype() {
        ir::Dtype::I1 => { buf.push_str("bool"); }
        ir::Dtype::I32 => { buf.push_str("i32"); }
        ir::Dtype::F32 => { buf.push_str("f32"); }
    }
    if !ty.is_scalar() {
        buf.push_str(&format!("[{}]", ty.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")));
    }
    buf
}

fn printer_visitor(max_lines: usize) -> CachedVisitor<PrinterState> {
    type S = PrinterState;
    let visitor = CachedVisitor::new()
        .add_visitor(move |state: &mut S, v: &dyn Visitor<S>, op: &Op| {
            let return_list = op.outputs().into_iter().map(|x| {
                let name = x.name().unwrap_or_else(|| { op.name().to_string().to_lowercase() });
                let unique_name = state.get_unique_name(name);
                format!("%{}: {}", unique_name, type_repr(x.type_of()))
            }).collect::<Vec<_>>();
            state.newline();
            if op.outputs().len() > 0 {
                state.push_token("let ");
                if op.outputs().len() > 1 {
                    state.push_token("(");
                    state.arg_list(return_list.into_iter(), max_lines);
                    state.push_token(") = ");
                } else {
                    state.arg_list(return_list.into_iter(), max_lines);
                    state.push_token(" = ");
                }
            }

            state.push_token(op.name());
            state.push_token("(");
            let arg_list = op.inputs().into_iter().map(|x| {
                let name = x.name().unwrap_or_else(|| { op.name().to_string().to_lowercase() });
                let unique_name = state.get_unique_name(name);
                format!("%{}", unique_name)
            }).collect::<Vec<_>>();
            state.arg_list(arg_list.into_iter(), max_lines);
            state.push_token(")");
            if op.blocks().len() > 0 {
                state.push_token(" {");
                state.indent();
                
                for block in op.blocks() {
                    state.newline();
                    state.push_token("|");
                    let arg_list = block.args.iter().map(|x| {
                        let name = x.name().unwrap_or_default();
                        let unique_name = state.get_unique_name(name);
                        format!("%{}: {}", unique_name, type_repr(x.type_of()))
                    }).collect::<Vec<_>>();
                    state.arg_list(arg_list.into_iter(), max_lines);
                    state.push_token("| {");
                    state.indent();
                    for op in block.body.iter() {
                        v.visit(v, state, op)?;
                    }
                    state.unident();
                    state.newline();
                    state.push_token("}");
                }

                state.unident();
                state.newline();
                state.push_token("}");
            }
            
            Ok(())
        });
    visitor
}

pub fn print(op: &Op) -> String {
    let mut state = PrinterState::new(4);
    let visitor = printer_visitor(80);
    visitor.visit(&visitor, &mut state, &op).unwrap();
    state.get_string()
}

fn check_program_correctness_visitor() -> SequentialVisitor<CorrectnessState> {
    type S = CorrectnessState;
    SequentialVisitor::new()
        .add_visitor(|state: &mut S, _: &dyn Visitor<S>, op: &Op| {
            for arg in op.inputs() {
                if !state.defined_values.contains(arg) {
                    return Err(Error::msg(format!("Value {:?} is used before being defined", arg)));
                }
            }
            for arg in op.outputs() {
                if state.defined_values.contains(arg) {
                    return Err(Error::msg(format!("Value {:?} is defined more than once", arg)));
                }
                state.defined_values.insert(arg.clone());
            }
            Ok(())
        })
        .add_visitor(VisitBlock::new(|state: &mut S, v: &dyn Visitor<S>, block: &Block| {
            for arg in block.args.iter() {
                if state.defined_values.contains(arg) {
                    return Err(Error::msg(format!("Value {:?} is defined more than once", arg)));
                }
            }
            for op in block.body.iter() {
                v.visit(v, state, op)?;
            }
            Ok(())
        }))
}



fn encoding_visitor() -> CachedVisitor<EncodingState> {
    CachedVisitor::new()
    .add_typed_visitor(|state: &mut EncodingState, _, op: &ir::ArangeOp| {
        state.encodings.insert(op.output.clone(), Encoding::Shared(SharedEncoding {
            shape: vec![1],
        }));
        Ok(())
    })
}