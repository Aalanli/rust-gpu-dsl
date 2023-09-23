
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use anyhow::{Result, Error};
use crate::lang::{Op, Block, Value, ir};

pub trait Visitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()>;
    fn should_visit(&self, op: &Op) -> bool;
}

struct BaseVisitor<S> {
    visit_block: Option<fn(&mut S, &dyn Visitor<S>, &Block) -> Result<()>>,
    visit_value: Option<fn(&mut S, &Value) -> Result<()>>,
}

impl<S> BaseVisitor<S> {
    pub fn new() -> Self {
        BaseVisitor {
            visit_block: None,
            visit_value: None,
        }
    }

    pub fn add_visit_block(mut self, f: fn(&mut S, &dyn Visitor<S>, &Block) -> Result<()>) -> Self {
        self.visit_block = Some(f);
        self
    }

    pub fn add_visit_value(mut self, f: fn(&mut S, &Value) -> Result<()>) -> Self {
        self.visit_value = Some(f);
        self
    }
}

impl<S> Visitor<S> for BaseVisitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        if let Some(f) = self.visit_value {
            for arg in op.inputs() {
                f(state, arg)?;
            }
        }
        if let Some(f) = self.visit_block {
            for block in op.blocks() {
                f(state, visitor, block)?;
            }
        } else {
            for block in op.blocks() {
                if let Some(f) = self.visit_value {
                    for arg in block.args.iter() {
                        f(state, arg)?;
                    }
                }
                for op in block.body.iter() {
                    visitor.visit(visitor, state, op)?;
                }
            }
        }
        if let Some(f) = self.visit_value {
            for arg in op.outputs() {
                f(state, arg)?;
            }
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

    pub fn add_fn<F, T: 'static>(self, f: F) -> Self
    where F: Fn(&mut S, &T) + 'static {
        let f = move |state: &mut S, _: &dyn Visitor<S>, t: &T| f(state, t);
        self.add_fn_rec(f)
    }

    pub fn add_fn_err<F, T: 'static>(self, f: F) -> Self
    where F: Fn(&mut S, &T) -> Result<()> + 'static {
        let f = move |state: &mut S, _: &dyn Visitor<S>, t: &T| f(state, t);
        self.add_fn_err_rec(f)
    }

    pub fn add_fn_rec<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) + 'static
    {
        let f = move |state: &mut S, visitor: &dyn Visitor<S>, op: &Op| {
            if let Some(t) = op.internal_as_any().downcast_ref::<T>() {
                f(state, visitor, t);
            }
            Ok(())
        };
        self.visitors.push(Box::new(FnVisitor { f, _s: std::marker::PhantomData, _t: std::marker::PhantomData }));
        self
    }

    pub fn add_fn_err_rec<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) -> Result<()> + 'static
    {
        self.visitors.push(Box::new(FnVisitor { f, _s: std::marker::PhantomData, _t: std::marker::PhantomData }));
        self
    }
}

impl<S: 'static> Visitor<S> for CachedVisitor<S> {
    fn visit(&self, visitor: &dyn Visitor<S>, state: &mut S, op: &Op) -> Result<()> {
        if let Some(idx) = self.cache.borrow().get(&op.internal_as_any().type_id()) {
            self.visitors[*idx].visit(visitor, state, op)?;
        } else {
            for (idx, cur_visitor) in self.visitors.iter().enumerate() {
                if cur_visitor.should_visit(op) {
                    self.cache.borrow_mut().insert(op.internal_as_any().type_id(), idx);
                    cur_visitor.visit(visitor, state, op)?;
                    break;
                }
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

    pub fn add_fn<F, T: 'static>(self, f: F) -> Self
    where F: Fn(&mut S, &T) + 'static {
        let f = move |state: &mut S, _: &dyn Visitor<S>, t: &T| f(state, t);
        self.add_fn_rec(f)
    }

    pub fn add_fn_err<F, T: 'static>(self, f: F) -> Self
    where F: Fn(&mut S, &T) -> Result<()> + 'static {
        let f = move |state: &mut S, _: &dyn Visitor<S>, t: &T| f(state, t);
        self.add_fn_err_rec(f)
    }

    pub fn add_fn_rec<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) + 'static
    {
        let f = move |state: &mut S, visitor: &dyn Visitor<S>, op: &Op| {
            if let Some(t) = op.internal_as_any().downcast_ref::<T>() {
                f(state, visitor, t);
            }
            Ok(())
        };
        self.visitors.push(Box::new(FnVisitor { f, _s: std::marker::PhantomData, _t: std::marker::PhantomData }));
        self
    }

    pub fn add_fn_err_rec<F, T: 'static>(mut self, f: F) -> Self
    where F: Fn(&mut S, &dyn Visitor<S>, &T) -> Result<()> + 'static
    {
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
trait Lens<T> {
    fn view(&mut self) -> &mut T;
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
    .add_visitor(
        BaseVisitor::new().add_visit_block(|state: &mut S, _: &dyn Visitor<S>, block: &Block| {
            for arg in block.args.iter() {
                if state.defined_values.contains(arg) {
                    return Err(Error::msg(format!("Value {:?} is defined more than once", arg)));
                }
                state.defined_values.insert(arg.clone());
            }
            Ok(())
        })
    )
}

fn encoding_visitor() -> CachedVisitor<EncodingState> {
    CachedVisitor::new()
    .add_fn(|state: &mut EncodingState, op: &ir::ArangeOp| {
        state.encodings.insert(op.output.clone(), Encoding::Shared(SharedEncoding {
            shape: vec![1],
        }));
    })
}