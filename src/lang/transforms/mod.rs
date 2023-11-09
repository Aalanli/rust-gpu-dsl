
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Index;
use anyhow::{Result, Error};
use crate::lang::ir::{self, Value, Op, Block, OpId};

use super::ir::{IRModule, BlockId};

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



pub struct CorrectnessState {
    defined_values: HashSet<Value>,
}


struct PrinterState {
    indent_size: usize,
    indent: usize,
    buf: String,
    hints: HashMap<String, usize>,
    names: HashMap<Value, String>,
}

impl PrinterState {
    fn new(indent_size: usize) -> Self {
        PrinterState {
            indent_size,
            indent: 0,
            buf: String::new(),
            hints: HashMap::new(),
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

    fn get_unique_name(&mut self, val: &Value, hint: Option<impl Into<String>>) -> String {
        if let Some(name) = self.names.get(val) {
            return name.clone();
        }
        let hint: Option<String> = hint.map(|x| x.into());
        let mut name = hint.unwrap_or("".to_string());
        
        if let Some(i) = self.hints.get_mut(&name) {
            name.push_str(&format!("{}", i));
            *i += 1;
        } else {
            self.hints.insert(name.clone(), 1);
            name.push_str("0");
        }
        self.names.insert(val.clone(), name.clone());
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
                let unique_name = state.get_unique_name(x, Some(name));
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
            let op_name = if op.internal_as_any().type_id() == TypeId::of::<ir::ElementWiseOp>() {
                let op = op.internal_as_any().downcast_ref::<ir::ElementWiseOp>().unwrap();
                match &op.name {
                    ir::ElementwiseFnOption::Intrinsic(op) => {
                        format!("Elementwise{:?}", op)
                    }
                    ir::ElementwiseFnOption::Extern(name, _eltype) => {
                        name.clone()
                    }
                }
            } else if op.internal_as_any().type_id() == TypeId::of::<ir::ReduceOp>() {
                let op = op.internal_as_any().downcast_ref::<ir::ReduceOp>().unwrap();
                format!("Reduce{:?}", op.op)
            } else {
                op.name().to_string()
            };

            state.push_token(&op_name);
            state.push_token("(");
            let arg_list = op.inputs().into_iter().map(|x| {
                let name = x.name().unwrap_or_else(|| { op.name().to_string().to_lowercase() });
                let unique_name = state.get_unique_name(x, Some(name));
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
                        let unique_name = state.get_unique_name(x, Some(name));
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

struct RewriterState {
    remap_value: HashMap<Value, Value>,
    ops: Vec<Vec<Op>>,
}

impl RewriterState {
    pub fn new() -> Self {
        RewriterState {
            remap_value: HashMap::new(),
            ops: vec![vec![]],
        }
    }

    pub fn build_block(&mut self, args: Vec<Value>, f: impl FnOnce(&mut Self, &Vec<Value>)) -> Block {
        self.ops.push(vec![]);
        f(self, &args);
        let ops = self.ops.pop().unwrap();
        Block::new(args, ops)
    }

    pub fn push_op(&mut self, op: Op) {
        self.ops.last_mut().unwrap().push(op);
    }

    pub fn remap(&self, val: &Value) -> Option<&Value> {
        self.remap_value.get(val)
    }

    pub fn rewrite_op(&mut self, op: &Op) -> Op {
        let new_inputs = op.inputs().iter().map(|x| {
            self.remap(x).unwrap_or(x).clone()
        }).collect::<Vec<_>>();
        let new_outputs = op.outputs().iter().map(|x| {
            self.remap(x).unwrap_or(x).clone()
        }).collect::<Vec<_>>();
        let new_blocks = op.blocks().iter().map(|x| {
            self.rewrite_block(x)
        }).collect::<Vec<_>>();
        if new_inputs.iter().zip(op.inputs().iter()).all(|(a, b)| a == *b) 
        && new_outputs.iter().zip(op.outputs().iter()).all(|(a, b)| a == *b) 
        && new_blocks.iter().collect::<Vec<_>>() == op.blocks() {
            op.clone()
        } else {
            let op_enum = match op.get_inner() {
                ir::OpEnum::ProgramID(_) => ir::OpEnum::ProgramID(ir::ProgramIDOp { output: new_outputs[0].clone() }),
                ir::OpEnum::Load(_) => ir::OpEnum::Load(ir::LoadOp { ptr: new_inputs[0].clone(), mask: new_inputs.get(1).cloned(), value: new_inputs.get(2).cloned(), output: new_outputs[0].clone() }),
                ir::OpEnum::Store(_) => ir::OpEnum::Store(ir::StoreOp::build(&new_inputs[0], &new_inputs[1], new_inputs.get(2)).unwrap()),
                ir::OpEnum::Expand(op) => ir::OpEnum::Expand(ir::ExpandOp::build(&new_inputs[0], op.dim as i32).unwrap()),
                ir::OpEnum::Broadcast(_) => ir::OpEnum::Broadcast(ir::BroadcastOp { input: new_inputs[0].clone(), output: new_outputs[0].clone() }),
                ir::OpEnum::Reduce(_) => ir::OpEnum::Reduce(todo!()),
                ir::OpEnum::ElementWise(_) => ir::OpEnum::ElementWise(todo!()),
                ir::OpEnum::Dot(_) => ir::OpEnum::Dot(todo!()),
                ir::OpEnum::Full(_) => ir::OpEnum::Full(todo!()),
                ir::OpEnum::Constant(_) => ir::OpEnum::Constant(todo!()),
                ir::OpEnum::Arange(_) => ir::OpEnum::Arange(todo!()),
                ir::OpEnum::For(_) => ir::OpEnum::For(todo!()),
                ir::OpEnum::SCFFOR(_) => ir::OpEnum::SCFFOR(todo!()),
                ir::OpEnum::FunctionOp(_) => ir::OpEnum::FunctionOp(todo!()),
                ir::OpEnum::Assign(_) => ir::OpEnum::Assign(todo!()),
            };
            Op::new(op_enum, op.location().clone())
        }
    }

    pub fn rewrite_block(&mut self, block: &Block) -> Block {
        let new_args = block.args.iter().map(|x| {
            self.remap(x).unwrap_or(x).clone()
        }).collect::<Vec<_>>();
        let new_ops = block.body.iter().map(|x| {
            self.rewrite_op(x)
        }).collect::<Vec<_>>();
        if new_args == block.args && new_ops == block.body {
            block.clone()
        } else {
            Block::new(new_args, new_ops)
        }
    }
}


fn convert_to_pure_for() -> impl Visitor<RewriterState> {
    CachedVisitor::new()
    .add_typed_visitor(|v, s, op: &ir::ForOp| {

        Ok(())
    })
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


fn encoding_visitor() -> impl Visitor<EncodingState> {
    CachedVisitor::new()
    .add_typed_visitor(|state: &mut EncodingState, _, op: &ir::ArangeOp| {
        state.encodings.insert(op.output.clone(), Encoding::Shared(SharedEncoding {
            shape: vec![1],
        }));
        Ok(())
    })
}

/// bottom-up, inside-out
fn walk_postorder(irm: &mut IRModule, mut f: impl FnMut(&mut IRModule, &ir::OpId)) {
    fn walk_postorder_helper(
        irm: &mut IRModule, 
        f: &mut impl FnMut(&mut IRModule, &ir::OpId), 
        op: &ir::OpId
    ) {
        let mut key = irm.op_blocks_back(op);
        while let Some(bk) = &key {
            let mut ok = irm.block_back(bk);
            while let Some(ok_) = &ok {
                walk_postorder_helper(irm, f, ok_);
                ok = irm.op_prev(ok_);
            }
            key = irm.block_prev(bk);
        } 
        f(irm, op);
    }
    let Some(last_root) = irm.root_ops().last() else { return; };
    walk_postorder_helper(irm, &mut f, &last_root);
}


fn for_each_op(irm: &mut IRModule, block: &BlockId, mut f: impl FnMut(&mut IRModule, &ir::OpId)) {
    let mut op_key = irm.block_front(block);
    while let Some(op) = &op_key {
        f(irm, op);
        op_key = irm.op_next(op);
    }
}

fn canoncalize_for(irm: &mut IRModule) {
    walk_postorder(irm, |irm, op_id| {
        if *irm.op_ty(op_id) == ir::For {
            // get nonlocals
            let mut nonlocals = vec![];
            let mut locals = HashSet::new();
            let block = irm.op_blocks_front(op_id).unwrap();
            // insert the induction var
            locals.insert(irm.block_args(&block)[0].clone());

            for_each_op(irm, &block, |irm, op| {
                for arg in irm.op_operands(op) {
                    if !locals.contains(arg) {
                        nonlocals.push(arg.clone());
                        locals.insert(arg.clone());
                    }
                }
                for ret in irm.op_returns(op) {
                    locals.insert(ret.clone());
                }
            });

            // remap nonlocals to block args
            let mut block_args = irm.drain_block_args(&block);
            block_args.extend(
                nonlocals.iter().map(|x| {
                    let ty = irm.value_type(x).clone();
                    let arg = irm.build_value(ty);
                    arg
                })
            );
            irm.set_block_args(&block, block_args);

            for_each_op(irm, &block, |irm, op| {
                for i in 0..irm.op_operands(op).len() {
                    let arg = irm.op_operands(op)[i].clone();
                    if let Some(j) = nonlocals.iter().position(|x| x == &arg) {
                        irm.set_op_operand(op, j, nonlocals[j].clone());                        
                    }
                }
            });

            // create yield op
            let yield_op = irm.build_op(ir::SCFYield, irm.block_args(&block).to_vec(), vec![], vec![]);
            irm.block_push_back(&block, yield_op);

            // create new return values
            let returns = nonlocals.iter().map(|x| {
                let ty = irm.value_type(x).clone();
                irm.build_value(ty)}
            ).collect::<Vec<_>>();

            irm.set_op_returns(op_id, returns);

            // create new assigns to map nonlocals to returns in the parent block
            let parent_block = irm.op_parent_block(op_id).unwrap();
            for i in 0..nonlocals.len() {
                let assign_op = irm.build_op(ir::Assign, vec![nonlocals[i].clone(), irm.op_returns(op_id)[i].clone()], vec![], vec![]);
                irm.block_push_back(&parent_block, assign_op);
            }
            
        }
    });
}
