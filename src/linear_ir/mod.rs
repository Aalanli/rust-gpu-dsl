// Each Value is the result of one Operation. Operations must list all Values that it uses, so that all
// values in a block that is not produced from an Operation in that block are block arguments.

use anyhow::{Error, Result};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ID(usize);

/// Type has value semantics, and are equal if they have the same data
#[derive(Clone, Eq, PartialEq)]
pub enum Type {
    Scalar(Dtype),
    Tensor(TensorType), // technically, scalars can be represented as a tensor with rank 0, should that be done?
}

#[derive(Clone, Eq, PartialEq)]
pub enum Dtype {
    I32,
    F32,
    Bool,
    I32Ptr,
    F32Ptr,
    BoolPtr,
}

#[derive(Clone, Eq, PartialEq)]
pub struct TensorType {
    dtype: Dtype,
    shape: Vec<usize>,
    // todo: Encoding
}

/// Since Values are unique to each operation, they are equal if they have the same memory address
#[derive(Clone)]
pub struct Value(Rc<ValueImpl>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Value {
    /// Signfies that this is the result of an Operation, or a block argument introduced by the semantics of an Operation
    /// eg, the induction variable of the for_op
    pub fn new(type_of: Type) -> Self {
        Value(Rc::new(ValueImpl { type_of }))
    }

    pub fn type_of(&self) -> &Type {
        &self.0.type_of
    }

    pub fn id_of(&self) -> ID {
        ID(Rc::as_ptr(&self.0) as usize)
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}

struct ValueImpl {
    type_of: Type,
}

pub struct Block {
    args: Vec<Value>,
    operations: Vec<Op>,
}

impl Block {
    pub fn new(args: Vec<Value>) -> Self {
        Block {
            args,
            operations: vec![],
        }
    }

    pub fn args(&self) -> &[Value] {
        &self.args
    }

    pub fn push_op(&mut self, op: Op) {
        self.operations.push(op);
    }

    pub fn verify(&self) -> Result<()> {
        let mut produced_values: HashSet<&Value> = HashSet::from_iter(self.args.iter());

        for op in &self.operations {
            op.verify()?; // TODO: add context
                          // verify that all operands have a block argument as parent
            produced_values.extend(op.produces());
        }

        for op in &self.operations {
            for use_ in op.uses() {
                if !produced_values.contains(use_) {
                    return Err(Error::msg("Use of Value in block is not the result of a block argument or an Operation in the block"));
                }
            }
        }
        Ok(())
    }
}

pub struct OperationBase {
    pub blocks: Vec<Block>,
    pub uses: Vec<Value>,
    pub return_values: Vec<Value>,
}

impl OperationBase {
    pub fn new<'a>(
        uses: impl Iterator<Item = Value>,
        returns: impl Iterator<Item = Type>, // do this so produced Values are guaranteed to be unique
    ) -> Self {
        let return_values = returns.map(Value::new).collect();
        OperationBase {
            blocks: Vec::new(),
            uses: uses.collect(),
            return_values,
        }
    }

    // Additional Ops wrap around OperationBase and add additional verification logic
    pub fn verify(&self) -> Result<()> {
        for block in &self.blocks {
            block.verify()?;
        }
        let mut contains: HashSet<&Value> = HashSet::new();
        for value in &self.return_values {
            if contains.contains(value) {
                return Err(Error::msg("Value produced by multiple operations"));
            }
            contains.insert(value);
        }

        for block in &self.blocks {
            for value in block.args() {
                if contains.contains(value) {
                    return Err(Error::msg("Value is a block argument, but already exists"));
                }
            }
            for op in &block.operations {
                Self::verify_unique_producer(op, &mut contains)?;
            }
        }
        Ok(())
    }

    fn verify_unique_producer<'a>(op: &'a Op, contains: &mut HashSet<&'a Value>) -> Result<()> {
        for value in op.produces() {
            if contains.contains(value) {
                return Err(Error::msg("Value produced by multiple operations"));
            }
            contains.insert(value);
        }
        for block in op.blocks() {
            for val in block.args() {
                if contains.contains(val) {
                    return Err(Error::msg("Value is a block argument, but already exists"));
                }
                contains.insert(val);
            }
            for op in &block.operations {
                Self::verify_unique_producer(op, contains)?;
            }
        }
        Ok(())
    }
}

pub enum ArgumentSemantic {
    Move,
    Ref, // immutable reference
         // we do not allow mutable references, as they are equivalent to move and return
}

pub struct Op;
impl Op {
    pub fn blocks(&self) -> &[Block] {
        todo!()
    }

    pub fn produces(&self) -> &[Value] {
        todo!()
    }

    pub fn uses(&self) -> &[Value] {
        todo!()
    }

    pub fn verify(&self) -> Result<()> {
        todo!()
    }

    pub fn id(&self) -> ID {
        todo!()
    }
}

pub struct ForOp {
    base: OperationBase,
}

impl ForOp {
    pub fn verify(&self) -> Result<()> {
        self.base.verify()?;
        // operands = [start: int, end: int, step: int, carried...]
        // return_values = [carried...]
        // block_args = [induction_var: int, carried...]
        // block_yield = [carried...]
        if self.base.blocks.len() != 1 {
            return Err(Error::msg("ForOp must have exactly one block"));
        }
        if self.base.uses.len() > 3 {
            return Err(Error::msg("ForOp must have more than 3 operands"));
        }
        for op in self.base.uses[0..3].iter() {
            if op.type_of() != &Type::Scalar(Dtype::I32) {
                return Err(Error::msg(
                    "First 3 operands of ForOp must be $start, $end, $step, of type i32",
                ));
            }
        }
        if self.base.uses.len() - 3 != self.base.return_values.len() {
            return Err(Error::msg("ForOp must have exactly 3 more operands than return values, as the rest of values are carried"));
        }
        for (user, ret) in self.base.uses[3..]
            .iter()
            .zip(self.base.return_values.iter())
        {
            if user.type_of() != ret.type_of() {
                return Err(Error::msg(
                    "ForOp return values must have the same type as the corresponding operand",
                ));
            }
        }
        // TODO: add block check
        Ok(())
    }
}

pub struct UsageAnalysis<'a> {
    value_source: HashMap<&'a Value, &'a Op>,
    value_uses: HashMap<&'a Value, Vec<&'a Op>>,
}

impl<'a> UsageAnalysis<'a> {
    pub fn run(&mut self, op: &'a Op) -> Result<()> {
        for use_ in op.uses() {
            self.value_uses.entry(use_).or_default().push(op);
        }
        for src in op.produces() {
            if self.value_source.contains_key(src) {
                return Err(Error::msg("Value produced by multiple operations"));
            }
            self.value_source.insert(src, op);
        }
        for block in op.blocks() {
            for op in &block.operations {
                self.run(op)?;
            }
        }

        Ok(())
    }

    // If return is None, then the value is a block argument
    pub fn value_source(&self, value: &'a Value) -> Option<&'a Op> {
        self.value_source.get(value).copied()
    }

    pub fn value_uses(&self, value: &'a Value) -> &[&'a Op] {
        self.value_uses.get(value).map(|v| &v[..]).unwrap_or(&[])
    }
}

pub trait DataFlowAnalysis {
    fn join(&self, result_uses: &[bool], is_pure: bool) -> Vec<bool>;
}

impl DataFlowAnalysis for Op {
    /// pessimistic join
    fn join(&self, result_uses: &[bool], is_pure: bool) -> Vec<bool> {
        if !is_pure || result_uses.iter().any(|x| *x) {
            vec![true; self.uses().len()]
        } else {
            vec![false; self.uses().len()]
        }
    }
}
