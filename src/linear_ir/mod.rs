// Each Value is the result of one Operation. Operations can have Operands that either Read or Consume Values,
// after a Value is consumed it can no longer be used. A block specifies a sequence of Operations that are
// executed in order. All Operands must have their values specified in a block argument, while each Operation
// must specify the Values that it reads and consumes

use std::rc::Rc;
use anyhow::{Result, Error};

/// Type has value semantics, and are equal if they have the same data
#[derive(Clone, Eq, PartialEq)]
pub struct Type;

/// Since Values are unique to each operation, they are equal if they have the same memory address
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
        Value(Rc::new(ValueImpl {
            type_of,
            passed_from: None,
        }))
    }

    /// Signifies that this Value is a block argument with passing semantics, from the Operand of a parent Operation
    pub fn passed_from(operand: &Operand) -> Value {
        Value(Rc::new(ValueImpl {
            type_of: operand.type_of().clone(),
            passed_from: Some(operand.clone()),
        }))
    }

    pub fn type_of(&self) -> &Type {
        &self.0.type_of
    }

    pub fn id_of(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    fn clone(&self) -> Self {
        Value(self.0.clone())
    }
}

struct ValueImpl {
    type_of: Type,
    passed_from: Option<Operand>,
}

#[derive(PartialEq, Eq)]
pub enum Operand {
    Consuming(ConsumingOperand),
    Reading(ReadingOperand),
}

impl Operand {
    pub fn type_of(&self) -> &Type {
        match self {
            Operand::Consuming(op) => op.type_of(),
            Operand::Reading(op) => op.type_of(),
        }
    }

    pub fn source(&self) -> &Value {
        match self {
            Operand::Consuming(op) => op.source(),
            Operand::Reading(op) => op.source(),
        }
    }

    fn clone(&self) -> Self {
        match self {
            Operand::Consuming(op) => Operand::Consuming(op.clone()),
            Operand::Reading(op) => Operand::Reading(op.clone()),
        }
    }
}

#[derive(PartialEq, Eq)]
pub struct ConsumingOperand {
    parent_of: Value
}

impl ConsumingOperand {
    pub fn from_value(value: Value) -> Self {
        ConsumingOperand { parent_of: value }
    }

    pub fn source(&self) -> &Value {
        &self.parent_of
    }

    pub fn type_of(&self) -> &Type {
        self.parent_of.type_of()
    }

    fn clone(&self) -> Self {
        ConsumingOperand { parent_of: self.parent_of.clone() }
    }
}
#[derive(PartialEq, Eq)]
pub struct ReadingOperand {
    parent_of: Value
}

impl ReadingOperand {
    pub fn from_value(value: &Value) -> Self {
        ReadingOperand { parent_of: value.clone() }
    }

    pub fn source(&self) -> &Value {
        &self.parent_of
    }

    pub fn type_of(&self) -> &Type {
        self.parent_of.type_of()
    }

    pub fn clone(&self) -> Self {
        ReadingOperand { parent_of: self.parent_of.clone() }
    }
}

pub struct Block {
    args: Vec<Value>,
    operations: Vec<Op>,
}

impl Block {
    pub fn new(args: Vec<Value>) -> Self {
        Block { args, operations: vec![] }
    }

    pub fn args(&self) -> &[Value] {
        &self.args
    }

    pub fn push_op(&mut self, op: Op) {
        self.operations.push(op);
    }

    pub fn verify(&self) -> Result<()> {
        for op in &self.operations {
            op.verify()?; // TODO: add context
            // verify that all operands have a block argument as parent
            for oper in op.reading_operands().iter().map(|x| x.source()).chain(op.consuming_operands().iter().map(|x| x.source())) {
                if !self.args.contains(oper) {
                    return Err(Error::msg("Operand source not in block args"));
                }
            }
            // TODO: check that a block terminator exists
        }
        Ok(())
    } 
}

pub struct OperationBase {
    blocks: Vec<Block>,
    consuming_operands: Vec<ConsumingOperand>,
    return_values: Vec<Value>,
    reading_operands: Vec<ReadingOperand>,
}

impl OperationBase {
    pub fn new<'a>(
        consumes: impl Iterator<Item = Value>,
        returns: impl Iterator<Item = Type>,
        reads: impl Iterator<Item = &'a Value>,
    ) -> Self {
        let consuming_operands = consumes.map(ConsumingOperand::from_value).collect();
        let reading_operands = reads.map(ReadingOperand::from_value).collect();
        let return_values = returns.map(Value::new).collect();
        OperationBase {
            blocks: Vec::new(),
            consuming_operands,
            return_values,
            reading_operands,
        }
    }

    pub fn verify(&self) -> Result<()> {
        for block in &self.blocks {
            block.verify()?;
            // verify that all sources of block arguments are in the list of arguments
            // block terminator is checked by the specific operation
            for arg in block.args() {
                if let Some(operand) = arg.0.passed_from.as_ref() {
                    match operand {
                        Operand::Consuming(op) => {
                            if !self.consuming_operands.contains(op) {
                                return Err(Error::msg("Block argument source not in consuming operands"));
                            }
                        }
                        Operand::Reading(op) => {
                            if !self.reading_operands.contains(op) {
                                return Err(Error::msg("Block argument source not in reading operands"));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}


pub struct Op;
impl Op {
    fn blocks(&self) -> &[Block] {
        todo!()
    }

    fn consuming_operands(&self) -> &[ConsumingOperand] {
        todo!()
    }

    fn reading_operands(&self) -> &[ReadingOperand] {
        todo!()
    }

    fn verify(&self) -> Result<()> {
        todo!()
    }
}

pub struct UsageAnalysis {

}
