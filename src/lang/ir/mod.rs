use anyhow::{Error, Result};
use core::panic;
use std::cell::RefCell;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Range, RangeBounds};
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use crate::utils::{VecDoubleList, VecListKey};
use super::Location;

/// Type has value semantics, and are equal if they have the same data
/// Everying is a tensor type, scalars are represented by tensors with rank 0
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Type {
    eltype: ElType,
    shape: Vec<usize>, 
    // todo: Encoding
}

impl Type {
    pub fn scalar(eltype: ElType) -> Self {
        Type {
            eltype,
            shape: vec![],
        }
    }

    pub fn i32_scalar() -> Self {
        Type::scalar(ElType::Val(Dtype::I32))
    }

    pub fn f32_scalar() -> Self {
        Type::scalar(ElType::Val(Dtype::F32))
    }

    pub fn i32_ptr() -> Self {
        Type::scalar(ElType::Ptr(Dtype::I32))
    }

    pub fn f32_ptr() -> Self {
        Type::scalar(ElType::Ptr(Dtype::F32))
    }

    pub fn is_bool(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::I1) => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::F32) => true,
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match &self.eltype {
            ElType::Val(Dtype::I32) => true,
            _ => false,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn is_ptr(&self) -> bool {
        match &self.eltype {
            ElType::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn tensor(eltype: ElType, shape: &[usize]) -> Self {
        Type {
            eltype,
            shape: shape.into(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn to_pointer(&self) -> Self {
        let dtype = if let ElType::Val(dtype) = &self.eltype {
            ElType::Ptr(dtype.clone())
        } else {
            self.eltype.clone()
        };
        Type {
            eltype: dtype,
            shape: self.shape.clone(),
        }
    }

    pub fn to_value(&self) -> Self {
        let dtype = if let ElType::Ptr(dtype) = &self.eltype {
            ElType::Val(dtype.clone())
        } else {
            self.eltype.clone()
        };
        Type {
            eltype: dtype,
            shape: self.shape.clone(),
        }
    }

    pub fn eltype(&self) -> &ElType {
        &self.eltype
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ElType {
    Ptr(Dtype),
    Val(Dtype),
}

impl ElType {
    pub fn bits(&self) -> usize {
        match self {
            ElType::Ptr(_) => 64,
            ElType::Val(dtype) => match dtype {
                Dtype::I1 => 8,
                Dtype::I32 => 32,
                Dtype::F32 => 32,
            },
        }
    }

    pub fn inner_dtype(&self) -> &Dtype {
        match self {
            ElType::Ptr(dtype) => dtype,
            ElType::Val(dtype) => dtype,
        }
    }

    pub fn i32() -> Self {
        ElType::Val(Dtype::I32)
    }

    pub fn f32() -> Self {
        ElType::Val(Dtype::F32)
    }

    pub fn bool() -> Self {
        ElType::Val(Dtype::I1)
    }

    pub fn ptr_i32() -> Self {
        ElType::Ptr(Dtype::I32)
    }

    pub fn ptr_f32() -> Self {
        ElType::Ptr(Dtype::F32)
    }

    pub fn ptr_bool() -> Self {
        ElType::Ptr(Dtype::I1)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Dtype {
    I1,
    I32,
    F32,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Constant {
    I32(i32),
    F32(f32),
    Bool(bool),
}

impl Constant {
    pub fn dtype(&self) -> Dtype {
        match self {
            Constant::I32(_) => Dtype::I32,
            Constant::F32(_) => Dtype::F32,
            Constant::Bool(_) => Dtype::I1,
        }
    }

    pub fn type_of(&self) -> Type {
        Type::scalar(ElType::Val(self.dtype()))
    }
}

impl From<i32> for Constant {
    fn from(val: i32) -> Self {
        Constant::I32(val)
    }
}

impl From<f32> for Constant {
    fn from(val: f32) -> Self {
        Constant::F32(val)
    }
}

impl From<bool> for Constant {
    fn from(val: bool) -> Self {
        Constant::Bool(val)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct ID(usize);

impl ID {
    fn inc(&mut self) -> Self { 
        let id = self.0;
        self.0 += 1;
        ID(id)
    }
    fn unique() -> Self { 
        let mut rng = rand::thread_rng();
        ID(rng.gen())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpId {
    generation: ID,
    module_id: ID,
    key: VecListKey,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockId {
    generation: ID,
    module_id: ID,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueId {
    generation: ID,
    module_id: ID
}

#[derive(Debug)]
pub struct OpProto {
    gen: ID,
    module_id: ID,
}

struct OpImpl_ {
    op: Operations,
    args: Vec<ValueId>,
    returns: Vec<ValueId>,
    blocks: VecDoubleList<ID>,
}

#[derive(Debug)]
pub struct BlockProto {
    gen: ID,
    module_id: ID
}

struct BlockImpl_ {
    args: Vec<ValueId>,
    ops: VecDoubleList<ID>,
}

#[derive(Debug)]
pub struct ValueProto {
    gen: ID,
    module_id: ID,
}

struct ValueImpl_ {
    ty: Type
}

pub struct IRModule {
    globals: VecDoubleList<ID>,
    values: HashMap<ID, ValueImpl_>,
    ops: HashMap<ID, OpImpl_>,
    blocks: HashMap<ID, BlockImpl_>,

    value_source: HashMap<ID, ID>,
    op_parent: HashMap<ID, Option<ID>>,
    block_parent: HashMap<ID, ID>,

    gen: ID,
    module_id: ID,
}

impl IRModule {
    pub fn is_valid_op_id(&self, op: &OpId) -> bool { self.ops.contains_key(&op.generation) && op.module_id == self.module_id }
    fn check_op(&self, op: &OpId) { 
        if !self.is_valid_op_id(op) {
            panic!("Op {:?} is not a valid id of this IRModule {}", op, self.module_id.0);
        }
    }
    fn check_op_proto(&self, proto: &OpProto) { 
        if proto.module_id != self.module_id || !self.ops.contains_key(&proto.gen) {
            panic!("OpProto {:?} is not a valid id of this IRModule {}", proto, self.module_id.0);
        }
    }
    fn check_value(&self, value: &ValueId) {
        if value.module_id != self.module_id || !self.values.contains_key(&value.generation) {
            panic!("ValueId {:?} is not a valid id of this IRModule {}", value, self.module_id.0);
        }
    }
    fn check_value_proto(&self, proto: &ValueProto) {
        if proto.module_id != self.module_id || !self.values.contains_key(&proto.gen) {
            panic!("ValueProto {:?} is not a valid id of this IRModule {}", proto, self.module_id.0);
        }
    }

    pub fn is_root(&self, op: &OpId) -> bool { 
        self.check_op(op);
        self.globals.is_valid_key(&op.key)
    }

    pub fn root_ops(&self) -> impl Iterator<Item = OpId> + '_ { 
        self.globals.iter().zip(self.globals.iter_key()).map(|(id, k)| {
            OpId { generation: *id, module_id: self.module_id, key: k }
        })
    }

    pub fn build_op(
        &mut self, 
        op_ty: Operations,
        args: impl IntoIterator<Item = ValueId>, 
        returns: impl IntoIterator<Item = ValueProto>, 
        blocks: impl IntoIterator<Item = BlockProto>
    ) -> OpProto {
        
        let gen = self.gen.inc();
        let args: Vec<_> = args.into_iter().collect();
        let returns = returns.into_iter().map(|x: ValueProto| {
            self.check_value_proto(&x);
            ValueId { generation: gen, module_id: self.module_id }
        }).collect();
        let blocks = blocks.into_iter().map(|x: BlockProto| {
            if self.module_id != x.module_id {
                panic!("BlockProto {:?} is not in this module", x);
            }
            if !self.blocks.contains_key(&gen) {
                panic!("BlockProto {:?} is not constructed in this module", x);
            }
            x.gen
        }).collect();
        let op_impl = OpImpl_{
            op: op_ty,
            args,
            returns,
            blocks
        };
        self.ops.insert(gen, op_impl);
        OpProto { gen, module_id: self.module_id }
    }

    pub fn build_block(
        &mut self,
        args: impl IntoIterator<Item = ValueProto>,
        ops: impl IntoIterator<Item = OpProto>) -> BlockProto {
        
        let args = args.into_iter().map(|x: ValueProto| {
            self.check_value_proto(&x);
            ValueId { generation: x.gen, module_id: self.module_id }
        }).collect();
        let ops = ops.into_iter().map(|x| {
            self.check_op_proto(&x);
            x.gen
        }).collect();
        let block_impl = BlockImpl_{
            args,
            ops
        };
        let gen = self.gen.inc();
        self.blocks.insert(gen, block_impl);

        BlockProto { gen, module_id: self.module_id }
    }
    
    pub fn build_value(&mut self, ty: Type) -> ValueProto {todo!()}
    // pub fn build_value(&self, ty: Type) -> ValueProto { 
    //     ValueProto {
    //         ty,
    //         id: None,
    //         module_id: self.module_id,
    //     }
    // }
    pub fn value_type(&mut self, value: &ValueId) -> &Type { 
        self.check_value(value);
        &self.values.get(&value.generation).unwrap().ty
    }

    pub fn op_ty(&self, op: &OpId) -> &Operations { 
        if let Some(x) = self.ops.get(&op.generation) {
            &x.op
        } else {
            panic!("Op {:?} not found", op)
        }
    }
    pub fn set_op_ty(&mut self, op: &OpId, op_ty: Operations) { 
        if let Some(x) = self.ops.get_mut(&op.generation) {
            x.op = op_ty;
        } else {
            panic!("Op {:?} not found", op)
        }
    }
    pub fn set_op_operands(&mut self, op: &OpId, values: impl IntoIterator<Item = ValueId>) { todo!() }
    pub fn set_op_operand(&mut self, op: &OpId, idx: usize, value: ValueId) { todo!() }
    pub fn set_op_returns(&mut self, op: &OpId, values: impl IntoIterator<Item = ValueProto>) -> Vec<ValueProto> { 
        self.check_op(op);
        let mut new_values: Vec<_> = values.into_iter().map(|x: ValueProto| {
            self.check_value_proto(&x);
            ValueId { generation: x.gen, module_id: self.module_id }
        }).collect();
        for i in new_values.iter() {
            self.value_source.insert(i.generation.clone(), op.generation.clone());
        }
        let op_impl = self.ops.get_mut(&op.generation).unwrap();

        std::mem::swap(&mut new_values, &mut op_impl.returns);
        let old_values: Vec<_> = new_values.into_iter().map(|x| {
            ValueProto { gen: x.generation, module_id: self.module_id }
        }).collect();

        for i in old_values.iter() {
            if let None = self.value_source.remove(&i.gen) {
                panic!("Value {:?} is not in use", i);
            }
        }
        old_values
    }
    pub fn op_operands(&self, op: &OpId) -> &[ValueId] { todo!() }
    pub fn op_returns(&self, op: &OpId) -> &[ValueId] { todo!() }
    pub fn drain_op_returns(&mut self, op: &OpId) -> Vec<ValueProto> { todo!() }

    pub fn op_blocks_front(&self, op: &OpId) -> Option<BlockId> { todo!() }
    pub fn op_blocks_back(&self, op: &OpId) -> Option<BlockId> { todo!() }
    pub fn op_parent_block(&self, op: &OpId) -> Option<BlockId> { todo!() }
    pub fn op_next(&self, op: &OpId) -> Option<OpId> { 
        self.check_op(op);
        let parent = self.op_parent.get(&op.generation).unwrap();
        if let Some(parent) = parent {
            let block = self.blocks.get(&parent).unwrap();
            let key = block.ops.next(&op.key)?;
            let next_gen = block.ops.get(&key).unwrap();
            Some(OpId { generation: *next_gen, module_id: self.module_id, key })
        } else {
            let next_key = self.globals.next(&op.key)?;
            let next = self.globals.get(&next_key).unwrap();
            Some(OpId { generation: *next, module_id: self.module_id, key: next_key })
        }
    }
    pub fn op_prev(&self, op: &OpId) -> Option<OpId> { todo!() }
    pub fn op_insert_before(&mut self, op: &OpId, proto: OpProto) -> OpId { 
        self.check_op(op);
        self.check_op_proto(&proto);
        let parent = *self.op_parent.get(&op.generation).unwrap();
        let new_key = if let Some(parent) = parent {
            let block = self.blocks.get_mut(&parent).unwrap();
            block.ops.insert_before(&op.key, proto.gen);
            self.op_parent.insert(proto.gen, Some(parent));
            block.ops.prev(&op.key).unwrap()
        } else {
            self.globals.insert_before(&op.key, proto.gen);
            self.op_parent.insert(proto.gen, None);
            self.globals.prev(&op.key).unwrap()
        };

        OpId { generation: proto.gen, module_id: self.module_id, key: new_key }
    }
    pub fn op_insert_after(&mut self, op: &OpId, proto: OpProto) -> OpId { todo!() }
    pub fn op_remove(&mut self, op: &OpId) -> OpProto {
        self.check_op(op);
        let parent = self.op_parent.remove(&op.generation).unwrap();
        if let Some(parent) = parent {
            self.blocks.get_mut(&parent).unwrap().ops.remove(&op.key);
        } else {
            let rm_key = self.globals.find(|x| x == &op.generation);
            if let Some(rm_key) = rm_key {
                self.globals.remove(&rm_key);
            } else {
                panic!("Op {:?} not found", op);
            }
        }
        OpProto { gen: op.generation, module_id: self.module_id }
    }

    pub fn block_args(&self, block: &BlockId) -> &[ValueId] { todo!() }
    pub fn set_block_args(&self, block: &BlockId, args: impl IntoIterator<Item = ValueProto>) -> Vec<ValueProto> { todo!() }
    pub fn drain_block_args(&mut self, block: &BlockId) -> Vec<ValueProto> { todo!() }

    pub fn block_front(&self, block: &BlockId) -> Option<OpId> { todo!() }
    pub fn block_back(&self, block: &BlockId) -> Option<OpId> { todo!() }
    pub fn block_push_front(&mut self, block: &BlockId, op: OpProto) -> OpId { todo!() }
    pub fn block_push_back(&mut self, block: &BlockId, op: OpProto) -> OpId { todo!() }
    pub fn block_parent_op(&self, block: &BlockId) -> Option<OpId> { todo!() }
    pub fn block_next(&self, block: &BlockId) -> Option<BlockId> { todo!() }
    pub fn block_prev(&self, block: &BlockId) -> Option<BlockId> { todo!() }
    pub fn block_insert_before(&mut self, block: &BlockId, proto: BlockProto) -> BlockId { todo!() }
    pub fn block_insert_after(&mut self, block: &BlockId, proto: BlockProto) -> BlockId { todo!() }
    pub fn block_remove(&mut self, block: &BlockId) -> BlockProto { todo!() }
}

use rand::Rng;



#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operations {
    Function,

    ProgramId,
    Load,
    Store,

    Const,
    Arange,
    ExpandDim,
    Broadcast,

    Reduce(IntrinsicElementwise),
    ElementWise(IntrinsicElementwise),
    Dot,

    For,
    Assign,
    SCFFor,
    SCFYield
}

pub use Operations::*;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicElementwise {
    Add, Sub, Mul, Div, Neg, Rem,
    Pow, Exp, Log, Sqrt,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or, Xor, Not,
    LogicalAnd, LogicalOr, // short circuiting
    Shr, Shl,
    Ceil, Floor, Round,
    Max, Min,
    Where,
}

impl IntrinsicElementwise {
    pub fn num_operands(&self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Rem | Self::Pow => 2,
            Self::Neg | Self::Not => 1,
            Self::Exp | Self::Log | Self::Sqrt => 1,
            Self::Eq | Self::Ne | Self::Lt | Self::Gt | Self::Le | Self::Ge => 2,
            Self::And | Self::Or | Self::Xor => 2,
            Self::LogicalAnd | Self::LogicalOr => 2,
            Self::Shr | Self::Shl => 2,
            Self::Ceil | Self::Floor | Self::Round => 1,
            Self::Max | Self::Min => 2,
            Self::Where => 3,
        }
    }

    pub fn verify_type(&self, vals: &[Type]) -> Result<()> {
        if self.num_operands() != vals.len() {
            return Err(Error::msg(format!(
                "Intrinsic {:?} requires {} operands, got {}",
                self,
                self.num_operands(),
                vals.len()
            )));
        }
        let a = &vals[0];
        if !vals.iter().all(|x| x.shape() == a.shape()) {
            return Err(Error::msg(format!(
                "Intrinsic {:?} requires all operands to have the same shape, got {:?}",
                self, vals.iter().collect::<Vec<_>>()
            )));
        }

        if self.num_operands() == 2 {
            let b = &vals[1];
            let all_same = a == b;
            match self {
                Self::Add | Self::Sub => {
                    if !((a.is_float() && b.is_float()) || (a.is_int() && b.is_int()) || (a.is_ptr() && b.is_int()) || (a.is_int() && b.is_ptr())) {
                        return Err(Error::msg(format!(
                            "+ requires either floating + floating, int + int, int + ptr or ptr + int, got {:?}",
                            vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Mul | Self::Div | Self::Max | Self::Min => {
                    if !all_same && !(a.is_float() || a.is_int()) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and either floating or integral, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Rem => {
                    if !all_same && !a.is_int() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and integral, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Pow => {
                    if !all_same && !a.is_float() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and floating, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Eq | Self::Ne | Self::Lt | Self::Gt | Self::Le | Self::Ge => {
                    if !all_same {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::And | Self::Or | Self::Xor => {
                    if !(all_same && (a.is_int() || a.is_bool())) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and either integral or boolean, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::LogicalAnd | Self::LogicalOr => {
                    if !(all_same && a.is_bool()) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have the same element type and boolean, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Shl | Self::Shr => {
                    if !(b.is_int() && (a.is_int() || a.is_ptr() || a.is_bool())) {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires first operand to be either integral boolean or pointer and the second to be integral, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                _ => {}
            }
        } else if self.num_operands() == 3 {
            if let Self::Where = self {
                let b = &vals[1];
                let c = &vals[2];
                if !(a.is_bool() && b == c) {
                    return Err(Error::msg(format!(
                        "Intrinsic {:?} requires first operand to be boolean and the other two to have the same element type, got {:?}",
                        self, vals.iter().map(|x| x).collect::<Vec<_>>()
                    )));
                }
            }
        } else {
            match self {
                Self::Neg => {
                    if !a.is_float() && !a.is_int() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have either floating or integral element type, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                }
                Self::Not => {
                    if !a.is_bool() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have boolean element type, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                },
                Self::Exp | Self::Log | Self::Sqrt | Self::Ceil | Self::Floor | Self::Round => {
                    if !a.is_float() {
                        return Err(Error::msg(format!(
                            "Intrinsic {:?} requires all operands to have floating element type, got {:?}",
                            self, vals.iter().map(|x| x).collect::<Vec<_>>()
                        )));
                    }
                },
                _ => {}
            }
        }
        Ok(())
    }

    fn binary_upcast<'a>(a: &'a Type, b: &'a Type) -> &'a Type {
        if a.is_ptr() {
            a
        } else if b.is_ptr() {
            return b;
        } else if a.is_float() {
            return a;
        } else if b.is_float() {
            return b;
        } else if a.is_int() {
            return a;
        } else if b.is_int() {
            return b;
        } else {
            return a;
        }
    }

    pub fn return_type<'a>(&self, vals: &'a [Type]) -> &'a Type {
        if self.num_operands() == 2 {
            return Self::binary_upcast(&vals[0], &vals[1]);
        } else if self.num_operands() == 3 {
            if let Self::Where = self {
                return &vals[1];
            }
            unreachable!();
        } else if self.num_operands() == 1 {
            return &vals[0];
        }
        unreachable!()
    }
}


mod test {
    #[derive(Clone, Copy)]
    struct GC<'a> {
        _marker: std::marker::PhantomData<&'a Ref<'a, ()>>,
    }

    #[derive(Copy)]
    struct Ref<'a, T> {
        gc: &'a GC<'a>,
        id: usize,
        _marker: std::marker::PhantomData<&'a T>,
    }

    impl<T> Clone for Ref<'_, T> {
        fn clone(&self) -> Self {
            Ref {
                gc: self.gc,
                id: self.id,
                _marker: std::marker::PhantomData,
            }
        }
    }

    struct MyStruct<'a> {
        item: Option<Ref<'a, MyStruct<'a>>>,
    }

    impl<'a> GC<'a> {
        fn new() -> Self {
            todo!()
        }

        fn get_mut<'b: 'a, T>(&'a self, id: Ref<'b, T>, f: impl FnOnce(&mut T)) {
            todo!()
        }

        fn make<T: 'a>(&'a self, item: T) -> Ref<'a, T> {
            todo!()
        }

        fn clear(&mut self) { todo!() }
    }

    struct MyStruct2<'a, 'b> {
        item: Option<Ref<'a, MyStruct2<'a, 'b>>>,
        i: &'b i32,
    }

    fn test() {
        let mut gc = GC::new();

        let a = gc.make(MyStruct { item: None });
        let b = gc.make(MyStruct { item: None });

        gc.get_mut(a, |x| {
            x.item = Some(b);
        });

        gc.clear();

        let i = 1;
        let a = gc.make(MyStruct2 { item: None, i: &i });
        {
            let j = 2;
            let b = gc.make(MyStruct2 { item: None, i: &j });
            gc.get_mut(a.clone(), |x: &mut MyStruct2| {
                x.item = Some(b);
            });
        }

        gc.get_mut(a, |x| {
            let b = x.item.as_ref().unwrap().clone();
            gc.get_mut(b, |y| {
                println!("{}", y.i);
            });
        })
    }

    #[derive(Clone, Copy, Debug)]
    struct MyVec {
    }

    impl MyVec {
        fn new() -> Self {
            todo!()
        }

        fn push<T>(&mut self, item: T) {
            todo!()
        }
    }

    fn test2() {
        let mut a = MyVec::new();
        {
            let i = 1;
            a.push(&i);
        }
        println!("{:?}", a);
    }

    // struct T <'a> {
    //     a: &'a i32
    // }

    // fn foo<T: 'static>(a: T) {}
    // fn bar() {
    //     let a = 1;
    //     foo(T { a: &a });

    // }

}

/*
mod test2 {
    use std::marker::PhantomData;

    struct GC<'a> {
        _m: PhantomData<&'a i32>
    }

    trait Trace {}

    impl<T: Copy> Trace for T {}

    impl<'a> GC<'a> {
        fn new() -> GC<'a> {
            GC { _m: PhantomData }
        }

        fn alloc<T: 'a + Trace>(&'a mut self, a: T) -> &'a T { todo!() }

        fn free(&mut self) {}
    }

    struct GC2<'a, T> {
        _m: PhantomData<&'a T>
    }

    impl<'a, T> GC2<'a, T> {
        fn new() -> GC2<'a, T> {
            GC2 { _m: PhantomData }
        }

        fn alloc(&mut self, a: &'a T) { todo!() }

        fn free(&mut self) {}
    }

    fn test() {
        let mut gc = GC::new();
        {
            let a = 1;
            gc.alloc(&a);
            // gc.free();

        }

        // gc.alloc(&a);
        gc.free();

        let mut gc2 = GC2::new();

        {
            let a = 1;
            gc2.alloc(&a);
        }
        gc2.free();


        let mut v = Vec::new();
        {
            let a = 23;
            v.push(&a);
        }
        v.clear();
    }
}
*/

mod test3 {
    struct GC {}
    
    #[derive(Clone)]
    struct RefToken {}

    #[derive(Clone)]
    struct Ref<'a> {
        gc: &'a GC,
        id: usize,
    }

    impl GC {
        fn alloc<T>(&mut self, x: T) -> RefToken {
            todo!()
        }
    }

    struct Query {}
    impl Query {
        fn get(&self, x: RefToken) -> Ref<'_> {
            todo!()
        }
    }

    struct MyStruct<'a> {
        item: Option<Ref<'a>>,
    }

    fn test() {
        let mut gc = GC {};
        let query = Query {};
        let a = gc.alloc(MyStruct { item: None });
        let b = gc.alloc(MyStruct { item: None });
        let a = gc.alloc(MyStruct {
            item: Some(query.get(a)),
        });
        let b = gc.alloc(MyStruct {
            item: Some(query.get(b)),
        });

        {
            let p = 12;
            let token = gc.alloc(&p);
        }
    }
}