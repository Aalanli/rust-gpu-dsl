
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::Index;
use anyhow::{Result, Error};
use crate::lang::ir::{self, OpId};

use super::ir::{IRModule, BlockId};


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
