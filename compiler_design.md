# Some Thoughts about compiler design patterns

1. Immutable visitor pattern
2. Index key pattern
3. MLIR


# 1. Immutable visitor pattern
- good for object orientated languages
- 

```rust
struct OpId(/*...*/);
struct ValueId(/*...*/);
struct BlockId(/*...*/);

struct IRModule {/*...*/}

struct OperandChain(/*...*/);
struct UseChain(/*...*/);
struct BlockChain(/*...*/);
struct OpChain(/*...*/);

impl IRModule {
    fn op_operands(&self, op: &OpId) -> &[ValueId];
    fn op_on_operand(&mut self, impl FnMut(&mut Self, ValueId));
    fn op_operand_chain(&mut self, )
}

```