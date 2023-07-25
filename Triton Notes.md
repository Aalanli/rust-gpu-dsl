### GPU IR Attributes

The layout is a function $\mathcal{L}: \mathbb{Z}^d \to S$, where $S$ is a set of integers which specifies the indices of threads allowed to access the data, for each input.

There are two types of layouts, shared and distributed.

**shared**
params: `vec`, `perPhase`, `maxPhase`, `order`
The swizzling information
Stored on shared memory

**Distributed**
Characterized by $\mathcal L$
Stored on Registers

> Distributed encodings have a layout function that is entirely characterized by a d-dimensional tensor L. Note that L doesn't need to have the same shape (or even the same rank) as the tensor it is encoding.

$$\mathcal{L}(A, i_d)=\{L[i_d + k_d * A_d] : k_d \in \mathbb N, i_d + k_d * A_d < L_d\}$$
Where $A_d = \text{shape}(A)[d]$, and $L_d = \text{shape}(L)[d]$

For example, for a tensor/layout pair
```
A = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]
```


Then the data of A would be distributed as follow between the 16 CUDA threads:

```
L(A) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
```

Blocked Encoding
params: `sizePerThread`, `threadsPerWarp`, `warpsPerCTA`

> An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout used to promote memory coalescing in LoadInst and StoreInst. It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

> For example, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows.

```
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
```
for
```
#triton_gpu.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
}>
```


Mma Encoding
Encoding for C, and D, the outputs expected by Tensor Cores (on registers)

Slice Encoding
Idk
"This is useful for constructing the inverse layout of an expand_dims operation during some optimization passes."

Dot Operand Encoding
A, and B operands expected by mma instruction

### TTIR Types

Float := F16 | F32 | F64
FloatTensor := TensorOf(Float)
FloatLike := Float | FloatTensor

BoolTensor := TensorOf(I1)
BoolLike := I1 | BoolTensor

Int := I1 | I8 | I16 | I32 | I64
IntTensor := TensorOf(Int)
IntLike := Int | IntTensor

Ptr := PtrOf(AnyType)
PtrTensor := TensorOf(Ptr)
PtrLike := Ptr | PtrTensor

FpIntTensor := FloatTensor | IntTensor
Tensor := FpIntTensor | PtrTensor

TensorPtr := PtrOf(Tensor)

AnyType := FloatLike | IntLike | PtrLike | TensorPtr


### TTIR Ops

Load (ptr: PtrLike | TensorPtr, mask: BoolLike, other: AnyType)
Store (ptr: PtrLike | TensorPtr, value: AnyType, mask: BoolLike)

splat (src: AnyType, result: Tensor)
expand_dims
view
broadcast
cat
trans

get_program_id
get_num_programs

dot
reduce
scan

element_wise

make_range

make_tensor_ptr

#### TTGIR Ops

convert_layout

async_wait
async_commit_group

select

extract_slice

insert_slice_async

```
%1 = triton_gpu.alloc_tensor : tensor<2x32xf32>
%2 = triton_gpu.insert_slice_async %0, %1, %index { axis = 0 } : tensor<32x!tt.ptr<f32>, #AL> -> tensor<2x32xf32, #A>

triton_gpu.async_wait { num = 0 : i32 }
```

### MLIR Passes

I think there are two stages, the first stage constructs the order of operations, and the computation graph, called the ttir dielect. While the second stage assigns concrete layouts and gpu specific operations.

After parsing the AST, the ttir is optimized as follows:

0. rewrite_tensor_pointer_pass
	- Seems to be converting i32 pointer offsets to i64 and generating mask for load and store ops
1. inliner
	- Inlines all functions within the kernel
2. triton_combine
	- Triton specific subgraph rewrite patterns
3. canonicalizer
	- standarization
4. reorder_broadcast
	- elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
	- elementwise(broadcast(a)) => broadcast(elementwise(a))
5. cse
	- Common subexpression elimination
6. licm
	- Loop invariant code motion
7. symbol_dce
	- Dead code elimination

Then the ttir is converted to ttgir using `convert_trition_to_tritongpu_pass`
Which assigns concrete layouts, starting with `make_range`, and propagating the layouts in shape manipulation ops like `expand_dims` and `reshape`, inserting convertion ops for `dot` and `transpose`, the latter requires a convertion to shared memory. 

Then we have ttgir optimization passes

1. coalesce
2. remove_layout_conversions
3. accelerate_matmul
4. remove_layout_conversions
5. optimize_dot_operands
6. pipeline
7. prefetch
8. optimize_dot_operands
9. remove_layout_conversions
10. decompose_conversions
11. reorder_instructions
12. cse
13. symbol_dce

#### Axis Info Analysis
- Run before coalesce and pipeline passes
```c++
/// The _contiguity_ information maps the `d`-th
/// dimension to the length of the shortest
/// sequence of contiguous integers along it.
/// Suppose we have an array of N elements,
/// with a contiguity value C,
/// the array can be divided into a list of
/// N/C sequences of C contiguous elements.
/// Since we have N = 2^k, C must be a power of two.
/// For example:
/// [10, 11, 12, 13, 18, 19, 20, 21]
/// [20, 21, 22, 23, 28, 29, 30, 31]
/// Would have contiguity [1, 4].
/// and
/// [12, 16, 20, 24]
/// [13, 17, 21, 25]
/// [14, 18, 22, 26]
/// [15, 19, 23, 27]
/// [18, 22, 26, 30]
/// [19, 23, 27, 31]
/// Would have contiguity [2, 1].
DimVectorT contiguity;

/// The _divisibility_ information maps the `d`-th
/// dimension to the largest power-of-two that
/// divides the first element of all groups of
// _contiguity_ values along it
/// For example:
/// [10, 11, 12, 13, 18, 19, 20, 21]
/// [20, 21, 22, 23, 28, 29, 30, 31]
//  would have divisibility [1, 2]
//  and
/// [12, 16, 20, 24]
/// [13, 17, 21, 25]
/// [14, 18, 22, 26]
/// [15, 19, 23, 27]
//  would have divisibility [4, 1]
//  On the other hand:
//  [0, 1, 2, 0, 4, 5, 6, 7]
//  would have divisibility 1 because
//  _contiguity_=1
DimVectorT divisibility;

/// The _constancy_ information maps the `d`-th
/// dimension to the length of the shortest
/// sequence of constant integer along it. This is
/// particularly useful to infer the contiguity
/// of operations (e.g., add) involving a constant.
/// Suppose we have an array of N elements,
/// with a constancy value C,
/// the array can be divided into a list of
/// N/C sequences of C elements with the same value.
/// Since we have N = 2^k, C must be a power of two.
/// For example
/// [8, 8, 8, 8, 12, 12, 12, 12]
/// [16, 16, 16, 16, 20, 20, 20, 20]
/// would have constancy [1, 4]
DimVectorT constancy;
```

#### Coalesce
Runs AxisInfo analysis on each operation, forward using `join` operation (should be called meet?). 
- Join is the elementwise gcd of contiguity, divisibility and constancy

Then coalesce gathers all the parent and child operations dependent upon the current operation, sorts into topological order, and filters by all the order that is the same as the current one.
- The order is an array of integers specifying the fastest to slowest varying dimensions. (0 is the fastest varying index)

Then calculates the maximum alignment, and thus the maximum number of elements per thread, out of all the filtered layouts. Call that `m`
Then the coalesced encoding becomes a blocked layout with size per thread `[m, 1, ...]`. Where the length is the rank and the rest of the indices is 1.

Then for all operations on tensors that has the same order and shape, we replace the encoding with the new encoding, and convert back afterwards.
	(Hopefully the conversions are removed in the remove conversion pass)

This is only calculated for the following memory (io) operations
- LoadOp
- AtomoicRMWOp
- AtomicCASOp
- InsertSliceAsyncOp
- StoreOp



```
module {
  tt.func public @kernel_0123(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
    %4 = tt.addptr %3, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %5 = tt.load %4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<2> : tensor<64xi32>
    %6 = arith.muli %1, %cst : tensor<64xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %9 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
    %10 = arith.addf %5, %9 : tensor<64xf32>
    %11 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
    %12 = tt.addptr %11, %2 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %12, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<64xf32>
    tt.return
  }
}
```


#### Final Ops, before llvm conversion
1. ConvertLayoutOp
2. DotOp
3. ElementwiseOp
4. Load/StoreOp
5. ReduceOp
6. ScanOp
7. ViewOp

## Triton - Pre MLIR

#### Passes
1. inliner
2. dce
3. peephole
4. dce
5. pipeline
6. dce
7. dissassociate
8. dce
9. align
10. axes
11. layouts
12. peephole
13. dce
14. cts
15. align
16. axes
17. layouts
18. coalesce

align
	is_constant_: Map( `Value` -> `[cst_info]` )
	max_contiguous_: Map( `Value` -> `[int]` )
	starting_multiple_: Map( `Value` -> `[int]` )

inliner
axes
pipeline
