# NumPy Fancy Indexing: A Comprehensive Guide for Deep Learning & ML

## What Is Fancy Indexing?

**Fancy indexing** (also called **advanced indexing**, **array indexing**, or
**indirect indexing**) is a NumPy feature that lets you select, read, or write
multiple array elements at once by passing an **array-like of indices** (a list,
a NumPy array, or a boolean mask) instead of a single integer or a slice.

```python
import numpy as np

arr = np.arange(10)          # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
indices = np.array([1, 3, 7])
arr[indices]                  # → array([1, 3, 7])  ← fancy indexing
```

Compare this with **basic indexing**, which uses single integers or slices and
always returns a *view* of the original array:

```python
arr[2]      # single integer  → 2
arr[1:4]    # slice           → array([1, 2, 3])  (view, no copy)
```

Fancy indexing **always returns a copy**, not a view. This distinction matters
for performance and mutation semantics.

---

## All the Names It Goes By

| Name | Context |
|---|---|
| **Fancy indexing** | The most common informal name in the NumPy community |
| **Advanced indexing** | The official term in the [NumPy documentation](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) |
| **Array indexing** | Used when emphasizing that the index itself is an array |
| **Indirect indexing** | Borrowed from low-level programming (pointer tables, gather/scatter) |
| **Gather / Scatter** | The GPU and framework equivalent (`torch.gather`, `tf.gather`, `jax.lax.gather`) |
| **Integer array indexing** | When the index array contains integers (as opposed to booleans) |
| **Boolean (mask) indexing** | The boolean sub-variant of advanced indexing |

> **Rule of thumb:** If the thing inside the brackets is an `ndarray`, a Python
> `list`, or a boolean array, you are doing fancy/advanced indexing.

---

## The Two Flavors of Fancy Indexing

### 1. Integer Array Indexing

You supply one or more arrays of **integer positions**.

```python
arr = np.array([10, 20, 30, 40, 50])

# 1-D: pick elements at positions 0, 3, 4
arr[[0, 3, 4]]               # → array([10, 40, 50])

# Duplicate indices are fine — each lookup is independent
arr[[2, 2, 2]]               # → array([30, 30, 30])

# Negative indices work too
arr[[-1, -2]]                # → array([50, 40])
```

#### Multi-dimensional integer array indexing

With 2-D (or higher) arrays, you provide one index array **per axis**. The
arrays are broadcast together, and each resulting element is selected
coordinate-wise:

```python
matrix = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

rows = np.array([0, 1, 2])
cols = np.array([3, 1, 2])
matrix[rows, cols]            # → array([3, 5, 10])
# Selects (0,3), (1,1), (2,2)
```

You can also mix fancy indexing with slices or scalars. When you do, NumPy
broadcasts the fancy index across the basic-indexed dimensions:

```python
matrix[[0, 2], :]            # rows 0 and 2, all columns
matrix[:, [1, 3]]            # all rows, columns 1 and 3
```

### 2. Boolean (Mask) Indexing

You supply a boolean array of the **same shape** (or broadcastable shape) as the
array being indexed. Elements where the mask is `True` are selected:

```python
arr = np.array([10, 20, 30, 40, 50])
mask = np.array([True, False, True, False, True])
arr[mask]                     # → array([10, 30, 50])
```

Boolean indexing is most commonly generated from a **condition**:

```python
arr[arr > 25]                 # → array([30, 40, 50])
```

The result is always a **1-D** array of the matched elements, regardless of the
original shape:

```python
matrix = np.arange(12).reshape(3, 4)
matrix[matrix % 3 == 0]      # → array([0, 3, 6, 9])
```

---

## Fancy Indexing for Writing (Assignment)

One of the most powerful features: you can **assign** to fancy-indexed
locations.

```python
arr = np.zeros(10)
indices = [1, 4, 7]
arr[indices] = 99.0
# arr → [0, 99, 0, 0, 99, 0, 0, 99, 0, 0]
```

This is the core mechanism behind operations like multi-hot encoding, one-hot
encoding, sparse updates, and embedding table construction.

### Caveat: Repeated Indices in Assignment

When the same index appears multiple times, only the **last** assignment wins
(NumPy doesn't accumulate):

```python
arr = np.zeros(5)
arr[[0, 0, 0]] = [1, 2, 3]
# arr[0] → 3  (last write wins, NOT 1+2+3)
```

If you need **accumulation**, use `np.add.at`:

```python
arr = np.zeros(5)
np.add.at(arr, [0, 0, 0], [1, 2, 3])
# arr[0] → 6.0  (accumulated)
```

This distinction is critical in gradient computations and scatter-add operations
in ML frameworks.

---

## How Fancy Indexing Is Used in Deep Learning and ML

### 1. Multi-Hot Encoding (Bag-of-Words Representations)

This is the pattern used in **Chapter 4** of *Deep Learning with Python* for
encoding IMDB review sequences and Reuters newswire sequences:

```python
def multi_hot_encode(sequences, num_classes):
    """Encode variable-length integer sequences as fixed-size multi-hot vectors."""
    results = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1.0   # ← fancy indexing!
    return results

# Example: a review contains word indices [5, 12, 0, 87]
# results[i][[5, 12, 0, 87]] = 1.0 sets those four positions to 1
```

Here `sequence` is a **list of integer indices** — so `results[i][sequence]` is
fancy indexing that sets all those positions to `1.0` in a single vectorized
operation. No Python loop over individual word indices.

### 2. One-Hot Encoding

A special case where each sample has exactly **one** active index:

```python
def one_hot_encode(labels, num_classes=46):
    results = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        results[i, label] = 1.0       # ← single integer index per row
    return results
```

There is also a fully vectorized version using fancy indexing on both axes
simultaneously:

```python
def one_hot_encode_vectorized(labels, num_classes):
    n = len(labels)
    results = np.zeros((n, num_classes))
    results[np.arange(n), labels] = 1.0  # ← 2-D fancy indexing, no loop
    return results
```

This is equivalent to `keras.utils.to_categorical(labels)`.

### 3. Embedding Lookups

The foundational operation of every embedding layer (`nn.Embedding` in PyTorch,
`keras.layers.Embedding` in Keras) is fancy indexing into a weight matrix:

```python
# Conceptual embedding lookup
embedding_matrix = np.random.randn(vocab_size, embedding_dim)  # (V, D)
token_ids = [42, 7, 1001, 3]                                   # sequence
embedded = embedding_matrix[token_ids]                          # (4, D)
# Each row of `embedded` is the embedding vector for that token.
```

Under the hood, frameworks implement this as a **gather** operation on GPU, but
the semantics are identical to NumPy fancy indexing.

### 4. Gathering Predictions / Selecting Class Scores

After a model outputs a probability distribution over classes, you often need
the score for the **true** class:

```python
# logits shape: (batch_size, num_classes)
logits = model.predict(x_batch)

# true_labels shape: (batch_size,)
true_labels = np.array([3, 0, 7, 2])

# Get the predicted score for the correct class in each sample:
scores = logits[np.arange(len(true_labels)), true_labels]
# → 1-D array of length batch_size
```

This pattern is central to computing **cross-entropy loss** manually and appears
in every deep learning framework's internals.

### 5. Boolean Masking for Filtering

Extremely common for data preprocessing, cleaning, and conditional operations:

```python
# Remove outliers
prices = np.array([150_000, 200_000, 9_999_999, 175_000, 50])
valid = (prices > 10_000) & (prices < 5_000_000)
clean_prices = prices[valid]   # → array([150000, 200000, 175000])

# Mask padded timesteps in sequences
mask = (token_ids != 0)        # True for real tokens, False for padding
attention_scores[~mask] = -1e9 # set padded positions to -inf before softmax
```

### 6. Selecting Specific Channels, Timesteps, or Spatial Locations

In computer vision and time-series work, fancy indexing lets you cherry-pick
along any axis:

```python
# Select specific feature maps from a conv output (batch, H, W, C)
feature_maps = conv_output[:, :, :, [3, 7, 15]]  # pick channels 3, 7, 15

# Select specific timesteps from a sequence (batch, T, features)
key_timesteps = [0, 10, 49]  # first, middle, last
selected = sequence_data[:, key_timesteps, :]     # (batch, 3, features)
```

### 7. Shuffle and Permutation

Shuffling a dataset is fancy indexing with a permuted index array:

```python
perm = np.random.permutation(len(x_train))
x_train_shuffled = x_train[perm]
y_train_shuffled = y_train[perm]
```

### 8. Scatter Updates in Frameworks (The GPU Counterpart)

Every major framework provides GPU-accelerated fancy indexing under different
names:

| NumPy | PyTorch | TensorFlow | JAX |
|---|---|---|---|
| `a[idx]` (read) | `torch.gather`, `a[idx]` | `tf.gather` | `jax.numpy` indexing, `jax.lax.gather` |
| `a[idx] = val` (write) | `scatter_`, `index_put_` | `tf.tensor_scatter_nd_update` | `a.at[idx].set(val)` |
| `np.add.at(a, idx, val)` | `scatter_add_` | `tf.tensor_scatter_nd_add` | `a.at[idx].add(val)` |

JAX's `.at[].set()` / `.at[].add()` syntax is the closest to NumPy fancy
indexing while remaining compatible with JAX's functional, immutable-array
model.

---

## Performance Notes

| Aspect | Basic indexing (slices) | Fancy indexing |
|---|---|---|
| **Returns** | View (shared memory) | Copy (new memory) |
| **Speed** | Faster (no data movement) | Slower (memory allocation + copy) |
| **GPU equivalent** | Strided access | Gather/Scatter kernels |
| **Mutation** | Mutates original via view | Mutates original (assignment), but reads are independent copies |

**Best practices:**

- Prefer **slices** when contiguous ranges suffice — they create views and are
  faster.
- Use **fancy indexing** when you need non-contiguous, arbitrary, or computed
  positions.
- On GPU, gather/scatter operations are well-optimized, so fancy indexing
  patterns translate efficiently.
- Watch out for **repeated-index assignment** — use scatter-add (or
  `np.add.at`) when accumulation is intended.

---

## Quick Reference: Fancy Indexing Cheat Sheet

```python
import numpy as np

a = np.arange(20).reshape(4, 5)
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

# --- Integer array indexing ---
a[[0, 3]]                    # rows 0 and 3             → shape (2, 5)
a[:, [1, 4]]                 # columns 1 and 4          → shape (4, 2)
a[[0, 2], [1, 3]]            # elements (0,1) and (2,3) → array([1, 13])
a[np.ix_([0, 2], [1, 3])]    # submatrix rows×cols      → shape (2, 2)

# --- Boolean mask indexing ---
a[a > 10]                    # all elements > 10        → 1-D array
a[a % 2 == 0]                # all even elements        → 1-D array

# --- Assignment ---
a[[1, 3], [0, 4]] = -1       # set (1,0) and (3,4) to -1
a[a < 0] = 0                 # reset negatives to 0

# --- Accumulation ---
b = np.zeros(5)
np.add.at(b, [0, 0, 1], 1)   # b → [2, 1, 0, 0, 0]

# --- np.ix_ for cross-product indexing ---
rows = [0, 2]
cols = [1, 3, 4]
submatrix = a[np.ix_(rows, cols)]  # shape (2, 3) — all combos of rows × cols
```

---

## Summary

Fancy indexing is one of the most important NumPy features for ML/DL
practitioners. It powers:

- **Encoding schemes** — multi-hot, one-hot, label encoding
- **Embedding lookups** — the foundation of NLP and recommendation systems
- **Loss computation** — gathering true-class scores from logit matrices
- **Data manipulation** — shuffling, filtering, selecting subsets
- **Gradient operations** — scatter-add for accumulating gradients at repeated
  indices

Every time you see `array[list_or_array_of_indices]` — whether in NumPy,
PyTorch, TensorFlow, or JAX — you're looking at fancy indexing (or its
GPU-accelerated equivalent). Mastering it unlocks fluent reading and writing of
deep learning code.
