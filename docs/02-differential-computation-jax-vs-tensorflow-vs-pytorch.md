# Differential Computation: JAX vs TensorFlow vs PyTorch

> **A comparative guide to the native automatic differentiation mechanisms
> in the three major deep learning frameworks — focusing on philosophy,
> API design, and practical implications.**

---

## Table of Contents

- [Differential Computation: JAX vs TensorFlow vs PyTorch](#differential-computation-jax-vs-tensorflow-vs-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [The Big Picture: Three Philosophies of Autodiff](#the-big-picture-three-philosophies-of-autodiff)
  - [1. Defining Differentiable Computations](#1-defining-differentiable-computations)
    - [TensorFlow: Record-Then-Differentiate](#tensorflow-record-then-differentiate)
    - [PyTorch: Compute-Then-Backpropagate](#pytorch-compute-then-backpropagate)
    - [JAX: Transform a Function](#jax-transform-a-function)
    - [Side-by-Side: Simple Gradient of x^2](#side-by-side-simple-gradient-of-x2)
    - [Comparison Table: Gradient Computation Mechanics](#comparison-table-gradient-computation-mechanics)
  - [2. State Management and Mutability](#2-state-management-and-mutability)
    - [TensorFlow: tf.Variable with In-Place Mutation](#tensorflow-tfvariable-with-in-place-mutation)
    - [PyTorch: Tensors with requires_grad and In-Place Mutation](#pytorch-tensors-with-requires_grad-and-in-place-mutation)
    - [JAX: Immutable Arrays and Functional Updates](#jax-immutable-arrays-and-functional-updates)
    - [Comparison Table: State Management](#comparison-table-state-management)
  - [3. Gradient Computation in Detail](#3-gradient-computation-in-detail)
    - [TensorFlow: The GradientTape Context Manager](#tensorflow-the-gradienttape-context-manager)
    - [PyTorch: The Autograd Engine](#pytorch-the-autograd-engine)
    - [JAX: Function Transformations](#jax-function-transformations)
    - [Comparison Table: Gradient Computation Features](#comparison-table-gradient-computation-features)
  - [4. Higher-Order Gradients](#4-higher-order-gradients)
    - [Comparison Table: Higher-Order Differentiation](#comparison-table-higher-order-differentiation)
  - [5. Handling Complex Loss Functions](#5-handling-complex-loss-functions)
    - [Multi-Variable Gradients](#multi-variable-gradients)
    - [Auxiliary Outputs](#auxiliary-outputs)
    - [Comparison Table: Complex Gradient Scenarios](#comparison-table-complex-gradient-scenarios)
  - [6. JIT Compilation and Performance](#6-jit-compilation-and-performance)
    - [Comparison Table: Compilation and Performance](#comparison-table-compilation-and-performance)
  - [7. End-to-End Training Loop Comparison](#7-end-to-end-training-loop-comparison)
    - [TensorFlow Training Step](#tensorflow-training-step)
    - [PyTorch Training Step](#pytorch-training-step)
    - [JAX Training Step](#jax-training-step)
    - [Comparison Table: Training Loop Anatomy](#comparison-table-training-loop-anatomy)
  - [8. The Conceptual Models Compared](#8-the-conceptual-models-compared)
    - [Forward / Backward Mode Autodiff](#forward--backward-mode-autodiff)
    - [Comparison Table: Conceptual Models](#comparison-table-conceptual-models)
  - [9. Strengths and Trade-Offs](#9-strengths-and-trade-offs)
    - [Comparison Table: Practical Strengths and Weaknesses](#comparison-table-practical-strengths-and-weaknesses)
  - [10. Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)
  - [Glossary](#glossary)
    - [Autograd](#autograd)
    - [Automatic Differentiation (Autodiff)](#automatic-differentiation-autodiff)
    - [Backpropagation (Backprop)](#backpropagation-backprop)
    - [Computation Graph](#computation-graph)
    - [Dynamic Graph](#dynamic-graph)
    - [Eager Execution](#eager-execution)
    - [GradientTape](#gradienttape)
    - [has_aux](#has_aux)
    - [Jaxpr](#jaxpr)
    - [JIT (Just-In-Time Compilation)](#jit-just-in-time-compilation)
    - [Metaprogramming](#metaprogramming)
    - [Pure Function](#pure-function)
    - [Pytree](#pytree)
    - [requires_grad](#requires_grad)
    - [Reverse-Mode Autodiff](#reverse-mode-autodiff)
    - [tf.Variable](#tfvariable)
    - [Tracing](#tracing)
    - [value_and_grad](#value_and_grad)
    - [XLA (Accelerated Linear Algebra)](#xla-accelerated-linear-algebra)

---

## Introduction

All three major deep learning frameworks — **TensorFlow**, **PyTorch**, and **JAX** —
support **automatic differentiation** (autodiff), the technique that makes gradient-based
optimization possible. However, they expose this capability through profoundly different
programming interfaces, each reflecting a distinct design philosophy:

| Framework  | Autodiff Philosophy                                                                |
| ---------- | ---------------------------------------------------------------------------------- |
| TensorFlow | **Tape-based recording** — record operations, then replay for gradients            |
| PyTorch    | **Dynamic computation graph** — build graph on-the-fly, backpropagate through it   |
| JAX        | **Function transformation** — transform a pure function into its gradient function |

This document dives deep into these three approaches, using code examples (including a
complete linear classifier) to illustrate the practical consequences of each design choice.

> **Scope:** We focus on the _native_, low-level differentiation APIs of each framework.
> High-level libraries (Keras, PyTorch Lightning, Flax, etc.) build on top of these
> primitives but are outside the scope of this document. See
> [01-deep-learning-ecosystem-mapping.md](01-deep-learning-ecosystem-mapping.md) for the
> full ecosystem view.

---

## The Big Picture: Three Philosophies of Autodiff

Before looking at code, it helps to understand the mental model behind each framework:

| Aspect                     | TensorFlow                                   | PyTorch                                                              | JAX                                                                                 |
| -------------------------- | -------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Paradigm**               | Imperative with tape recording               | Imperative with dynamic graph                                        | Functional / metaprogramming                                                        |
| **Gradient trigger**       | Explicit context manager (`GradientTape`)    | Calling `.backward()` on a loss                                      | Calling a transformed function (`jax.grad`)                                         |
| **State model**            | Mutable `tf.Variable`                        | Mutable tensors with `.grad`                                         | Immutable arrays; state threaded through functions                                  |
| **Default execution**      | Eager (with optional `@tf.function` tracing) | Eager (with optional `torch.compile`)                                | Eager (with optional `@jax.jit` tracing)                                            |
| **Who records the graph?** | The tape records while open                  | The autograd engine records every op on tensors with `requires_grad` | No graph — `jax.grad` traces the function at call time                              |
| **Graph lifetime**         | One tape = one forward pass                  | Graph exists until `.backward()` clears it                           | No persistent graph; tracing happens on each `jax.grad` call (cached by `@jax.jit`) |

The key insight: **TensorFlow and PyTorch are both imperative** — you write normal Python
code and the framework secretly builds a computation graph behind the scenes. **JAX is
functional** — you write a pure Python function and JAX _transforms_ it into a new function
that computes gradients.

---

## 1. Defining Differentiable Computations

### TensorFlow: Record-Then-Differentiate

TensorFlow uses the `tf.GradientTape` context manager. Within the `with` block, every
operation on watched tensors is recorded onto the tape. After exiting, you call
`tape.gradient(target, sources)` to compute gradients:

```python
input_var = tf.Variable(initial_value=3.0)
with tf.GradientTape() as tape:
    result = tf.square(input_var)          # recorded on the tape
gradient = tape.gradient(result, input_var) # d(result)/d(input_var) = 6.0
```

Key behaviors:

- **`tf.Variable` is automatically watched.** Constants require an explicit `tape.watch()`.
- **The tape is consumed after one `tape.gradient()` call** (unless `persistent=True`).
- The tape is a _context_ — you choose exactly which forward-pass region to record.

### PyTorch: Compute-Then-Backpropagate

PyTorch records operations on any tensor that has `requires_grad=True`. There is no
explicit tape — the autograd engine silently builds a dynamic computation graph as you
execute operations. You then call `.backward()` on a scalar result to populate `.grad`
on each leaf tensor:

```python
input_var = torch.tensor(3.0, requires_grad=True)
result = torch.square(input_var)   # graph node created silently
result.backward()                  # backpropagation through the graph
gradient = input_var.grad           # 6.0
```

Key behaviors:

- **Gradients accumulate** in `.grad` across calls — you must zero them manually.
- **The computation graph is freed after `.backward()`** (unless `retain_graph=True`).
- There is no explicit "recording region" — everything with `requires_grad=True` is tracked.

### JAX: Transform a Function

JAX takes a radically different approach: **metaprogramming**. You define a plain Python
function, then use `jax.grad()` or `jax.value_and_grad()` to obtain a _new_ function that
computes gradients. There is no tape, no graph stored on tensors, and no mutable state:

```python
def compute_loss(input_var):
    return jnp.square(input_var)

grad_fn = jax.grad(compute_loss)              # Returns a *function*
input_var = jnp.array(3.0)
gradient = grad_fn(input_var)                  # 6.0
```

Key behaviors:

- **`jax.grad` is a higher-order function** — it takes a function and returns a function.
- **No mutable state** — all arrays are immutable. Updated values must be returned.
- **The original function must return a scalar** (the loss). Auxiliary outputs can be
  returned via `has_aux=True`.

### Side-by-Side: Simple Gradient of x^2

```python
# ─── TensorFlow ─────────────────────────
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = tf.square(x)
grad = tape.gradient(y, x)           # 6.0

# ─── PyTorch ────────────────────────────
x = torch.tensor(3.0, requires_grad=True)
y = torch.square(x)
y.backward()
grad = x.grad                         # 6.0

# ─── JAX ────────────────────────────────
grad_fn = jax.grad(lambda x: jnp.square(x))
grad = grad_fn(jnp.array(3.0))        # 6.0
```

### Comparison Table: Gradient Computation Mechanics

| Step                           | TensorFlow                                         | PyTorch                            | JAX                                                 |
| ------------------------------ | -------------------------------------------------- | ---------------------------------- | --------------------------------------------------- |
| **1. Mark differentiable**     | Use `tf.Variable` (auto-watched) or `tape.watch()` | Set `requires_grad=True` on tensor | N/A — any function argument is differentiable       |
| **2. Compute forward pass**    | Inside `with tf.GradientTape():`                   | Normal code execution              | Define a pure function                              |
| **3. Trigger differentiation** | `tape.gradient(loss, vars)`                        | `loss.backward()`                  | Call `grad_fn(args)` or `jax.grad(fn)(args)`        |
| **4. Access gradients**        | Returned by `tape.gradient()`                      | Stored in `.grad` attribute        | Returned by the grad function call                  |
| **5. Clean up**                | Tape auto-disposed after `.gradient()`             | Graph freed after `.backward()`    | Nothing to clean up (stateless)                     |
| **Reuse for multiple grads**   | `persistent=True` on tape                          | `retain_graph=True` on backward    | Just call `grad_fn` again (it's a regular function) |

---

## 2. State Management and Mutability

The handling of model parameters (weights and biases) reveals deep philosophical
differences between the frameworks.

### TensorFlow: tf.Variable with In-Place Mutation

```python
W = tf.Variable(tf.random.uniform(shape=(2, 1)))
b = tf.Variable(tf.zeros(shape=(1,)))

# Update in-place
W.assign_sub(gradient_W * learning_rate)
b.assign_sub(gradient_b * learning_rate)
```

- `tf.Variable` is a **mutable** wrapper around a tensor.
- Updates use methods like `.assign()`, `.assign_sub()`, `.assign_add()`.
- The variable retains identity across updates (same Python object).

### PyTorch: Tensors with requires_grad and In-Place Mutation

```python
W = torch.rand(2, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Update in-place (must disable gradient tracking)
with torch.no_grad():
    W -= gradient_W * learning_rate
    b -= gradient_b * learning_rate
W.grad = None  # Must manually clear gradients!
b.grad = None
```

- Parameters are regular tensors with `requires_grad=True`.
- In-place updates must happen inside `torch.no_grad()` to avoid breaking the graph.
- **Gradient accumulation** is a common source of bugs — forgetting to zero gradients
  causes incorrect updates.

### JAX: Immutable Arrays and Functional Updates

```python
def training_step(state, inputs, targets):
    W, b = state
    loss, grads = grad_fn(state, inputs, targets)
    dW, db = grads
    new_state = (W - dW * learning_rate, b - db * learning_rate)
    return loss, new_state  # Return new arrays; old ones are unchanged
```

- **All JAX arrays are immutable** — there is no in-place mutation.
- State must be explicitly passed into functions and new state returned.
- This makes every function **pure** — same inputs always produce same outputs.
- State is typically bundled as a tuple, dict, or nested structure (**pytree**).

### Comparison Table: State Management

| Feature                    | TensorFlow                        | PyTorch                                                       | JAX                                       |
| -------------------------- | --------------------------------- | ------------------------------------------------------------- | ----------------------------------------- |
| **Parameter type**         | `tf.Variable`                     | `torch.Tensor` (with `requires_grad`) or `torch.nn.Parameter` | Plain `jnp.array` (immutable)             |
| **Mutability**             | Mutable (`.assign()`)             | Mutable (in-place ops)                                        | **Immutable** (no in-place ops)           |
| **Update mechanism**       | `.assign_sub()` / `.assign()`     | `-=` inside `no_grad()`                                       | Return new arrays from function           |
| **Gradient storage**       | Returned by tape                  | Accumulated in `.grad`                                        | Returned by `grad_fn`                     |
| **Manual cleanup needed?** | No                                | Yes — must zero `.grad`                                       | No — stateless by design                  |
| **State identity**         | Same object identity after update | Same object identity                                          | **New** object each update                |
| **Thread safety**          | Variable locks                    | No built-in protection                                        | Inherently safe (no shared mutable state) |

---

## 3. Gradient Computation in Detail

### TensorFlow: The GradientTape Context Manager

The `GradientTape` is TensorFlow's native autodiff mechanism. It acts as a "recorder"
for operations:

```python
# Basic usage
with tf.GradientTape() as tape:
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
gradients = tape.gradient(loss, [W, b])
# gradients is a list: [dL/dW, dL/db]
```

**How it works internally:**

1. When the tape is active, it records every operation on watched variables.
2. `tape.gradient()` replays the recorded ops in reverse (backpropagation).
3. The tape is then disposed (single-use by default).

**Watching non-variables:**

```python
x = tf.constant(3.0)          # Not a Variable — not watched by default
with tf.GradientTape() as tape:
    tape.watch(x)              # Explicitly watch it
    y = tf.square(x)
grad = tape.gradient(y, x)    # Works: 6.0
```

**Getting multiple gradients (persistent tape):**

```python
with tf.GradientTape(persistent=True) as tape:
    y = tf.square(x)
    z = tf.sin(x)
dy_dx = tape.gradient(y, x)
dz_dx = tape.gradient(z, x)   # Would fail without persistent=True
del tape                        # Must manually delete persistent tapes
```

### PyTorch: The Autograd Engine

PyTorch's autograd builds a **dynamic computation graph** (DAG) as you execute operations.
Each tensor node stores a reference to the function that created it:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                     # y.grad_fn = <PowBackward0>
z = y * 3                     # z.grad_fn = <MulBackward0>
z.backward()                   # Traverse graph backward: dz/dy -> dy/dx
print(x.grad)                  # 18.0
```

**Key autograd behaviors:**

```python
# Gradients ACCUMULATE — common bug source
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)    # 4.0

y = x ** 2
y.backward()
print(x.grad)    # 8.0  <-- accumulated! (4.0 + 4.0)

x.grad = None     # Reset manually
y = x ** 2
y.backward()
print(x.grad)    # 4.0  <-- correct again
```

**Disabling gradient tracking for updates:**

```python
with torch.no_grad():
    # Operations here don't build a graph
    W -= learning_rate * W.grad
```

### JAX: Function Transformations

JAX's approach is fundamentally different — it treats differentiation as a **program
transformation**. `jax.grad` is a higher-order function that takes a function and returns
a new function:

```python
# jax.grad: returns only gradients
grad_fn = jax.grad(compute_loss)
grads = grad_fn(params, inputs, targets)

# jax.value_and_grad: returns (loss_value, gradients)
val_grad_fn = jax.value_and_grad(compute_loss)
loss, grads = val_grad_fn(params, inputs, targets)
```

**How it works internally:**

1. `jax.grad(f)` does not execute `f` — it returns a new function.
2. When the new function is called, JAX **traces** `f` to build an intermediate
   representation (a _jaxpr_).
3. JAX differentiates the jaxpr symbolically and executes the result.
4. With `@jax.jit`, the traced and differentiated code is compiled to XLA and cached.

**Differentiating with respect to the first argument (default):**

By default, `jax.grad` differentiates with respect to the _first_ argument. You structure
your function so that the first argument is the parameter "state":

```python
def compute_loss(state, inputs, targets):
    W, b = state                # state is a pytree (tuple, dict, etc.)
    predictions = model(inputs, W, b)
    return mean_squared_error(targets, predictions)

grad_fn = jax.value_and_grad(compute_loss)
loss, grads = grad_fn((W, b), inputs, targets)
# grads has the same structure as state: (dL/dW, dL/db)
```

You can also differentiate with respect to other arguments via `argnums`:

```python
# Differentiate with respect to args 0 and 1
grad_fn = jax.grad(compute_loss, argnums=(0, 1))
```

### Comparison Table: Gradient Computation Features

| Feature                       | TensorFlow                                       | PyTorch                                        | JAX                                                  |
| ----------------------------- | ------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------------- |
| **API entry point**           | `tf.GradientTape()`                              | `tensor.backward()`                            | `jax.grad()` / `jax.value_and_grad()`                |
| **Returns gradients as**      | List matching `sources` arg                      | Stored in `.grad` attributes                   | Pytree matching first arg structure                  |
| **Loss + gradients together** | Compute loss inside tape, call `tape.gradient()` | Compute loss, call `.backward()`, read `.grad` | Use `jax.value_and_grad()` — returns `(loss, grads)` |
| **Select which params**       | Pass list to `tape.gradient()`                   | Set `requires_grad=True`                       | Structure as first arg, or use `argnums`             |
| **Gradient accumulation**     | No (fresh each tape)                             | Yes — accumulates by default                   | No (stateless — fresh each call)                     |
| **Persistent gradients**      | `persistent=True`                                | `retain_graph=True`                            | N/A — `grad_fn` is reusable by nature                |
| **Second derivatives**        | Nested tapes                                     | `create_graph=True`                            | Nested `jax.grad()` calls                            |

---

## 4. Higher-Order Gradients

Computing second (or higher) derivatives is straightforward in all three frameworks, but
the mechanisms differ:

**TensorFlow — nested tapes:**

```python
x = tf.Variable(0.0)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = 4.9 * x ** 2       # position = ½ g t²
    dy_dx = inner_tape.gradient(y, x)   # velocity = g t
d2y_dx2 = outer_tape.gradient(dy_dx, x) # acceleration = g = 9.8
```

**PyTorch — create_graph=True:**

```python
x = torch.tensor(0.0, requires_grad=True)
y = 4.9 * x ** 2
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]  # 9.8
```

**JAX — compose jax.grad:**

```python
def f(x):
    return 4.9 * x ** 2

df = jax.grad(f)           # First derivative
d2f = jax.grad(df)         # Second derivative — just compose!
d2f(0.0)                   # 9.8
```

### Comparison Table: Higher-Order Differentiation

| Aspect                 | TensorFlow                          | PyTorch                                       | JAX                                                   |
| ---------------------- | ----------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **Mechanism**          | Nest `GradientTape`                 | `create_graph=True` flag                      | Compose `jax.grad` calls                              |
| **Syntactic overhead** | Moderate (nesting)                  | Low (flag)                                    | **Minimal** (just compose)                            |
| **Arbitrary order**    | Yes (deeper nesting)                | Yes (chain `autograd.grad`)                   | Yes (chain `jax.grad`)                                |
| **Elegance**           | Becomes verbose                     | Moderate                                      | **Most elegant** — natural function composition       |
| **Memory overhead**    | Each tape holds a copy of the graph | Retains graph across multiple backward passes | Traces expanded graph; compiled once if `@jax.jit`-ed |

JAX's composability here is arguably its most distinctive advantage: because `jax.grad`
returns a plain function, you can compose it with itself, with `jax.jit`, with `jax.vmap`,
and more — all using ordinary function composition.

---

## 5. Handling Complex Loss Functions

### Multi-Variable Gradients

In real models, you need gradients for _multiple_ parameters (e.g., weights and biases
of every layer). Each framework handles this differently:

**TensorFlow:**

```python
with tf.GradientTape() as tape:
    loss = compute_loss(W, b, inputs, targets)
grad_W, grad_b = tape.gradient(loss, [W, b])  # Pass a list of variables
```

**PyTorch:**

```python
loss = compute_loss(W, b, inputs, targets)
loss.backward()
grad_W = W.grad   # Access individually
grad_b = b.grad
```

**JAX:** Bundle all parameters as the first argument (a pytree):

```python
def compute_loss(state, inputs, targets):
    W, b = state
    ...
    return loss

grad_fn = jax.value_and_grad(compute_loss)
loss, grads = grad_fn((W, b), inputs, targets)
grad_W, grad_b = grads  # grads mirrors the structure of state
```

### Auxiliary Outputs

Sometimes the loss function computes useful by-products (e.g., predictions, metrics).

**TensorFlow** — just compute them inside the tape; everything is a normal Python variable:

```python
with tf.GradientTape() as tape:
    predictions = model(inputs)
    loss = mse(targets, predictions)
grads = tape.gradient(loss, variables)
# predictions is already available — no special mechanism needed
```

**PyTorch** — similarly straightforward. Everything computed before `.backward()` is
available as normal variables:

```python
predictions = model(inputs)
loss = mse(targets, predictions)
loss.backward()
# predictions is still available
```

**JAX** — requires `has_aux=True` because the function must normally return a scalar:

```python
def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mse(targets, predictions)
    return loss, predictions               # Return (scalar, auxiliary)

grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
(loss, predictions), grads = grad_fn((W, b), inputs, targets)
```

### Comparison Table: Complex Gradient Scenarios

| Scenario                        | TensorFlow                           | PyTorch                          | JAX                                      |
| ------------------------------- | ------------------------------------ | -------------------------------- | ---------------------------------------- |
| **Multi-variable grads**        | Pass list to `tape.gradient()`       | Each tensor has its own `.grad`  | Bundle as pytree in first arg            |
| **Auxiliary outputs**           | Just use normal Python vars          | Just use normal Python vars      | Requires `has_aux=True`                  |
| **Non-scalar loss**             | Works (but needs care)               | Must be scalar for `.backward()` | Must be scalar for `jax.grad()`          |
| **Gradient wrt non-first arg**  | List of targets in `tape.gradient()` | Any tensor with `requires_grad`  | Use `argnums` parameter                  |
| **Nested parameter structures** | Flat list of variables               | Flat via `model.parameters()`    | Arbitrary pytrees (tuples, dicts, lists) |

---

## 6. JIT Compilation and Performance

All three frameworks offer just-in-time (JIT) compilation to accelerate repeated
computations. Importantly, the interaction between JIT and autodiff differs:

**TensorFlow:**

```python
@tf.function(jit_compile=True)      # XLA compilation
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mse(targets, predictions)
    grads = tape.gradient(loss, [W, b])
    W.assign_sub(grads[0] * lr)
    b.assign_sub(grads[1] * lr)
    return loss
```

- `@tf.function` traces the Python code into a TensorFlow graph.
- `jit_compile=True` additionally compiles the graph with XLA.
- **Side effects are allowed** (e.g., `Variable.assign`) because TF's graph supports them.

**PyTorch:**

```python
compiled_step = torch.compile(training_step)
# or
@torch.compile
def training_step(inputs, targets):
    ...
```

- `torch.compile` uses TorchDynamo to trace and Inductor to compile.
- Supports dynamic shapes and Python control flow (with some limitations).
- **Side effects are allowed** (in-place mutation, `.backward()`).

**JAX:**

```python
@jax.jit
def training_step(state, inputs, targets):
    loss, grads = grad_fn(state, inputs, targets)
    dW, db = grads
    W, b = state
    new_state = (W - dW * lr, b - db * lr)
    return loss, new_state
```

- `@jax.jit` traces the function to a jaxpr and compiles it with XLA.
- **The function must be stateless and pure** — no side effects, no mutation.
- All state that changes must be passed in as arguments and returned as outputs.
- JIT and `jax.grad` compose naturally: you can `jit(grad(f))` or `grad(jit(f))`.

### Comparison Table: Compilation and Performance

| Feature                     | TensorFlow                                        | PyTorch                           | JAX                                                 |
| --------------------------- | ------------------------------------------------- | --------------------------------- | --------------------------------------------------- |
| **JIT decorator**           | `@tf.function` / `@tf.function(jit_compile=True)` | `@torch.compile`                  | `@jax.jit`                                          |
| **Compilation backend**     | XLA                                               | TorchInductor (default)           | XLA                                                 |
| **Side effects allowed?**   | Yes (`Variable.assign`)                           | Yes (in-place ops, `.backward()`) | **No** — must be pure function                      |
| **Dynamic shapes**          | Limited                                           | Supported (with guards)           | Supported via `static_argnums` / shape polymorphism |
| **Composability with grad** | Tape inside `@tf.function`                        | `.backward()` inside compiled fn  | `jax.jit(jax.grad(f))` — natural composition        |
| **TPU support**             | Yes (via XLA)                                     | Limited                           | **Best** (designed for XLA/TPU)                     |
| **Caching**                 | Traces cached per input signature                 | Graph cached with guards          | Traces cached per static arg shapes                 |

---

## 7. End-to-End Training Loop Comparison

Below is the complete training step for a linear classifier in each framework. The model
computes `y = Wx + b`, and the loss is mean squared error:

### TensorFlow Training Step

```python
import tensorflow as tf

# --- State: mutable tf.Variables ---
W = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(tf.zeros(shape=(output_dim,)))
learning_rate = 0.1

def model(inputs, W, b):
    return tf.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    return tf.reduce_mean(tf.square(targets - predictions))

@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:             # 1. Open tape
        predictions = model(inputs, W, b)       # 2. Forward pass (recorded)
        loss = mean_squared_error(targets, predictions)
    dW, db = tape.gradient(loss, [W, b])        # 3. Compute gradients
    W.assign_sub(dW * learning_rate)            # 4. Update in-place
    b.assign_sub(db * learning_rate)
    return loss

# Training loop
for step in range(40):
    loss = training_step(inputs, targets, W, b)
```

### PyTorch Training Step

```python
import torch

# --- State: tensors with requires_grad ---
W = torch.rand(input_dim, output_dim, requires_grad=True)
b = torch.zeros(output_dim, requires_grad=True)
learning_rate = 0.1

def model(inputs, W, b):
    return torch.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    return torch.mean(torch.square(targets - predictions))

def training_step(inputs, targets, W, b):
    predictions = model(inputs, W, b)           # 1. Forward pass (graph built)
    loss = mean_squared_error(targets, predictions)
    loss.backward()                              # 2. Backprop (populate .grad)
    with torch.no_grad():                        # 3. Disable tracking
        W -= W.grad * learning_rate              # 4. Update in-place
        b -= b.grad * learning_rate
    W.grad = None                                # 5. Clear gradients!
    b.grad = None
    return loss

# Training loop
for step in range(40):
    loss = training_step(inputs, targets, W, b)
```

### JAX Training Step

```python
import jax
import jax.numpy as jnp

# --- State: immutable arrays in a tuple ---
W = jnp.array(np.random.uniform(size=(input_dim, output_dim)))
b = jnp.zeros((output_dim,))
learning_rate = 0.1

def model(inputs, W, b):
    return jnp.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    return jnp.mean(jnp.square(targets - predictions))

def compute_loss(state, inputs, targets):       # 1. Pure loss function
    W, b = state
    predictions = model(inputs, W, b)
    return mean_squared_error(targets, predictions)

grad_fn = jax.value_and_grad(compute_loss)      # 2. Transform into grad function

@jax.jit
def training_step(state, inputs, targets):
    loss, grads = grad_fn(state, inputs, targets) # 3. Forward + backward in one call
    dW, db = grads
    W, b = state
    new_state = (W - dW * learning_rate,          # 4. Create new state (no mutation)
                 b - db * learning_rate)
    return loss, new_state

# Training loop — state is threaded through
state = (W, b)
for step in range(40):
    loss, state = training_step(state, inputs, targets)
```

### Comparison Table: Training Loop Anatomy

| Step                        | TensorFlow                           | PyTorch                                      | JAX                                           |
| --------------------------- | ------------------------------------ | -------------------------------------------- | --------------------------------------------- |
| **1. Initialize params**    | `tf.Variable(...)`                   | `torch.rand(..., requires_grad=True)`        | `jnp.array(...)` (plain arrays)               |
| **2. Forward pass**         | Inside `GradientTape` block          | Normal code (graph built silently)           | Inside a pure `compute_loss` function         |
| **3. Compute gradients**    | `tape.gradient(loss, [W, b])`        | `loss.backward()`; read `.grad`              | `grad_fn(state, ...)` returns `(loss, grads)` |
| **4. Update parameters**    | `W.assign_sub(dW * lr)`              | `W -= W.grad * lr` in `no_grad()`            | `W_new = W - dW * lr` (new array)             |
| **5. Clean up**             | Tape auto-disposed                   | Must zero `.grad`                            | Nothing (stateless)                           |
| **6. JIT compilation**      | `@tf.function(jit_compile=True)`     | `@torch.compile`                             | `@jax.jit`                                    |
| **State across iterations** | Variables mutated in-place           | Tensors mutated in-place                     | New state returned; rebind `state` variable   |
| **Number of manual steps**  | 4 (open tape, forward, grad, update) | 5 (forward, backward, no_grad, update, zero) | 3 (call grad_fn, compute new state, return)   |

---

## 8. The Conceptual Models Compared

### Forward / Backward Mode Autodiff

All three frameworks implement **reverse-mode automatic differentiation** (backpropagation),
which is efficient when there are many inputs (parameters) and few outputs (scalar loss).
However, the _way_ they express this differs:

| Framework  | Model                                                             |
| ---------- | ----------------------------------------------------------------- |
| TensorFlow | **Tape metaphor** — record a tape, play it backward               |
| PyTorch    | **Graph metaphor** — build a DAG of operations, traverse backward |
| JAX        | **Transformation metaphor** — transform $f$ into $\nabla f$       |

### Comparison Table: Conceptual Models

| Dimension                    | TensorFlow                   | PyTorch                                 | JAX                                             |
| ---------------------------- | ---------------------------- | --------------------------------------- | ----------------------------------------------- |
| **Core abstraction**         | Tape (recording device)      | Dynamic computation graph               | Function transformation                         |
| **When is the graph built?** | During `with GradientTape()` | During any op on grad-tracked tensor    | During `jax.grad(fn)(args)` tracing             |
| **Graph persistence**        | Tape scope                   | Until `.backward()`                     | No persistent graph (traced on demand)          |
| **Programming style**        | Imperative + context manager | Imperative + implicit tracking          | **Functional** (stateless, pure)                |
| **Closest math analogy**     | Recording & replaying        | Chain rule on a DAG                     | $\nabla$ operator on functions                  |
| **Composability**            | Moderate (nest tapes)        | Moderate (chains)                       | **High** (`grad`, `jit`, `vmap` compose freely) |
| **Debugging ease**           | Good (eager by default)      | **Best** (fully eager, standard Python) | Harder (tracing can obscure errors)             |

---

## 9. Strengths and Trade-Offs

### Comparison Table: Practical Strengths and Weaknesses

| Dimension                    | TensorFlow                                                             | PyTorch                               | JAX                                          |
| ---------------------------- | ---------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------- |
| **Learning curve**           | Moderate — tape pattern is intuitive but `tf.function` adds complexity | **Easiest** — feels like plain Python | Steepest — functional paradigm + XLA tracing |
| **Debugging**                | Good in eager; harder under `@tf.function`                             | **Best** — standard Python debugging  | Hardest — traced jaxprs are opaque           |
| **Raw speed**                | Good (XLA)                                                             | Good (Inductor)                       | **Often fastest** (XLA, designed for speed)  |
| **TPU support**              | Good                                                                   | Limited                               | **Best** (native XLA/TPU)                    |
| **Purity enforcement**       | None (allows side effects)                                             | None (allows side effects)            | **Strict** (no side effects in JIT)          |
| **Gradient accumulation**    | No (per-tape)                                                          | Yes (manual clear needed)             | No (per-call)                                |
| **Higher-order derivatives** | Supported (nested tapes)                                               | Supported (`create_graph`)            | **Elegant** (compose `jax.grad`)             |
| **NumPy compatibility**      | Separate API (`tf.math`)                                               | Separate API (`torch.*`)              | **Near-identical** (`jnp.*`)                 |
| **Custom training loops**    | Moderate verbosity                                                     | Low verbosity                         | Higher verbosity (must thread state)         |
| **Parallelism primitives**   | `tf.distribute.Strategy`                                               | DDP / FSDP                            | `jax.pmap` / `jax.sharding` (first-class)    |
| **Ecosystem maturity**       | Very mature                                                            | **Most mature** (research standard)   | Growing rapidly                              |

---

## 10. Quick Reference Cheat Sheet

| Task                              | TensorFlow                                   | PyTorch                                       | JAX                                                    |
| --------------------------------- | -------------------------------------------- | --------------------------------------------- | ------------------------------------------------------ |
| Create a differentiable variable  | `tf.Variable(x)`                             | `torch.tensor(x, requires_grad=True)`         | `jnp.array(x)` (all arrays are differentiable)         |
| Open a gradient context           | `with tf.GradientTape() as tape:`            | N/A (automatic)                               | N/A (transform the function instead)                   |
| Compute forward pass              | Code inside tape block                       | Normal code                                   | Define a pure function                                 |
| Compute gradients                 | `tape.gradient(loss, vars)`                  | `loss.backward(); var.grad`                   | `jax.grad(fn)(args)` or `jax.value_and_grad(fn)(args)` |
| Compute loss + gradients together | Compute loss in tape, then `tape.gradient()` | Compute loss, `.backward()`, read `.grad`     | `jax.value_and_grad(fn)(args)` — single call           |
| Update parameters                 | `var.assign_sub(lr * grad)`                  | `with torch.no_grad(): var -= lr * grad`      | `new_var = var - lr * grad` (return it)                |
| Zero gradients                    | N/A (new tape each time)                     | `var.grad = None` or `optimizer.zero_grad()`  | N/A (stateless)                                        |
| Second derivative                 | Nested `GradientTape`                        | `torch.autograd.grad(..., create_graph=True)` | `jax.grad(jax.grad(fn))`                               |
| JIT compile                       | `@tf.function(jit_compile=True)`             | `@torch.compile`                              | `@jax.jit`                                             |
| Auxiliary outputs from loss       | Just use Python variables                    | Just use Python variables                     | `has_aux=True` in `value_and_grad()`                   |
| Differentiate wrt specific args   | Pass them in the list to `tape.gradient()`   | Set `requires_grad=True` on them              | `argnums=(0, 2)` in `jax.grad()`                       |

---

## Glossary

### Autograd

PyTorch's automatic differentiation engine. Builds a dynamic computation graph during the
forward pass and traverses it backward on `.backward()`.

### Automatic Differentiation (Autodiff)

A family of techniques for computing exact derivatives of programs. Both forward-mode and
reverse-mode exist; deep learning frameworks use reverse-mode (backpropagation) by default.

### Backpropagation (Backprop)

The reverse-mode autodiff algorithm: compute the gradient of a scalar loss with respect to
all parameters by applying the chain rule from output to input through the computation graph.

### Computation Graph

A directed acyclic graph (DAG) where nodes are operations and edges are tensors. The graph
represents the flow of data from inputs to loss (forward) and from loss to gradients
(backward).

### Dynamic Graph

A computation graph that is built on-the-fly during execution (as opposed to being defined
statically before execution). Both PyTorch and eager-mode TensorFlow use dynamic graphs.

### Eager Execution

Running operations immediately as they are called, returning concrete values (as opposed
to building a graph for later execution). All three frameworks default to eager mode for
interactive development.

### GradientTape

TensorFlow's context manager for recording differentiable computations. Operations on
watched tensors inside `with tf.GradientTape() as tape:` are recorded for later gradient
computation.

### has_aux

A parameter in `jax.value_and_grad()` that tells JAX the loss function returns a tuple
`(loss, auxiliary_outputs)` instead of just a scalar loss. This allows extracting
additional computed values alongside gradients.

### Jaxpr

JAX's intermediate representation (IR): a functional expression language used internally
when JAX traces a Python function for transformation (grad, jit, vmap, etc.).

### JIT (Just-In-Time Compilation)

Compiling a function the first time it is called (and caching the compiled version for
subsequent calls). All three frameworks support JIT compilation for performance:
`@tf.function`, `@torch.compile`, `@jax.jit`.

### Metaprogramming

Having functions that operate on other functions — e.g., `jax.grad(f)` takes a function `f`
and returns a new function that computes the gradient of `f`. This is JAX's core design pattern.

### Pure Function

A function with no side effects and no dependence on external mutable state. Given the same
inputs, it always returns the same outputs. JAX requires functions to be pure for `@jax.jit`
and `jax.grad`.

### Pytree

A JAX term for a nested data structure composed of tuples, lists, and dicts (with array
leaves). `jax.grad` returns gradients as a pytree matching the structure of the
differentiated argument.

### requires_grad

A boolean flag on PyTorch tensors that enables gradient tracking. The autograd engine only
builds computation graph nodes for operations involving tensors with `requires_grad=True`.

### Reverse-Mode Autodiff

The variant of autodiff where the computation graph is traversed from outputs to inputs.
Efficient when there are many inputs (parameters) and few outputs (scalar loss) — the
standard case in deep learning.

### tf.Variable

TensorFlow's mutable tensor wrapper. Variables are automatically watched by `GradientTape`
and support in-place updates (`.assign()`, `.assign_sub()`, `.assign_add()`).

### Tracing

The process by which a framework executes a function with abstract/symbolic inputs to build
an internal representation (graph, jaxpr, etc.) for optimization and compilation. Used by
`@tf.function`, `@torch.compile`, and `@jax.jit`.

### value_and_grad

A JAX utility function. `jax.value_and_grad(f)` returns a function that computes both the
output of `f` and the gradient of `f` with respect to its first argument, in a single call.
More efficient than calling `f` and `jax.grad(f)` separately.

### XLA (Accelerated Linear Algebra)

A compiler for linear algebra operations developed by Google. Used by both TensorFlow
(via `jit_compile=True`) and JAX (via `@jax.jit`) to optimize and fuse operations for
CPU, GPU, and TPU.
