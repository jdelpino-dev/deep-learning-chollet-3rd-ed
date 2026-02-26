# Deep Learning Ecosystem Mapping: A Two-Level Architecture

> **How today's deep learning frameworks articulate across levels — from low-level
> tensor engines to high-level modeling and training interfaces.**

---

## Table of Contents

- [Deep Learning Ecosystem Mapping: A Two-Level Architecture](#deep-learning-ecosystem-mapping-a-two-level-architecture)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [The Two-Level Model](#the-two-level-model)
  - [Level 1 — Core Tensor + Autodiff Engines](#level-1--core-tensor--autodiff-engines)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    - [JAX](#jax)
      - [How JAX transforms work: tracing and jaxprs](#how-jax-transforms-work-tracing-and-jaxprs)
      - [Pure functions and explicit state](#pure-functions-and-explicit-state)
      - [Composability: the key differentiator](#composability-the-key-differentiator)
  - [Level 2 — High-Level Modeling and Training Interfaces](#level-2--high-level-modeling-and-training-interfaces)
    - [TensorFlow Ecosystem (tf.keras)](#tensorflow-ecosystem-tfkeras)
    - [PyTorch Ecosystem (torch.nn + Community)](#pytorch-ecosystem-torchnn--community)
      - [torch.nn / nn.Module — The Core Abstraction](#torchnn--nnmodule--the-core-abstraction)
      - [PyTorch Lightning / Fabric](#pytorch-lightning--fabric)
      - [Hugging Face Trainer / Accelerate](#hugging-face-trainer--accelerate)
      - [fastai](#fastai)
      - [Other PyTorch Ecosystem Libraries](#other-pytorch-ecosystem-libraries)
    - [JAX Ecosystem (Flax / Haiku / Equinox + Optax)](#jax-ecosystem-flax--haiku--equinox--optax)
      - [Flax](#flax)
      - [Haiku](#haiku)
      - [Equinox](#equinox)
      - [Optax](#optax)
    - [Keras 3 — The Multi-Backend API](#keras-3--the-multi-backend-api)
      - [Key Capabilities](#key-capabilities)
      - [Progressive Disclosure of Complexity](#progressive-disclosure-of-complexity)
      - [Relationship to tf.keras](#relationship-to-tfkeras)
  - [How the Levels Articulate](#how-the-levels-articulate)
    - [The Vertical Relationship](#the-vertical-relationship)
    - [Cross-Framework Bridges](#cross-framework-bridges)
    - [The Spectrum of Control](#the-spectrum-of-control)
  - [Ecosystem Comparison Matrix](#ecosystem-comparison-matrix)
  - [Choosing the Right Stack](#choosing-the-right-stack)
  - [Glossary](#glossary)
    - [Autodiff](#autodiff)
    - [Autograd](#autograd)
    - [Backpropagation](#backpropagation)
    - [Callback](#callback)
    - [Computation Graph](#computation-graph)
    - [DataLoader](#dataloader)
    - [DeepSpeed](#deepspeed)
    - [Eager Execution](#eager-execution)
    - [Forward Pass](#forward-pass)
    - [FSDP](#fsdp)
    - [Functional API](#functional-api)
    - [Gradient](#gradient)
    - [JIT](#jit)
    - [Layer](#layer)
    - [Mixed Precision](#mixed-precision)
    - [Model](#model)
    - [NumPy](#numpy)
    - [OpenVINO](#openvino)
    - [Optimizer](#optimizer)
    - [Parallelization](#parallelization)
    - [Pipeline](#pipeline)
    - [Program Transforms](#program-transforms)
    - [Progressive Disclosure](#progressive-disclosure)
    - [PyTree](#pytree)
    - [scikit-learn](#scikit-learn)
    - [Sequential Model](#sequential-model)
    - [Tensor](#tensor)
    - [TorchDynamo](#torchdynamo)
    - [Training Loop](#training-loop)
    - [Transformer](#transformer)
    - [Vectorization](#vectorization)
    - [XLA](#xla)
    - [ZeRO](#zero)

---

## Introduction

The modern deep learning landscape is not a single monolithic framework — it is
an interconnected **ecosystem of tools** organized in layers. Understanding this
layered architecture is essential for making informed choices about which tools
to adopt, how to combine them, and where each fits in the workflow from research
prototype to production deployment.

This document maps today's deep learning ecosystem into **two principal levels**:

1. **Core [tensor](#tensor) + [autodiff](#autodiff) engines** (and
   [program transforms](#program-transforms))
2. **High-level modeling and training interfaces** (built-in and ecosystem)

Each level serves a distinct role, and the way they articulate — connect and
compose — is what gives practitioners the flexibility to operate anywhere along
the control-vs-convenience spectrum.

---

## The Two-Level Model

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  LEVEL 2 — High-Level Modeling & Training Interfaces                     │
│                                                                          │
│  ┌────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  tf.keras   │  │  torch.nn/Module    │  │  Flax / Haiku / Equinox │  │
│  │             │  │  + Lightning        │  │  + Optax                │  │
│  │             │  │  + HF Trainer       │  │                          │  │
│  │             │  │  + fastai / etc.    │  │                          │  │
│  └──────┬──────┘  └─────────┬───────────┘  └────────────┬─────────────┘  │
│         │                   │                            │               │
│  ┌──────┴───────────────────┴────────────────────────────┴──────────┐    │
│  │                     Keras 3 (multi-backend)                      │    │
│  │              Runs on TF · JAX · PyTorch · OpenVINO               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LEVEL 1 — Core Tensor + Autodiff Engines                                │
│                                                                          │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐              │
│  │  TensorFlow   │   │   PyTorch     │   │     JAX       │              │
│  │  (TF)         │   │              │   │               │              │
│  └───────────────┘   └───────────────┘   └───────────────┘              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Level 1 — Core Tensor + Autodiff Engines

Level 1 provides the fundamental computational substrate: multi-dimensional
array ([tensor](#tensor)) operations, [automatic
differentiation](#autodiff), hardware acceleration (GPU/TPU), and — in
JAX's case — composable [program transforms](#program-transforms).

These engines are **not typically used directly** for building full models by most
practitioners; instead, they serve as the foundation that Level 2 tools build
upon. However, they remain accessible for researchers who need maximum control.

### TensorFlow

| Aspect | Details |
| --- | --- |
| **Developed by** | Google Brain / Google DeepMind |
| **Core API** | `tf.Tensor`, `tf.GradientTape`, `tf.function` |
| **Execution model** | [Eager execution](#eager-execution) by default (since TF 2.0), with optional graph compilation via `tf.function` |
| **Compilation** | [XLA](#xla) compiler for optimized execution on GPU/TPU |
| **Key low-level features** | `tf.Variable`, `tf.GradientTape` for manual gradient computation, `tf.data` for data pipelines, `tf.distribute` for distributed training |

TensorFlow provides both [eager execution](#eager-execution) for
interactive development and graph-based execution for production performance. The
`tf.GradientTape` API allows fine-grained control over gradient computation,
while `tf.function` traces Python functions into optimized computation graphs.

**Official reference:**
[TensorFlow Core Guide](https://www.tensorflow.org/guide/core)

### PyTorch

| Aspect | Details |
| --- | --- |
| **Developed by** | Meta AI (formerly Facebook AI Research) |
| **Core API** | `torch.Tensor`, `torch.autograd` |
| **Execution model** | [Eager execution](#eager-execution) (define-by-run), with `torch.compile` for graph compilation (since PyTorch 2.0) |
| **Compilation** | [TorchDynamo](#torchdynamo) + TorchInductor for optimized execution |
| **Key low-level features** | [Autograd](#autograd) engine for automatic differentiation, dynamic computation graphs, CUDA integration |

PyTorch's [autograd](#autograd) engine is the core mechanism for
automatic differentiation. It records operations on tensors in a dynamic
[computation graph](#computation-graph), enabling gradient computation
via [backpropagation](#backpropagation). The engine supports:

- **Reverse-mode autodiff** — `loss.backward()` computes gradients through the graph
- **Higher-order gradients** — gradients of gradients via `create_graph=True`
- **Custom gradient functions** — via `torch.autograd.Function`

Since PyTorch 2.0, `torch.compile` provides an optional compilation layer that
can significantly accelerate model execution without changing the eager-mode
programming model.

**Official references:**
[Autograd Tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) ·
[Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

### JAX

| Aspect | Details |
| --- | --- |
| **Developed by** | Google DeepMind / Google Brain |
| **Core API** | `jax.numpy` (NumPy-compatible), `jax.grad`, `jax.jit`, `jax.vmap`, `jax.pmap` |
| **Execution model** | Functional — pure functions with explicit state; compiled by default via [XLA](#xla) |
| **Compilation** | [XLA](#xla) compiler for all backends (CPU, GPU, TPU) |
| **Key low-level features** | Composable [program transforms](#program-transforms), [pytree](#pytree) manipulation, PRNG system |

JAX distinguishes itself from TensorFlow and PyTorch by its **functional
programming** paradigm and its emphasis on composable [program
transforms](#program-transforms):

| Transform | Purpose |
| --- | --- |
| `jax.grad` | [Automatic differentiation](#autodiff) — reverse mode by default; `jax.jvp` for forward mode |
| `jax.jit` | [JIT compilation](#jit) via XLA for accelerated execution |
| `jax.vmap` | Automatic [vectorization](#vectorization) — maps a function over batch dimensions |
| `jax.pmap` | [Parallelization](#parallelization) — maps a function across multiple devices (SPMD) |
| `jax.shard_map` | Manual [parallelization](#parallelization) — per-device code with explicit communication collectives |
| `jax.vjp` | Vector-Jacobian product (reverse-mode autodiff building block) |
| `jax.jvp` | Jacobian-vector product (forward-mode autodiff) |
| `jax.value_and_grad` | Computes both a function's value and its gradient in a single pass |

#### How JAX transforms work: tracing and jaxprs

The mechanism behind all JAX transforms is **tracing**. When a transform like
`jax.jit` is applied to a function, JAX calls that function with special
**tracer objects** instead of real array values. Tracers record the sequence of
operations the function performs, without executing them. The recorded sequence
is encoded as a **jaxpr** (JAX expression) — a simple intermediate
representation of a functional program comprising a sequence of primitive
operations:

```python
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# Inspect the jaxpr:
print(jax.make_jaxpr(selu)(jnp.arange(5.0)))
```

```text
{ lambda ; a:f32[5]. let
    b:bool[5] = gt a 0.0
    c:f32[5] = exp a
    d:f32[5] = mul 1.67 c
    e:f32[5] = sub d 1.67
    f:f32[5] = select_n b e a
    g:f32[5] = mul 1.05 f
  in (g,) }
```

Each transform then maps this sequence of input operations to a **transformed**
sequence. For example, `jax.jit` compiles the jaxpr via [XLA](#xla),
`jax.grad` applies reverse-mode differentiation rules to each primitive, and
`jax.vmap` adds batch dimensions to every operation.

> Importantly, the jaxpr does not capture side effects — there is nothing in
> it corresponding to `print()` or `list.append()`. JAX transforms are designed
> to understand side-effect-free (a.k.a. *functionally pure*) code.
> — [JAX: How transformations work](https://docs.jax.dev/en/latest/jit-compilation.html)

#### Pure functions and explicit state

JAX requires functions to be **pure** (no side effects, deterministic output
for a given input). State — such as model parameters and random number
generator keys — must be passed explicitly as function arguments, not held in
mutable global variables. This constraint is what makes composable transforms
possible: because each function is a pure mapping from inputs to outputs, JAX
can safely analyze, differentiate, vectorize, and compile it.

This aligns naturally with functional programming but requires ecosystem
libraries (Level 2) for convenient model building.

#### Composability: the key differentiator

The most powerful aspect of JAX transforms is that they are **composable** —
you can freely nest them in any order, and they work correctly:

```python
# Compose grad, vmap, and jit:
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn)))

# Higher-order derivatives by stacking grad:
d2f = jax.grad(jax.grad(f))
d3f = jax.grad(jax.grad(jax.grad(f)))
```

This composability arises because every transform takes a function and returns
a new function with the same signature conventions. The transforms are
orthogonal — each addresses a different concern (differentiation, compilation,
batching, parallelism) — and their implementations are designed to be layered
arbitrarily. The official docs demonstrate this directly:

> `jax.jit` and `jax.vmap` are designed to be composable, which means you can
> wrap a vmapped function with `jit`, or a jitted function with `vmap`, and
> everything will work correctly.
> — [JAX: Combining transformations](https://docs.jax.dev/en/latest/automatic-vectorization.html)
>
> **JAX itself is narrowly-scoped** and focuses on efficient array operations
> and program transformations. Built around JAX is an evolving ecosystem of
> machine learning and numerical computing tools.
> — [JAX Documentation](https://docs.jax.dev/)

**Official references:**
[JAX Documentation](https://docs.jax.dev/) ·
[Key Concepts](https://docs.jax.dev/en/latest/key-concepts.html) ·
[Tracing](https://docs.jax.dev/en/latest/tracing.html) ·
[JIT Compilation](https://docs.jax.dev/en/latest/jit-compilation.html) ·
[Automatic Vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html) ·
[Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) ·
[Parallel Programming](https://docs.jax.dev/en/latest/sharded-computation.html)

---

## Level 2 — High-Level Modeling and Training Interfaces

Level 2 provides the abstractions that most practitioners use day-to-day:
[layer](#layer) definitions, [model](#model) composition,
[optimizer](#optimizer) integration, [training
loops](#training-loop), and [callbacks](#callback). These tools
build on top of Level 1 engines and dramatically reduce the boilerplate needed to
go from idea to running experiment.

### TensorFlow Ecosystem (tf.keras)

**`tf.keras`** is the dominant high-level API for TensorFlow. It provides:

- **[Layers](#layer) and [Models](#model):** A rich library of
  pre-built layers (`Dense`, `Conv2D`, `LSTM`, etc.) and model-building patterns
  ([Sequential](#sequential-model), [Functional
  API](#functional-api), subclassing)
- **Training workflow:** `model.compile()` → `model.fit()` → `model.evaluate()` →
  `model.predict()`
- **Callbacks:** Built-in [callbacks](#callback) for checkpointing,
  early stopping, learning rate scheduling, TensorBoard logging
- **Distribution:** Integration with `tf.distribute` for multi-GPU/TPU training

`tf.keras` follows the principle of **[progressive disclosure of
complexity](#progressive-disclosure)**: simple workflows are quick and
easy, while advanced customization is possible by overriding methods like
`train_step()` or writing entirely custom training loops with
`tf.GradientTape`.

**Official references:**
[Keras Guide (TF)](https://www.tensorflow.org/guide/keras) ·
[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)

---

### PyTorch Ecosystem (torch.nn + Community)

PyTorch's high-level story is different from TensorFlow's: the core framework
provides the modeling abstraction (`torch.nn` / `nn.Module`), but the training
and orchestration layer comes from a **rich ecosystem** of community libraries.

#### torch.nn / nn.Module — The Core Abstraction

`torch.nn.Module` is the base class for all neural network modules in PyTorch.
Your [models](#model) subclass it and:

1. Define [layers](#layer) in `__init__()`
2. Define the [forward pass](#forward-pass) in `forward()`

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

`nn.Module` handles:

- **Parameter registration** — automatic tracking of all learnable parameters
- **Device management** — `.to(device)` moves all parameters
- **Serialization** — `state_dict()` / `load_state_dict()` for saving/loading
- **Nesting** — [modules](#model) can contain other modules (tree structure)

However, PyTorch does **not** ship a built-in `Trainer` or `fit()` method.
The [training loop](#training-loop) is up to the user — or to ecosystem
libraries.

**Official references:**
[torch.nn](https://docs.pytorch.org/docs/stable/nn.html) ·
[nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) ·
[Build Model Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

#### PyTorch Lightning / Fabric

[**PyTorch Lightning**](https://lightning.ai/docs/pytorch/stable/) restructures
PyTorch code into a `LightningModule` and hands the [training
loop](#training-loop) off to a `Trainer`:

```python
model = MyLightningModule()
trainer = Trainer()
trainer.fit(model, train_dataloader, val_dataloader)
```

The `Trainer` handles everything under the hood:

- Enabling/disabling gradients
- Running train, validation, and test [dataloaders](#dataloader)
- Calling [callbacks](#callback) at appropriate times
- Multi-GPU/TPU distribution (`strategy="ddp"`, `strategy="fsdp"`)
- [Mixed precision](#mixed-precision) training
- Checkpointing, logging, profiling

[**Lightning Fabric**](https://lightning.ai/docs/fabric/stable/) sits at a
different point on the control spectrum — it is a lightweight abstraction that
adds distributed training and hardware acceleration to **existing** PyTorch code
with minimal changes (5 lines), while leaving the training loop design entirely
to you.

> Fabric differentiates itself from a fully-fledged trainer like Lightning's
> Trainer in these key aspects: **Fast to implement** (no restructuring needed),
> **Maximum Flexibility** (write your own training logic), and **Maximum
> Control** (everything is opt-in).
> — [Lightning Fabric docs](https://lightning.ai/docs/fabric/stable/)

**Official references:**
[Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) ·
[Lightning Fabric](https://lightning.ai/docs/fabric/stable/)

#### Hugging Face Trainer / Accelerate

[**Hugging Face Trainer**](https://huggingface.co/docs/transformers/en/trainer)
is a complete training and evaluation loop purpose-built for
[Transformers](#transformer) models:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    num_train_epochs=2,
    eval_strategy="epoch",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
```

Trainer handles the four steps of a training loop: compute loss, compute
gradients with `accelerator.backward()`, update weights, and repeat. It supports
customization by subclassing (`compute_loss()`, `training_step()`, etc.) and
via [callbacks](#callback).

[**Accelerate**](https://huggingface.co/docs/accelerate/en/index) is a lower-level
library that enables the **same PyTorch code** to run across any distributed
configuration by adding just four lines of code:

```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)
# ... standard training loop, replacing loss.backward() with
# accelerator.backward(loss)
```

Accelerate supports [FSDP](#fsdp), [DeepSpeed](#deepspeed),
and [mixed precision](#mixed-precision) training. The `Trainer` itself
is powered by Accelerate under the hood.

**Official references:**
[HF Trainer](https://huggingface.co/docs/transformers/en/trainer) ·
[HF Accelerate](https://huggingface.co/docs/accelerate/en/index)

#### fastai

[**fastai**](https://docs.fast.ai/) provides its own high-level `Learner` API
that wraps a PyTorch model, [dataloaders](#dataloader), loss function,
and [optimizer](#optimizer) into a unified training object:

```python
from fastai.vision.all import *

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
                metrics=accuracy)
learn.fit_one_cycle(10, lr_max=1e-3)
```

Key features:

- **[Progressive disclosure of complexity](#progressive-disclosure)** —
  from `vision_learner()` one-liners to fully custom training loops
- **Callbacks** — extensive callback system for every phase of training
- **Transfer learning** — built-in `freeze()` / `unfreeze()` / `fine_tune()`
- **Learning rate finder** — automated LR range test
- **Test time augmentation (TTA)** — built-in TTA for improved predictions

**Official reference:** [fastai Learner](https://docs.fast.ai/learner.html)

#### Other PyTorch Ecosystem Libraries

| Library | Description |
| --- | --- |
| [**PyTorch Ignite**](https://docs.pytorch.org/ignite/) | A flexible, high-level training library that provides an engine and event system for composing training loops |
| [**Catalyst**](https://catalyst-dl.readthedocs.io/) | A PyTorch framework for deep learning R&D, focusing on reproducibility and rapid experimentation |
| [**skorch**](https://skorch.readthedocs.io/) | A [scikit-learn](#scikit-learn)-compatible wrapper for PyTorch that enables `net.fit(X, y)` syntax, grid search, and sklearn [pipelines](#pipeline) |

---

### JAX Ecosystem (Flax / Haiku / Equinox + Optax)

Because JAX itself is a low-level [tensor](#tensor) +
[autodiff](#autodiff) engine with no built-in neural network
abstractions, the high-level deep learning experience is provided entirely by
**ecosystem libraries**. The common "standard stack" pattern on JAX combines a
**modeling library** (for layers/models) with **Optax** (for optimizers).

#### Flax

[**Flax**](https://flax.readthedocs.io/) is Google's primary neural network
library for JAX. At its core is **Flax NNX**, a simplified API offering
first-class support for Python reference semantics:

```python
from flax import nnx
import optax

class Model(nnx.Module):
    def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dmid, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.2)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x, rngs):
        x = nnx.relu(self.dropout(self.bn(self.linear(x)), rngs=rngs))
        return self.linear_out(x)

model = Model(2, 64, 3, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
```

Flax NNX is an evolution of the earlier **Flax Linen** API, bringing a simpler,
more Pythonic approach. Linen used an `init`/`apply` pattern with explicit
parameter passing, while NNX allows the use of regular Python objects.

**Official reference:** [Flax Documentation](https://flax.readthedocs.io/)

#### Haiku

[**Haiku**](https://dm-haiku.readthedocs.io/) (by DeepMind) provides simple,
composable abstractions for machine learning research on JAX. It uses a
`transform` pattern that converts stateful Python functions into pure functions
JAX can work with:

```python
import haiku as hk
import jax

def forward(x):
    mlp = hk.nets.MLP([300, 100, 10])
    return mlp(x)

forward = hk.transform(forward)

rng = hk.PRNGSequence(jax.random.PRNGKey(42))
x = jnp.ones([8, 28 * 28])
params = forward.init(next(rng), x)
logits = forward.apply(params, next(rng), x)
```

> **Note:** Haiku is in maintenance mode. New projects are encouraged to use
> Flax NNX or Equinox.

**Official reference:** [Haiku Documentation](https://dm-haiku.readthedocs.io/)

#### Equinox

[**Equinox**](https://docs.kidger.site/equinox/) takes a unique approach:
[models](#model) are [pytrees](#pytree), and
`eqx.Module` simply registers your class as a PyTree. From that point, JAX
already knows how to work with it — no special framework magic:

```python
import equinox as eqx
import jax

class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias
```

Equinox also provides filtered transformations, [pytree](#pytree)
manipulation utilities, and runtime error handling — features not found in Flax
or Haiku.

**Official reference:** [Equinox Documentation](https://docs.kidger.site/equinox/)

#### Optax

[**Optax**](https://optax.readthedocs.io/) is the standard
[optimizer](#optimizer) library for JAX. Rather than bundling optimizers
inside the modeling library, JAX's ecosystem separates optimization into a
dedicated library of **composable gradient transformations**:

- Standard optimizers: `optax.adam()`, `optax.sgd()`, `optax.rmsprop()`
- Gradient clipping: `optax.clip_by_global_norm()`
- Learning rate schedules: `optax.warmup_cosine_decay_schedule()`
- Composition: chain multiple transformations together

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

> JAX's own `jax.example_libraries.optimizers` module exists only as
> **examples** — it explicitly recommends Optax for production use.

**Official reference:** [Optax Documentation](https://optax.readthedocs.io/)

---

### Keras 3 — The Multi-Backend API

[**Keras 3**](https://keras.io/keras_3/) occupies a unique position in this
mapping: it is a high-level API that runs **the same workflows** on top of
**TensorFlow, [JAX](#jax), or [PyTorch](#pytorch) backends** (plus
[OpenVINO](#openvino) for inference-only).

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```

#### Key Capabilities

| Feature | Details |
| --- | --- |
| **Multi-backend execution** | One codebase runs on TF, JAX, or PyTorch |
| **Best performance selection** | Dynamically choose the backend that delivers best performance for your model |
| **Cross-framework models** | A Keras model can be a PyTorch `Module`, a TF `SavedModel`, or a stateless JAX function |
| **`keras.ops` namespace** | Full NumPy API + neural network ops, backend-agnostic |
| **Multi-framework data pipelines** | `fit()` accepts `tf.data.Dataset`, PyTorch `DataLoader`, NumPy arrays, Pandas DataFrames |
| **Distribution** | Data parallelism + model parallelism via `keras.distribution` (JAX backend) |
| **OpenVINO backend** | Inference-only backend for optimized predictions on OpenVINO-supported hardware |

#### Progressive Disclosure of Complexity

Keras 3 follows [progressive disclosure of
complexity](#progressive-disclosure):

1. **High-level:** [Sequential](#sequential-model)/[Functional
   API](#functional-api) + `fit()`
2. **Medium:** Custom `train_step()` override
3. **Low-level:** Full custom training loops with backend-native APIs
   (`jax.grad`, `tf.GradientTape`, `torch.optim`)

#### Relationship to tf.keras

- **TensorFlow 2.0–2.15:** `tf.keras` was Keras 2, tightly coupled to TF
- **TensorFlow 2.16+:** `tf.keras` resolves to Keras 3 by default
- **Legacy Keras 2:** Available as the `tf_keras` package (maintenance mode)
- **Keras 3 (standalone):** Framework-agnostic, installable via `pip install keras`

**Official references:**
[Keras 3 Announcement](https://keras.io/keras_3/) ·
[About Keras](https://keras.io/getting_started/about/) ·
[Getting Started](https://keras.io/getting_started/)

---

## How the Levels Articulate

### The Vertical Relationship

The core insight in this mapping is that Level 2 tools are **not replacements**
for Level 1 engines — they are **built on top of** them. Every
[layer](#layer), every [optimizer](#optimizer) step, every
gradient computation ultimately reduces to Level 1 operations:

```plaintext
    High-level call                     What actually happens
    ─────────────────                   ────────────────────
    model.fit(data)              ──►    for batch in dataloader:
                                            predictions = model(batch)     # forward pass
                                            loss = loss_fn(predictions, y) # compute loss
                                            loss.backward()                # autodiff (Level 1)
                                            optimizer.step()               # update parameters
```

This means you can always **drop down** from Level 2 to Level 1 when you need
more control, and **rise back up** when convenience matters more.

### Cross-Framework Bridges

Keras 3 acts as a **horizontal bridge** across the three Level 1 engines. This
enables several powerful patterns:

- **Write once, run anywhere:** A single `model.py` runs on TF, JAX, or PyTorch
- **Mix and match:** Use a Keras model inside a PyTorch `Module`, or use a
  PyTorch `Module` inside a Keras model
- **Best backend selection:** Train on JAX (for TPU performance), deploy on TF
  (for TF-Serving/TF.js/TFLite), or use PyTorch (for ecosystem compatibility)

### The Spectrum of Control

Each ecosystem's Level 2 tools span a **spectrum from maximum control to maximum
convenience**:

```plaintext
    Maximum Control                                         Maximum Convenience
    ◄──────────────────────────────────────────────────────────────────────────►

    Raw PyTorch        Fabric        Lightning        fastai/HF Trainer
    ───────────────────────────────────────────────────────────────────
    Manual loop    + distributed    + full Trainer    + opinionated defaults
    Full control   + mixed prec     + callbacks       + transfer learning
                   + checkpoints    + logging         + LR finder
                                    + multi-GPU       + one-line training

    Raw JAX            Equinox          Flax NNX         Keras (on JAX)
    ───────────────────────────────────────────────────────────────────
    Pure functions  + pytree models  + nnx.Module      + fit()/compile()
    jax.grad/jit    + filtered       + eager init      + callbacks
    Manual state    transforms      + Optimizer        + distribution API
```

---

## Ecosystem Comparison Matrix

| Feature | TF + tf.keras | PyTorch + Lightning | JAX + Flax + Optax | Keras 3 (multi-backend) |
| --- | --- | --- | --- | --- |
| **Level 1 engine** | TensorFlow | PyTorch | JAX | TF / JAX / PyTorch |
| **Modeling abstraction** | `keras.Model` / `keras.Layer` | `nn.Module` / `LightningModule` | `nnx.Module` / `eqx.Module` | `keras.Model` / `keras.Layer` |
| **Built-in [training loop](#training-loop)** | `model.fit()` | `Trainer.fit()` | Manual (or Keras) | `model.fit()` |
| **[Autodiff](#autodiff)** | `tf.GradientTape` | `torch.autograd` | `jax.grad` | Backend-dependent |
| **[JIT compilation](#jit)** | `tf.function` / XLA | `torch.compile` | `jax.jit` (XLA) | Backend-dependent |
| **Distributed training** | `tf.distribute` | DDP / FSDP / DeepSpeed | `jax.pmap` / `jax.sharding` | `keras.distribution` |
| **Primary ecosystem** | Google / TF ecosystem | HF / Lightning / fastai | Google DeepMind | Cross-framework |
| **Best for** | Production ML, Google Cloud | Research, NLP (HF), flexibility | Large-scale research, TPU | Multi-framework portability |

---

## Choosing the Right Stack

**Choose TensorFlow + tf.keras when:**

- You need production deployment on Google Cloud, TF-Serving, TF.js, or TFLite
- You want a tightly integrated stack from data pipeline (`tf.data`) to
  deployment
- You're working with existing TF codebases

**Choose PyTorch + ecosystem when:**

- You want maximum flexibility in your [training loop](#training-loop)
- You're doing NLP/LLM work (Hugging Face ecosystem)
- You want an active open-source community with many options
- You value the define-by-run debugging experience

**Choose JAX + Flax/Equinox + Optax when:**

- You need high-performance training on TPUs
- You prefer functional programming and composable
  [transforms](#program-transforms)
- You're doing large-scale research (model parallelism, custom
  [parallelization](#parallelization))
- You want explicit control over state and randomness

**Choose Keras 3 when:**

- You want framework portability — one codebase, multiple backends
- You want to dynamically select the best-performing backend
- You want to maximize the audience for open-source model releases
- You value [progressive disclosure of
  complexity](#progressive-disclosure) and clean API design

---

## Glossary

### Autodiff

*Automatic Differentiation.*
A set of techniques for computing the [gradient](#gradient) of a
function specified by a computer program. Unlike symbolic differentiation
(applied to mathematical expressions) or numerical differentiation
(finite-difference approximation), autodiff works by applying the chain rule
systematically to elementary operations recorded during function evaluation. In
deep learning, reverse-mode autodiff ([backpropagation](#backpropagation))
is used to compute gradients of a scalar loss with respect to all model
parameters.

### Autograd

PyTorch's [automatic differentiation](#autodiff) engine. It records
operations on [tensors](#tensor) that have `requires_grad=True` in a
dynamic [computation graph](#computation-graph), and computes gradients
via `loss.backward()`. The name is a portmanteau of "automatic" and "gradient."

### Backpropagation

The algorithm for computing [gradients](#gradient) of a loss function
with respect to the parameters of a neural network. It is a specific application
of reverse-mode [autodiff](#autodiff) that propagates error signals
backward through the network, layer by layer.

### Callback

An object or function that is called at specific points during the [training
loop](#training-loop) (e.g., at the end of each epoch, before each
batch). Callbacks provide a mechanism for customizing training behavior — such
as logging metrics, saving checkpoints, or adjusting the learning rate — without
modifying the core loop code.

### Computation Graph

A directed acyclic graph (DAG) that records the sequence of operations applied
to [tensors](#tensor). Used by [autodiff](#autodiff) engines
to compute [gradients](#gradient). In PyTorch, the graph is built
dynamically (define-by-run); in TensorFlow with `tf.function`, it is traced and
compiled.

### DataLoader

A utility that loads data in batches, often with shuffling, parallel worker
processes, and prefetching. PyTorch's `torch.utils.data.DataLoader` and
TensorFlow's `tf.data.Dataset` are the primary examples.

### DeepSpeed

A deep learning optimization library by Microsoft, integrated into PyTorch
ecosystem tools (Lightning, HF Accelerate). It provides [ZeRO](#zero)
optimizations for training very large models by partitioning optimizer states,
gradients, and parameters across GPUs.

### Eager Execution

An imperative programming mode in which operations are evaluated immediately,
one at a time, as they are called from Python. Both PyTorch and TensorFlow 2.x
default to eager execution, making debugging and interactive development natural.
Contrast with graph-based execution where operations are first traced into a
[computation graph](#computation-graph) before executing.

### Forward Pass

The computation of a model's output from its input, proceeding through layers in
order. In PyTorch, this is defined by the `forward()` method of an `nn.Module`.
The reverse — computing [gradients](#gradient) — is the backward pass
([backpropagation](#backpropagation)).

### FSDP

*Fully Sharded Data Parallel.*
A PyTorch-native strategy for distributed training that shards model parameters,
gradients, and [optimizer](#optimizer) states across GPUs. Compared to
standard DDP (which replicates the full model on each GPU), FSDP dramatically
reduces per-GPU memory usage, enabling training of much larger models.

### Functional API

A Keras model-building pattern that allows arbitrary layer graph topologies
(branching, merging, multiple inputs/outputs). Unlike the
[Sequential](#sequential-model) model, the Functional API explicitly
connects layers via their input/output tensors.

### Gradient

The vector of partial derivatives of a function with respect to its parameters.
In deep learning, gradients of the loss function with respect to model weights
are computed via [backpropagation](#backpropagation) and used by
[optimizers](#optimizer) to update the weights.

### JIT

*Just-In-Time Compilation.*
A technique where code is compiled at runtime rather than ahead-of-time. In
JAX, `jax.jit` traces a Python function and compiles it via [XLA](#xla)
for optimized execution. In PyTorch 2.0+, `torch.compile` achieves similar
goals via [TorchDynamo](#torchdynamo).

### Layer

A fundamental building block of a neural network that performs a specific
computation (e.g., linear transformation, convolution, normalization). In
PyTorch, layers are `nn.Module` subclasses; in Keras, they subclass
`keras.Layer`; in Flax NNX, they subclass `nnx.Module`.

### Mixed Precision

A training technique that uses lower-precision floating-point formats (float16
or bfloat16) for most operations while keeping critical accumulations in float32.
This reduces memory usage and increases throughput on hardware with
mixed-precision support (modern GPUs, TPUs).

### Model

A composition of [layers](#layer) that defines a complete neural
network architecture. In code, it is typically an object (e.g., `nn.Module`,
`keras.Model`, `nnx.Module`) that implements a
[forward pass](#forward-pass) and holds learnable parameters.

### NumPy

The foundational Python library for numerical computing with multi-dimensional
arrays. JAX provides a `jax.numpy` API that is nearly identical to NumPy but
operates on accelerator-backed arrays and supports [differentiation](#autodiff),
[compilation](#jit), and [vectorization](#vectorization).

### OpenVINO

Intel's toolkit for optimizing and deploying inference workloads. Keras 3.8+
supports an OpenVINO backend for inference-only — models trained on any backend
can be loaded with the OpenVINO backend for optimized predictions on Intel
hardware.

### Optimizer

An algorithm that updates model parameters based on computed
[gradients](#gradient) to minimize the loss function. Common optimizers
include SGD, Adam, AdamW, and RMSprop. In PyTorch, optimizers live in
`torch.optim`; in the JAX ecosystem, they are provided by
[Optax](#optax); in Keras, they are part of `keras.optimizers`.

### Parallelization

The practice of distributing computation across multiple devices (GPUs/TPUs) or
nodes. In JAX, `jax.pmap` maps a function across devices; in PyTorch,
`DistributedDataParallel` replicates the model; in Keras 3, `keras.distribution`
provides a unified API.

### Pipeline

In the context of [scikit-learn](#scikit-learn), a sequence of data
processing steps and a final estimator bundled together. Libraries like
[skorch](#other-pytorch-ecosystem-libraries) make PyTorch models compatible with
sklearn pipelines by providing an sklearn-compatible interface.

### Program Transforms

In JAX, a **program transform** (or simply a **transformation**) is a
higher-order function that takes a Python function as input and returns a new,
transformed function as output. The JAX glossary defines them precisely:

> **transformation** — A higher-order function: that is, a function that takes
> a function as input and outputs a transformed function. Examples in JAX
> include `jax.jit`, `jax.vmap`, and `jax.grad`.
> — [JAX Glossary](https://docs.jax.dev/en/latest/glossary.html)

The core transforms are:

- **`jax.grad`** — Computes [gradients](#gradient) via reverse-mode
  [automatic differentiation](#autodiff). Given a scalar-valued function `f`,
  `jax.grad(f)` returns a new function that evaluates $\nabla f$. Can be
  stacked for higher-order derivatives:
  `jax.grad(jax.grad(f))` computes $f''$.
- **`jax.jit`** — [Just-in-time compiles](#jit) a function via [XLA](#xla).
  JAX traces the function with abstract tracer objects to extract a **jaxpr**
  (intermediate representation), which is then compiled to optimized machine
  code for the target hardware (CPU/GPU/TPU).
- **`jax.vmap`** — Automatic [vectorization](#vectorization). Transforms a
  function that operates on a single example into one that operates on a batch,
  by automatically adding batch dimensions to every operation. Eliminates the
  need to manually rewrite functions for batched inputs.
- **`jax.pmap`** — SPMD [parallelization](#parallelization). Maps a function
  across multiple devices, running the same computation on different shards of
  data in parallel.
- **`jax.shard_map`** — Manual parallelism. You write per-device code and
  use explicit communication collectives (e.g., `jax.lax.psum`).
- **`jax.jvp`** / **`jax.vjp`** — Lower-level transforms for forward-mode
  (Jacobian-vector product) and reverse-mode (vector-Jacobian product)
  autodiff respectively. `jax.grad` is built on `jax.vjp`.
- **`jax.value_and_grad`** — Computes both a function's return value and its
  gradient in a single pass.

**Composability** is the defining feature: transforms can be freely nested
in any order, and they work correctly together:

```python
# Compile a vectorized, differentiated function in one expression:
fast_batched_grad = jax.jit(jax.vmap(jax.grad(loss_fn)))
```

This works because (1) every transform takes and returns a function with the
same calling convention, (2) the transforms are orthogonal (each addresses
a different concern), and (3) JAX functions must be
*pure* — no side effects — so transforms can safely
analyze and rewrite them.

All transforms rely on a shared mechanism: **tracing**. When a transform is
applied, JAX calls the function with tracer objects that record the sequence of
primitive operations into a **jaxpr** (JAX expression). Each transform then
maps this recorded sequence to a transformed sequence — compilation for `jit`,
differentiation rules for `grad`, batched dimensions for `vmap`, etc.

**Official references:**
[Key Concepts: Transformations](https://docs.jax.dev/en/latest/key-concepts.html) ·
[Glossary](https://docs.jax.dev/en/latest/glossary.html) ·
[Tracing](https://docs.jax.dev/en/latest/tracing.html) ·
[JIT Compilation](https://docs.jax.dev/en/latest/jit-compilation.html) ·
[Automatic Vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html) ·
[Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html) ·
[Parallel Programming](https://docs.jax.dev/en/latest/sharded-computation.html)

### Progressive Disclosure

*Progressive Disclosure of Complexity.*
A design principle at the heart of Keras's API: simple workflows should be quick
and easy, while advanced workflows should be possible via a clear path that
builds upon prior knowledge. Users don't need to learn everything upfront, and
they don't fall off a "complexity cliff" when they need more control.

### PyTree

A JAX concept: any nested structure of Python containers (lists, tuples, dicts)
with array leaves. JAX transformations like `jax.grad` and `jax.jit` operate on
pytrees transparently — model parameters represented as nested dicts of arrays
are a common pytree. Equinox extends this by making `eqx.Module` instances
pytrees.

### scikit-learn

*Also known as sklearn.*
The dominant Python library for traditional machine learning (not deep learning).
Provides a consistent API (`fit(X, y)`, `predict(X)`, `score(X, y)`) for
classifiers, regressors, clustering, and preprocessing. Libraries like
[skorch](#other-pytorch-ecosystem-libraries) bridge deep learning models into
the sklearn ecosystem.

### Sequential Model

The simplest model-building pattern in Keras: a linear stack of layers where
each layer has exactly one input and one output. Created with
`keras.Sequential()` and built by calling `.add()`. Not suitable for models
with branching, multiple inputs/outputs, or shared layers — use the
[Functional API](#functional-api) instead.

### Tensor

A multi-dimensional array — the fundamental data structure in deep learning
frameworks. Scalars are 0-D tensors, vectors are 1-D, matrices are 2-D, and so
on. Operations on tensors (matrix multiplication, element-wise operations,
reductions) form the backbone of neural network computation. Each framework has
its own tensor type: `torch.Tensor`, `tf.Tensor`, `jax.Array`.

### TorchDynamo

A Python-level [JIT](#jit) compiler for PyTorch (since 2.0). It works
by intercepting Python bytecode execution, extracting PyTorch operations into a
graph (FX graph), and dispatching to a backend compiler (e.g., TorchInductor)
for optimized execution. Accessed via `torch.compile()`.

### Training Loop

The iterative process of training a neural network: (1) obtain a batch of data,
(2) compute the [forward pass](#forward-pass), (3) compute the loss,
(4) compute [gradients](#gradient) via
[backpropagation](#backpropagation), (5) update parameters via the
[optimizer](#optimizer), (6) repeat. Level 2 tools abstract this loop
to varying degrees.

### Transformer

A neural network architecture based on self-attention mechanisms, introduced in
"Attention Is All You Need" (Vaswani et al., 2017). Transformers are the
foundation of modern NLP (BERT, GPT, T5, etc.) and increasingly used in
computer vision and other domains. The Hugging Face `transformers` library
provides pre-trained transformer models.

### Vectorization

The technique of applying an operation simultaneously to all elements of an
array, leveraging hardware parallelism. In JAX, `jax.vmap` automatically
transforms a function written for a single example into one that operates on
batches — a form of automatic vectorization.

### XLA

*Accelerated Linear Algebra.*
A domain-specific compiler for linear algebra developed by Google. XLA compiles
computation graphs into optimized machine code for CPUs, GPUs, and TPUs. It is
the default compilation backend for JAX and is optionally available in
TensorFlow. XLA can fuse operations, eliminate unnecessary intermediates, and
optimize memory access patterns.

### ZeRO

*Zero Redundancy Optimizer.*
A family of memory optimization techniques in [DeepSpeed](#deepspeed)
that partition [optimizer](#optimizer) states, [gradients](#gradient),
and model parameters across data-parallel processes, eliminating memory
redundancy. ZeRO enables training models that would otherwise not fit in GPU
memory.

---

*Document generated from reference documentation of
[TensorFlow](https://www.tensorflow.org/),
[PyTorch](https://pytorch.org/),
[JAX](https://jax.dev/),
[Keras](https://keras.io/),
[Flax](https://flax.readthedocs.io/),
[Haiku](https://dm-haiku.readthedocs.io/),
[Equinox](https://docs.kidger.site/equinox/),
[Optax](https://optax.readthedocs.io/),
[PyTorch Lightning](https://lightning.ai/),
[Hugging Face](https://huggingface.co/),
[fastai](https://docs.fast.ai/),
[Ignite](https://docs.pytorch.org/ignite/),
[Catalyst](https://catalyst-dl.readthedocs.io/),
and [skorch](https://skorch.readthedocs.io/).*
