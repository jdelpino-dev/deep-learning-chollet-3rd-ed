# Local Setup — Mac Apple Silicon (M1/M2/M3/M4)

Guide for running the **Deep Learning with Python, 3rd Edition** notebooks
locally with GPU acceleration on Apple Silicon.

---

## Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) — manages Python installation, virtualenv, and dependencies
- [direnv](https://direnv.net/) (recommended) — auto-activates `.venv` via `.envrc` on `cd`
- [pipx](https://pypa.github.io/pipx/) (for global dev tools like `nbstripout`)

### Shell config (`~/.zshrc`)

Make sure these lines are present:

```bash
# uv
export PATH="$HOME/.local/bin:$PATH"

# direnv (optional but recommended)
eval "$(direnv hook zsh)"
```

---

## 1. Sync the environment

```bash
uv sync
```

That's it. `uv` reads `pyproject.toml`, resolves against `uv.lock`, creates
`.venv` in the project root, and installs all dependencies. The Python version
is pinned in `.python-version` (`3.12.13`) — uv installs it automatically if
it isn't available locally.

If direnv is set up, the `.venv` activates automatically when you `cd` into
the project. Otherwise, activate it manually:

```bash
source .venv/bin/activate
```

### Why Python 3.12?

- TensorFlow macOS ARM64 wheels are reliably published for 3.12.
- Apple's `jax-metal` plugin has tested wheels for 3.12.
- Python 3.13+ may work for PyTorch/JAX alone, but TF + Apple Metal plugin
  wheel availability is not guaranteed on 3.13/3.14 as of early 2026.

### Note on `tensorflow-metal`

TensorFlow 2.20 on macOS pulls in `tensorflow-metal` as a transitive
dependency. However, `tensorflow-metal==1.2.0` (the only version with a
Python 3.12 wheel) has a **broken rpath**: its `libmetal_plugin.dylib` expects
`_pywrap_tensorflow_internal.so` at a Bazel-generated relative path
(`_solib_darwin_arm64/...`) that does not exist in TF 2.18 or 2.20's installed
package layout. `tensorflow-metal==1.1.0` has no 3.12 wheel. This is an
Apple-side packaging bug with no workaround.

uv's resolver correctly excludes it — `tensorflow-metal` does not appear in
`uv.lock` and is never installed. No extra configuration is needed.

**This has zero functional impact on these notebooks:**

- All GPU-heavy training runs through **JAX + jax-metal** (primary backend)
- PyTorch GPU cells use **MPS** (built-in, no plugin needed)
- TF is only used for `tf.data` pipelines (CPU), `keras.datasets` (downloads),
  and a handful of `tf.GradientTape` demos (ch. 3, 10) that run fine on CPU

---

## 2. Verify the installation

```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import jax; print('JAX', jax.__version__); print(jax.devices())"
python -c "import torch; print('PyTorch', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
python -c "import keras; print('Keras', keras.__version__)"
```

Expected output (versions may differ slightly):

```plaintext
TF 2.20.0
JAX 0.4.38
[METAL(id=0)]
PyTorch 2.10.0
MPS: True
Keras 3.13.2
```

---

## 4. Set up `nbstripout` (clean notebook diffs)

`nbstripout` automatically strips notebook outputs, execution counts, and
metadata before git commits. This keeps diffs clean — only code and markdown
changes are tracked.

### Install globally via pipx (one-time)

```bash
pipx install nbstripout
```

### Configure for this repo

```bash
nbstripout --install
git config filter.nbstripout.extrakeys \
  'metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed'
```

### Verify

```bash
nbstripout --status
```

Expected output:

```plaintext
nbstripout is installed in repository '...'

Filter:
  clean = "..." -m nbstripout
  smudge = cat
  diff= "..." -m nbstripout -t
  extrakeys= metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed

Attributes:
  *.ipynb: filter: nbstripout

Diff Attributes:
  *.ipynb: diff: ipynb
```

### What gets stripped on commit

| Stripped                      | Kept                        |
| ----------------------------- | --------------------------- |
| Cell outputs (plots, logs)    | Cell source code            |
| Execution counts              | Markdown cells              |
| `metadata.kernelspec`         | Cell structure              |
| `metadata.language_info`      | Cell type (code/markdown)   |
| `metadata.widgets`            |                             |
| `cell.metadata.scrolled`      |                             |
| `cell.metadata.collapsed`     |                             |

Your local notebooks keep all outputs — you still see plots and training logs
while working. Git just never sees them.

---

## Recreating the environment

If the env gets corrupted or you need a clean slate:

```bash
rm -rf .venv
uv sync
```

---

## GPU backend summary

| Backend    | Version | GPU Acceleration        | Role in notebooks                     |
| ---------- | ------- | ----------------------- | ------------------------------------- |
| JAX        | 0.4.38  | Metal (via `jax-metal`) | Primary Keras training backend        |
| PyTorch    | 2.10.0  | MPS (built-in)          | `%%backend torch` cells               |
| TensorFlow | 2.20.0  | CPU only                | `tf.data` pipelines, `keras.datasets` |
| Keras      | 3.13.2  | Via JAX Metal           | Multi-backend framework               |
