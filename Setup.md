# Local Setup — Mac Apple Silicon (M1/M2/M3/M4)

Guide for running the **Deep Learning with Python, 3rd Edition** notebooks
locally with GPU acceleration on Apple Silicon.

---

## Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- [pyenv](https://github.com/pyenv/pyenv) + [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
- [pipx](https://pypa.github.io/pipx/) (for global dev tools)

### Shell config (`~/.zshrc`)

Make sure these lines are present:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

---

## 1. Create the Python environment

```bash
pyenv install 3.12.9
pyenv virtualenv 3.12.9 dl-chollet-3rd
pyenv local dl-chollet-3rd
```

`pyenv local` writes a `.python-version` file in the project root. With
`pyenv virtualenv-init` in your shell config, the virtualenv auto-activates
whenever you `cd` into this directory.

### Why Python 3.12?

- TensorFlow macOS ARM64 wheels are reliably published for 3.12.
- Apple's `jax-metal` plugin has tested wheels for 3.12.
- Python 3.13+ may work for PyTorch/JAX alone, but TF + Apple Metal plugin
  wheel availability is not guaranteed on 3.13/3.14 as of Feb 2026.

---

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall tensorflow-metal -y
```

### Why uninstall `tensorflow-metal`?

TensorFlow 2.20 on macOS pulls in `tensorflow-metal` as a dependency.
However, as of Feb 2026, `tensorflow-metal==1.2.0` (the only version with a
Python 3.12 wheel) has a **broken rpath**: its `libmetal_plugin.dylib` expects
`_pywrap_tensorflow_internal.so` at a Bazel-generated relative path
(`_solib_darwin_arm64/...`) that does not exist in TF 2.18 or 2.20's installed
package layout. `tensorflow-metal==1.1.0` has no 3.12 wheel. This is an
Apple-side packaging bug with no workaround.

**This has zero functional impact on these notebooks:**

- All GPU-heavy training runs through **JAX + jax-metal** (primary backend)
- PyTorch GPU cells use **MPS** (built-in, no plugin needed)
- TF is only used for `tf.data` pipelines (CPU), `keras.datasets` (downloads),
  and a handful of `tf.GradientTape` demos (ch. 3, 10) that run fine on CPU

---

## 3. Verify the installation

```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "import jax; print('JAX', jax.__version__); print(jax.devices())"
python -c "import torch; print('PyTorch', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
python -c "import keras; print('Keras', keras.__version__)"
```

Expected output (versions may differ slightly):

```
TF 2.20.0
JAX 0.4.36
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

```
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
| `metadata.kernelspec`         | Cell structure               |
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
pyenv virtualenv-delete dl-chollet-3rd -f
pyenv virtualenv 3.12.9 dl-chollet-3rd
pyenv local dl-chollet-3rd
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall tensorflow-metal -y
```

---

## GPU backend summary

| Backend    | Version | GPU Acceleration        | Role in notebooks                     |
| ---------- | ------- | ----------------------- | ------------------------------------- |
| JAX        | 0.4.36  | Metal (via `jax-metal`) | Primary Keras training backend        |
| PyTorch    | 2.10.0  | MPS (built-in)          | `%%backend torch` cells               |
| TensorFlow | 2.20.0  | CPU only                | `tf.data` pipelines, `keras.datasets` |
| Keras      | 3.13.2  | Via JAX Metal           | Multi-backend framework               |
