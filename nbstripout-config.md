# nbstripout Configuration

This document explains the full `nbstripout` setup for this repo and what must be done on every machine that clones it.

---

## What it does

`nbstripout` is a git filter that automatically strips Jupyter notebook cell outputs, execution counts, and noisy metadata before git commits. This keeps the repository history clean — diffs show only code and markdown changes, not execution artifacts like plots, training logs, or widget state.

What gets **stripped** on commit:

| Stripped | Kept |
| --------------------------------- | ---------------------------- |
| Cell outputs (plots, logs, etc.) | Cell source code |
| Execution counts | Markdown cells |
| `metadata.kernelspec` | Cell structure |
| `metadata.language_info` | Cell type (code / markdown) |
| `metadata.widgets` | |
| `cell.metadata.scrolled` | |
| `cell.metadata.collapsed` | |

Your local working copy keeps all outputs — you still see plots and training logs while working. Git just never records them.

---

## How it is wired up

There are two parts to the setup:

### 1. Git filter (`nbstripout --install`)

Running `nbstripout --install` writes the following into `.git/config` (local repo config, not committed):

```ini
[filter "nbstripout"]
    clean  = <python> -m nbstripout
    smudge = cat
    required = true
    extrakeys = metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed

[diff "ipynb"]
    textconv = <python> -m nbstripout -t
```

It also writes `.gitattributes` (committed) to apply the filter to all `.ipynb` files:

```
*.ipynb filter=nbstripout
*.ipynb diff=ipynb
```

The `clean` filter runs on every `git add`, stripping outputs before staging. The `smudge` filter is `cat` (no-op), so checked-out notebooks are left untouched.

### 2. Post-merge hook (`.githooks/post-merge`)

Located at `.githooks/post-merge`, this hook runs automatically after any `git pull` or `git merge` on the `master` branch. It:

1. Detects whether the current branch is `master` (skips on all other branches).
2. Checks that `nbstripout` is installed; warns and exits gracefully if not.
3. Strips outputs from all tracked `.ipynb` files via `nbstripout`.
4. Commits the result with message `chore: strip notebook outputs after upstream sync` if any notebooks were changed.

This prevents a common annoyance: upstream notebooks often include saved outputs, and without this hook, those outputs would land in your working tree and make notebooks appear perpetually modified even though you have changed nothing.

Git uses `.githooks/` as the hooks directory because of this setting in `.git/config`:

```ini
[core]
    hooksPath = .githooks
```

See the hook source at [`.githooks/post-merge`](.githooks/post-merge).

---

## Per-machine setup (required on every clone)

These steps are **not automatic** — git filters and hooks reference local tool paths, so each contributor must run this once after cloning.

### Step 1 — Install `nbstripout` globally

```bash
pipx install nbstripout
```

> `pipx` installs the tool into an isolated environment and puts it on your `PATH`. Do not install it into the project virtualenv — the filter must be available even outside the virtualenv.

If you do not have `pipx`:

```bash
brew install pipx
pipx ensurepath
```

### Step 2 — Register the git filter for this repo

From the project root:

```bash
nbstripout --install
git config filter.nbstripout.extrakeys \
  'metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed'
```

This writes the filter and diff config into `.git/config`. The `extrakeys` setting ensures those extra metadata fields are also stripped.

### Step 3 — Activate the custom hooks directory

The `core.hooksPath` setting also lives in `.git/config` and must be set manually:

```bash
git config core.hooksPath .githooks
```

This tells git to look for hooks in the committed `.githooks/` folder instead of the default `.git/hooks/`. Without this, the `post-merge` hook will never run.

### Step 4 — Verify

```bash
nbstripout --status
```

Expected output:

```
nbstripout is installed in repository '...'

Filter:
  clean  = "..." -m nbstripout
  smudge = cat
  diff   = "..." -m nbstripout -t
  extrakeys = metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed

Attributes:
  *.ipynb: filter: nbstripout

Diff Attributes:
  *.ipynb: diff: ipynb
```

Also confirm the hooks path:

```bash
git config core.hooksPath
# Expected: .githooks
```

---

## Summary of commands (copy-paste)

```bash
# 1. Install nbstripout globally
pipx install nbstripout

# 2. Register the filter for this repo
nbstripout --install
git config filter.nbstripout.extrakeys \
  'metadata.kernelspec metadata.language_info metadata.widgets cell.metadata.scrolled cell.metadata.collapsed'

# 3. Activate the committed hooks directory
git config core.hooksPath .githooks
```

---

## Files involved

| File | Committed | Purpose |
| ----------------------------------------- | --------- | ------------------------------------------- |
| `.githooks/post-merge` | Yes | Strips notebook outputs after `git pull` |
| `.gitattributes` | Yes | Applies the filter/diff driver to `*.ipynb` |
| `.git/config` (local) | No | Stores filter definition and `hooksPath` |
