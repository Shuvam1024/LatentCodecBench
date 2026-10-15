# Contributing

This is a research repository. Changes land as small pull requests against `main`, with tests and a devlog entry.

## Branches

Branch from `main`. Name the branch with the change type and a short description, lowercase, hyphens:

```text
feat/clip-manifest
fix/vmaf-alignment
docs/adr-0002
test/psnr-known-value
ci/ffmpeg-libvmaf
```

One plan step per pull request when the step is small enough to review on its own.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <imperative summary>
```

Types in use: `feat`, `fix`, `docs`, `test`, `ci`, `chore`, `refactor`, `perf`.

The subject is imperative and fits in one line. The body says why, when the subject is not enough. Example: `feat(metrics): add PSNR against a known patch`.

## Checks

Before opening a pull request:

```bash
uv sync --extra vae
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`uv sync` without the extra still installs the codec baseline. The VAE shape tests skip in that environment. CI installs the extra and sets `LATENTCODECBENCH_REQUIRE_VAE=1`, so a skipped shape test fails the job there.

Install the git hooks once, so Ruff runs on commit:

```bash
uv run pre-commit install
```

## Pull request checklist

- [ ] Tests cover the change (`uv run pytest`)
- [ ] Ruff and mypy are clean
- [ ] `docs/DEVLOG.md` has a dated entry for this PR
- [ ] A design choice that someone might reverse has an ADR in `docs/decisions/`
- [ ] No clips, results, or checkpoints are committed
- [ ] CI is green

## Notebooks

Colab notebooks in `notebooks/` call the library. They do not become the only copy of an algorithm. Weights stay out of git.
