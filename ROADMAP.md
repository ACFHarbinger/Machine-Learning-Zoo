# Roadmap: Improving the Codebase for Human Understanding

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This roadmap addresses documentation, code clarity, and developer experience gaps identified through an audit of the Machine Learning Zoo codebase. Each phase is ordered by impact: earlier phases unblock the most contributors and users.

---

## Current State

| Area | Coverage | Quality | Notes |
|------|----------|---------|-------|
| README | 70% | Good | Missing architecture diagram, advanced features, project context |
| Module docstrings | ~65-75% | Mixed | Strong in core modules; weak in factories, pipelines, specialized training |
| Type annotations | ~70% | Mixed | `from __future__ import annotations` in only ~18% of files; many broad `Any` types |
| Inline comments | ~60% | Good where present | Dense algorithmic code sometimes uncommented |
| Config documentation | ~20% | Poor | No comments in YAML files; no config README |
| Test documentation | ~75% | Good | Fixture purposes undocumented |
| Examples | ~50% | Poor | Mostly verification scripts, not learning resources |
| Architecture docs | ~50% | Moderate | Covers high-level; missing design rationale, dependency graph, extensibility |
| Onboarding | ~40% | Poor | No contribution ladder, no "how to add a new X" guide |

---

## Phase 1: Entry Points (README, Architecture, Glossary)

The first things a newcomer reads. Fixing these gives the highest per-reader impact.

### 1.1 Rewrite README.md

- [ ] Add a one-paragraph "What and why" explaining the project's purpose and target audience.
- [ ] Add an ASCII architecture diagram showing Backbone + Head composition, the sidecar IPC path, and the training pipeline.
- [ ] Expand the Features section to mention RL (single + multi-agent), neuro-symbolic, federated learning, fairness auditing, explainability, domain adaptation, and continual learning.
- [ ] Fix the programmatic example to use the actual API (`build_model` with `backbone_config` / `head_config` dicts, not `backbone_kwargs`).
- [ ] Add a "Quick Start" section with a 10-line end-to-end example (build model, forward pass, evaluate).
- [ ] Add a "Project Status" badge or note (Alpha, v0.1.0) so readers know what to expect.

### 1.2 Expand ARCHITECTURE.md

- [ ] Add a module dependency diagram showing the layering: `utils` -> `models` -> `training` / `inference` -> `pipeline` -> `ipc/sidecar`.
- [ ] Document why NDJSON was chosen over gRPC, REST, or Unix sockets for the IPC transport.
- [ ] Explain the Backbone + Head + ComposedModel pattern and why it exists (combinatorial flexibility, transfer learning).
- [ ] Explain the registry system (`BACKBONE_REGISTRY`, `HEAD_REGISTRY`, `MODEL_REGISTRY`) and how `@register_backbone` / `@register_head` decorators work.
- [ ] Add a "How a request flows" section tracing a complete `inference.complete` call from NDJSON stdin to model output.
- [ ] Add a "How training works" section tracing a `training.start` call through Lightning to model checkpoint.

### 1.3 Create GLOSSARY.md

- [ ] Define domain-specific terms used across the codebase: Backbone, Head, ComposedModel, Sidecar, NDJSON, FedAvg, EWC, MAPPO, CTDE, MMD, GRL, DANN, MAML, GAE, Integrated Gradients, Demographic Parity, Disparate Impact, Equalized Odds.
- [ ] For each term, include the source module path (e.g. "EWC -- `src/training/continual.py`").

---

## Phase 2: Code-Level Documentation (Docstrings & Type Hints)

Make the source code self-documenting so developers can understand modules without external docs.

### 2.1 Docstring Pass -- Priority Modules

Bring the following under-documented modules to full docstring coverage (every public class, method, and function):

- [ ] `src/models/factories/` -- document what each factory creates and how to extend it.
- [ ] `src/pipeline/core/` -- document each Lightning module variant (Supervised, Unsupervised, SSL, GAN, Diffusion, RL).
- [ ] `src/pipeline/hpo/` -- document the HPO workflow, search spaces, and how DEHB / Optuna / Ray Tune integrate.
- [ ] `src/pipeline/meta/` -- document MAML inner/outer loop and regime detection.
- [ ] `src/pipeline/online_learning/` -- document drift detection algorithms (ADWIN, DDM) and when to use them.
- [ ] `src/pipeline/active_learning/` -- document sampling strategies and their trade-offs.
- [ ] `src/features/pipeline.py` -- document each pipeline stage (validation, GPU features, scaling, selection, regime detection).
- [ ] `src/utils/` -- add module-level docstrings to every sub-package (`profiling/`, `io/`, `security/`, `validation/`, `export/`, `generation/`, `prompts/`, `logging/`).
- [ ] `src/ml_sidecar_main.py` -- document the `MlRequestHandler.dispatch` method table and each RPC namespace.

### 2.2 Docstring Standard

- [ ] Add a "Docstring Style" section to CONTRIBUTING.md specifying:
  - Google-style docstrings.
  - Required for every public class, method, and function.
  - Must include `Args:`, `Returns:`, and `Raises:` where applicable.
  - Include a short `Example:` block for non-trivial APIs.
- [ ] Add a sample "good docstring" and "bad docstring" to the guide.

### 2.3 Type Annotation Cleanup

- [ ] Add `from __future__ import annotations` to all files that lack it.
- [ ] Replace legacy `Dict[str, Any]`, `List[int]`, `Optional[X]`, `Tuple[...]` with modern `dict[str, Any]`, `list[int]`, `X | None`, `tuple[...]` syntax.
- [ ] Audit uses of bare `Any` -- replace with narrower types where the expected structure is known (especially in config dicts, factory return types, and callback signatures).
- [ ] Ensure mypy strict mode passes without `# type: ignore` on anything that could be properly typed.

---

## Phase 3: Configuration Documentation

Config files are the primary interface for users who don't read source code.

### 3.1 Annotate YAML Configs

- [ ] Add inline comments to `configs/config.yaml` explaining every field, its valid values, and its default.
- [ ] Add inline comments to each group file under `configs/algorithm/`, `configs/env/`, `configs/model/`, `configs/data/`, `configs/tasks/`.
- [ ] For numeric fields, note the unit (e.g. `val_check_interval: 1.0  # fraction of epoch`).

### 3.2 Create configs/README.md

- [ ] Explain the Hydra config hierarchy: root defaults, group overrides, CLI overrides.
- [ ] Show how to create a new config group (e.g. adding a new algorithm).
- [ ] Show how to override a single field from the CLI (`python main.py training.learning_rate=1e-3`).
- [ ] List every config group with a one-line description.

---

## Phase 4: Inline Comments for Complex Logic

Target areas where algorithmic complexity exceeds what docstrings alone can explain.

- [ ] `src/inference/engine.py` -- annotate the provider routing logic, token refresh flows, and streaming assembly.
- [ ] `src/training/domain_adaptation.py` -- annotate the RBF-MMD kernel computation step by step.
- [ ] `src/training/continual.py` -- annotate the Fisher Information Matrix estimation loop.
- [ ] `src/training/multi_agent_rl.py` -- annotate the GAE computation and PPO clipped objective.
- [ ] `src/models/neuro_symbolic.py` -- annotate the forward-chaining logic executor and the gated integration.
- [ ] `src/envs/multi_agent.py` -- annotate the reward sharing strategies and observation construction.
- [ ] `src/features/pipeline.py` -- annotate the GPU feature engineering pipeline stages.
- [ ] `src/pipeline/core/reinforce/` -- annotate REINFORCE / A2C / A3C gradient estimation.
- [ ] `src/models/spiking/snn.py` -- annotate the surrogate gradient and LIF dynamics.
- [ ] `src/models/probabilistic/` -- annotate RBM contrastive divergence, normalizing flow transformations, and diffusion noise schedules.

---

## Phase 5: Developer Guides (How to Extend)

Concrete guides so contributors don't have to reverse-engineer patterns from existing code.

### 5.1 Create EXTENSION_GUIDE.md

- [ ] **How to add a new Backbone**: step-by-step with a minimal example (create config dataclass, subclass `Backbone`, use `@register_backbone`, add to `__init__.py`).
- [ ] **How to add a new Head**: same pattern for heads.
- [ ] **How to add a new Environment**: subclass `MultiAgentEnvBase` or `gym.Env`, register in `__init__.py`.
- [ ] **How to add a new Training Strategy**: where to put the module, how to integrate with Lightning callbacks.
- [ ] **How to add a new Inference Provider**: extend `InferenceEngine` with a new `_complete_<provider>_stream` method.

### 5.2 Create MODULE_GUIDE.md

- [ ] For each top-level package under `src/`, write 2-3 sentences on its purpose, its key public classes, and which other packages it depends on.
- [ ] Include a table: Package | Purpose | Key Classes | Depends On.

---

## Phase 6: Test & Example Clarity

### 6.1 Test Documentation

- [ ] Add a `tests/README.md` explaining: directory structure, naming conventions, how to run subsets (`pytest -m slow`, `pytest tests/unit/`), and fixture organization.
- [ ] Add module-level docstrings to every file under `tests/fixtures/` explaining what each fixture provides and when to use it.
- [ ] Add a "How to write a test for a new module" section to CONTRIBUTING.md.

### 6.2 Example Upgrades

- [ ] Audit every file in `examples/` and categorize as "tutorial" or "verification".
- [ ] For each verification script, add a header docstring explaining: what it verifies, what output to expect, and which modules it exercises.
- [ ] Create at least 3 new tutorial-style examples:
  - `examples/tutorial_build_and_train.py` -- build a model, train on synthetic data, evaluate.
  - `examples/tutorial_multi_agent_rl.py` -- set up a cooperative env, train with MAPPO, print metrics.
  - `examples/tutorial_neuro_symbolic.py` -- build a neuro-symbolic model, apply constraints, inspect rule attention.
- [ ] Each tutorial should be runnable with `python examples/tutorial_*.py` and produce self-explanatory console output.

---

## Phase 7: Error Messages & Developer Experience

### 7.1 Error Message Audit

- [ ] Audit `try/except` blocks across `src/inference/`, `src/training/`, and `src/pipeline/`. Ensure every caught exception either re-raises with context or logs a message that tells the user what to do.
- [ ] For config-related errors, include the path to the relevant YAML file and the field that is invalid.
- [ ] For registry lookup failures, always include the list of available components (already done in `composed.py` and `registry.py` -- propagate this pattern everywhere).

### 7.2 Logging Standards

- [ ] Ensure every module uses `logger = logging.getLogger(__name__)` consistently.
- [ ] Standardize log levels: `DEBUG` for internal state, `INFO` for lifecycle events, `WARNING` for recoverable issues, `ERROR` for failures.
- [ ] Add a "Logging Conventions" section to DEVELOPMENT.md.

---

## Phase 8: Visual Documentation

### 8.1 Diagrams

- [ ] Create an ASCII or Mermaid diagram of the Backbone + Head composition system for ARCHITECTURE.md.
- [ ] Create a data-flow diagram for the sidecar IPC path (request -> handler -> engine -> model -> response).
- [ ] Create a module dependency graph (can be auto-generated from imports or hand-drawn).

### 8.2 Notebook Maintenance

- [ ] Keep `notebooks/getting_started.ipynb` in sync with API changes. Add a CI check that the notebook's imports resolve without errors.
- [ ] Consider adding a second notebook: `notebooks/advanced_training.ipynb` covering distributed training, HPO, and meta-learning workflows.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Public API docstring coverage | > 95% |
| Files with `from __future__ import annotations` | 100% |
| Config fields with inline comments | 100% |
| Modules with module-level docstrings | 100% |
| Tutorial-style examples | >= 3 |
| Architectural diagrams | >= 3 |
| Extension guide sections | >= 5 (backbone, head, env, training, inference) |

---

_This plan is scoped to documentation and clarity improvements. Feature development is tracked separately in [ROADMAP.md](ROADMAP.md)._
