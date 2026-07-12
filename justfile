# Machine Learning Zoo — Root Justfile
# https://github.com/casey/just
#
# Recipes are organised into per-domain sub-modules under tools/. Invoke a
# sub-module recipe directly (e.g. `just test::coverage`, `just run::sidecar`),
# or use the root shorthands below.

set shell := ["bash", "-c"]
set unstable := true

# --- Sub-module declarations (imported from tools/) ---

mod helper   "tools/helper/justfile"
mod dev      "tools/dev/justfile"
mod test     "tools/test/justfile"
mod quality  "tools/quality/justfile"
mod run      "tools/run/justfile"
mod docs     "tools/docs/justfile"
mod bench    "tools/bench/justfile"

# --- Default target ---

default: help

# List all commands across every sub-module
help:
    @just helper::help

# Project statistics
stats:
    @just helper::stats

# --- Setup & maintenance (→ tools/dev) ---

# Sync all dependencies
sync:
    @just dev::sync

# Install pre-commit hooks
hooks:
    @just dev::hooks

# Update dependencies
update:
    @just dev::update

# Remove caches and build artifacts
clean:
    @just dev::clean

# --- Quality (→ tools/quality) ---

# Format all code
fmt:
    @just quality::fmt

# Lint all code (CI-equivalent)
lint:
    @just quality::lint

# Auto-fix lint issues
fix:
    @just quality::fix

# Type-check
typecheck:
    @just quality::typecheck

# Audit dependencies
audit:
    @just quality::audit

# --- Testing (→ tools/test) ---
# Note: the bare `test` name is the sub-module; use `just test::test` or the
# shorthands below.

# Run the full test suite
test-run:
    @just test::test

# Coverage report
coverage:
    @just test::coverage

# Quality gate: lint + tests
check: lint test-run
    @echo "✅ Code quality check passed!"

# --- Run (→ tools/run) ---

# Run the mlzoo CLI
main *args:
    @just run::main {{args}}

# Run the ML sidecar
sidecar *args:
    @just run::sidecar {{args}}

# Start the inference API server
serve *args:
    @just run::serve {{args}}

# --- Docs & bench (distinct shorthand names; modules are docs/bench) ---

# Build the Sphinx docs
docs-build:
    @just docs::build

# Run benchmarks
bench-run *args:
    @just bench::run {{args}}
