# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A FastAPI WebSocket backend for real-time control system simulations. It serves as the backend for a Next.js frontend (acsl-simulator). Simulations run in an async loop and broadcast state to connected WebSocket clients at ~30fps.

## Commands

```bash
# Run the dev server (from project root)
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt
```

There are no tests or linters configured in this project.

## Architecture

### Simulator Registry Pattern

`app/simulators/__init__.py` defines `SIM_REGISTRY`, a dict mapping string keys to simulator factory lambdas:
- `"smd"` -> SpringMassDamperSimulator (dt=0.01)
- `"fp"` -> FurutaPendulumSimulator (dt=0.01)
- `"diffdrive"` -> DiffDriveSimulator (dt=0.02)

### Core Abstractions

- **`BaseSimulator`** (`app/simulators/base.py`): ABC that all simulators implement. Key methods: `reset()`, `step() -> SimState`, `apply_impulse()`, `set_params()`.
- **`SimulationManager`** (`app/simulators/manager.py`): Async wrapper that runs a simulator's `step()` in a loop, manages start/stop lifecycle, duration limits, and broadcasts state to listeners at `dt_broadcast` intervals (~1/30s).
- **`get_manager(sim_type)`** (`app/runtime.py`): Lazily creates and caches one `SimulationManager` per sim_type. All WebSocket clients for the same sim_type share a single manager.

### WebSocket Protocol

Single endpoint: `ws://.../ws/{sim_type}` (`app/routers/ws_sim.py`). Clients send JSON messages with a `type` field:
- `start`, `stop`, `reset` - simulation lifecycle
- `click` - apply impulse (force/torque)
- `set_params`, `set_control_params`, `set_estimator_params`, `set_reference`, `set_initial`, `set_exp_mode` - configuration

Server pushes: `state` (periodic sim state), `stopped` (when sim ends), `trace` (accumulated time-series data on stop).

### Simulator Structure (FP and SMD)

Each simulator (Furuta Pendulum, Spring-Mass-Damper) follows a PLANT/CONTROLLER/ESTIMATOR pattern:
- **CONTROLLER**: Strategy pattern via `set_control_params({"type": "lqr"|"pole_assignment"|"pid"|...})`. Implementations in `app/simulators/common/CONTROLLER/`.
- **ESTIMATOR**: Similar strategy pattern (`"observer"`, `"ekf"`). Implementations in `app/simulators/common/ESTIMATOR/`.
- Both support `time_mode`: `"continuous"` or `"discrete"`, affecting how system matrices (A,B,C,D vs Ad,Bd) are used.

### DiffDrive Simulator

Differential-drive robot with its own reference trajectory system (`app/simulators/diff_drive/reference.py`). Supports `"controller"` and `"external"` control modes.

## Key Conventions

- Comments and docstrings are in Japanese.
- Public state dataclasses (e.g., `PublicFPState`, `PublicSmdState`) are what get serialized to JSON and sent to the frontend via `__dict__`.
- Each simulator maintains a `_trace` dict for post-simulation time-series analysis.
- The `app/routers/params.py` REST router exists but the primary interface is the WebSocket router.
