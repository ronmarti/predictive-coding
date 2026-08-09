# Predictive Coding — MNIST

A toy implementation of a **parametrically scalable predictive coding (PC) network** applied to MNIST handwritten digit recognition, served as a **Solara reactive web app** with human-in-the-loop labelling and live Plotly learning curves.

Unlike backpropagation-trained networks, a predictive coding network has **no separate training phase**. It learns continuously as it processes each sample by minimising prediction errors at every layer — a biologically plausible algorithm grounded in the *free energy principle*.

---

## Table of Contents

- [How predictive coding works](#how-predictive-coding-works)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Web UI walkthrough](#web-ui-walkthrough)
- [Configuration parameters](#configuration-parameters)
- [Project structure](#project-structure)
- [Saving and loading weights](#saving-and-loading-weights)
- [Running without Docker](#running-without-docker)

---

## How predictive coding works

A PC network with $L$ layers maintains **latent state vectors** $\mu_l$ at each layer. Learning happens in two alternating phases per sample:

### 1 — Inference (no weight changes)

The network runs $T$ steps of gradient descent on the latent states $\mu_l$, holding the input $\mu_0$ and target $\mu_L$ clamped:

$$\epsilon_l = \mu_l - f(W_{l-1}\,\mu_{l-1})$$

$$\mu_l \leftarrow \mu_l - \gamma_\mu \left(\epsilon_l - W_l^\top (\epsilon_{l+1} \odot f')\right)$$

### 2 — Weight update (local Hebbian rule)

After the latent states converge, weights are updated using only locally available quantities — no backpropagation graph needed:

$$\Delta W_l = \mu_l^\top \cdot (\epsilon_{l+1} \odot f'(W_l\,\mu_l))$$

### Key properties

| Property | Backpropagation | Predictive coding |
|---|---|---|
| Credit assignment | Global (chain rule) | Local (layer-wise errors) |
| Weight update timing | After full forward+backward pass | After local inference convergence |
| Continual learning | Catastrophic forgetting | Naturally resistant |
| Biologically plausible | No | Yes |
| Few-shot learning | Weak | Strong |

---

## Architecture

The network is a stack of fully-connected layers. The architecture is controlled entirely by the `NODES` parameter:

```
NODES=784,300,100,10
        │    │    │  └─ output (10 digit classes)
        │    └────┘ hidden layers (any depth/width)
        └─ input (28×28 MNIST = 784 pixels)
```

All hidden layers use the configured activation function (`ReLU` or `Tanh`). The output layer always uses a linear activation so that prediction errors on the labels are smooth.

**To make the network deeper**, add more values:
```
NODES=784,512,256,128,64,10
```

**To make it wider**, increase hidden layer values:
```
NODES=784,1024,512,10
```

---

## Getting started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the **NVIDIA Container Toolkit** installed
- An NVIDIA GPU accessible to Docker (`docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` should succeed)

### 1 — Clone and configure

```bash
git clone https://github.com/ronmarti/predictive-coding.git
cd predictive-coding
```

Edit `.env` to tune the network (see [Configuration parameters](#configuration-parameters)). The defaults work well out of the box.

### 2 — Build and run

```bash
docker compose up --build
```

The first run downloads MNIST (~11 MB) and the PyTorch CUDA wheels (~2 GB). Subsequent starts are fast.

### 3 — Open the web app

Navigate to **[http://localhost:8765](http://localhost:8765)** in your browser.

---

## Web UI walkthrough

### Current digit panel

Displays the randomly sampled MNIST digit the network is currently looking at. Next to it, a horizontal bar chart shows the network's **confidence for each digit class (0–9)**. The predicted class is highlighted in red.

### Label it panel

#### Manual mode (default)

Click any of the **0–9 buttons** to tell the network the correct label. This triggers:
1. Full inference loop — latent states $\mu_l$ are updated for $T$ iterations with both input and true label clamped.
2. Hebbian weight update — weights shift to reduce prediction error.
3. Test-set evaluation — accuracy and loss are computed and appended to the chart.
4. Next digit — a new random sample is loaded.

Click **Skip** to advance to the next digit without any learning. The chart is not updated.

#### Auto mode

Check **"Auto mode (remove human from loop)"** to start a background thread that:
- Continuously samples random digits from the training set
- Supplies the true label automatically (no human needed)
- Runs inference + weight update on every sample
- Evaluates the test set every `EVAL_INTERVAL` samples and updates the chart

Uncheck the box at any time to stop and return to manual mode with a fresh digit ready.

### Learning progress chart

A live Plotly chart with two y-axes:
- **Blue line** — test-set accuracy (%) on the left axis
- **Red dashed line** — average cross-entropy loss on the right axis

Both are plotted against **samples seen** (not epochs — the model never repeats in order). The upward accuracy trend and downward loss trend demonstrate continual learning in action.

### Weights panel

See [Saving and loading weights](#saving-and-loading-weights).

### Hyperparameters panel

A collapsible card at the bottom of the page exposes the four runtime-tunable parameters. Changes take effect **immediately** on the live network — no container restart required.

| Control | What it does |
|---|---|
| **Auto batch size** | Number of training samples processed together in one weight update during auto mode. Larger values give a more stable Hebbian gradient estimate and better GPU utilisation. Has no effect in manual (human-in-the-loop) mode where each sample is processed one at a time. |
| **Inference iterations T** | How many gradient-descent steps the inference loop takes on the latent states $\mu_l$ before applying a weight update. More iterations bring the prediction errors closer to zero, producing a more accurate gradient, but reduce throughput. Think of this as the "inner loop" convergence budget. |
| **Inference step size μ_dt** | The learning rate of the *inner* inference loop — how far each latent state $\mu_l$ moves per iteration toward minimising its layer's prediction error. Larger values converge faster but can overshoot and oscillate. This is distinct from the outer weight learning rate. |
| **Learning rate** | The Adam step size applied to synaptic weights *after* inference has converged. This is the "outer" learning rate. Spans multiple orders of magnitude so a numeric text field is used rather than a slider. |

These four parameters can also be set as defaults in `.env` (see [Configuration parameters](#configuration-parameters)) so they take effect from the first sample without manual adjustment in the UI.

---

## Configuration parameters

All parameters are set in `.env`. Restart the container (`docker compose up`) to apply `.env` changes. The four marked ★ can also be changed live in the **Hyperparameters** panel without a restart.

| Variable | Default | ★ | Description |
|---|---|---|---|
| `NODES` | `784,300,100,10` | | Comma-separated layer widths. First must be 784 (MNIST input), last must be 10 (digit classes). Add/remove values to change depth. |
| `MU_DT` | `0.01` | ★ | Inference step size $\gamma_\mu$. Smaller = more stable but slower convergence per sample. |
| `N_INFER_ITERS` | `50` | ★ | Number of inference loop iterations $T$ per sample. More iterations = better gradient estimate for weight update, but slower. |
| `LR` | `0.0001` | ★ | Adam learning rate for weight updates. |
| `AUTO_BATCH_SIZE` | `64` | ★ | Mini-batch size used in auto mode. Larger values improve GPU utilisation. Has no effect in manual mode. |
| `ACTIVATION` | `relu` | | Hidden layer activation. Options: `relu`, `tanh`. |
| `EVAL_INTERVAL` | `100` | | Auto mode only: number of *samples* (not batches) between full test-set evaluations. |
| `DEVICE` | `cuda` | | Compute device. Falls back to `cpu` automatically if CUDA is unavailable. |
| `LOG_LEVEL` | `INFO` | | Python logging level: `DEBUG`, `INFO`, `WARNING`. |

### Recommended experiments

| Goal | Change |
|---|---|
| Shallow vs deep comparison | Run with `NODES=784,10`, then `NODES=784,300,100,10` |
| Faster auto mode | Reduce `N_INFER_ITERS` to 20 and `EVAL_INTERVAL` to 50 |
| More stable learning | Increase `N_INFER_ITERS` to 100, reduce `LR` to 5e-5 |
| Tanh dynamics | Set `ACTIVATION=tanh` |
| Better GPU utilisation | Increase `AUTO_BATCH_SIZE` to 128 or 256 |

---

## Project structure

```
predictive-coding/
├── Dockerfile                    # CUDA base image, installs deps, runs solara
├── docker-compose.yml            # GPU passthrough, port 8765, volume mounts
├── pyproject.toml                # Python dependencies
├── .env                          # Runtime configuration (edit this)
├── outputs/                      # Saved weight files (.safetensors) — Docker-mounted
└── application/
    ├── app.py                    # Solara Page() — top-level UI layout
    ├── config.py                 # Config dataclass reading from env vars
    ├── __main__.py               # Local debug fallback entry point
    ├── models/
    │   ├── activations.py        # ReLU, Tanh, LinearActivation + factory
    │   ├── pc_layer.py           # PcLayer — raw tensor weights, Hebbian grad
    │   ├── pc_network.py         # PcNetwork — inference loop + weight update
    │   └── pc_optimizer.py       # AdamOptimizer on layer.grad dicts (no autograd)
    ├── logic/
    │   ├── mnist_dataset.py      # MnistDataset — random_sample, test_loader
    │   ├── session_store.py      # SessionStore + all solara.reactive state
    │   ├── auto_runner.py        # AutoRunner — background learning thread
    │   └── evaluator.py          # Evaluator — test-set accuracy + loss
    ├── ui/
    │   ├── digit_display.py      # Digit image + confidence bar chart
    │   ├── label_input.py        # 0–9 label buttons + Skip
    │   ├── auto_mode_panel.py    # Auto mode checkbox
    │   ├── accuracy_loss_chart.py # Live Plotly dual-axis chart
    │   ├── weights_panel.py      # Save/Load weight files (safetensors)
    │   └── hyperparams_panel.py  # Live-editable T, μ_dt, lr, batch size
    └── utils/
        └── logging_config.py     # Structured logging setup
```

### Key design choices

- **No autograd** — weights are plain `torch.Tensor`, not `nn.Parameter`. Gradients are computed manually from prediction errors (the Hebbian rule). `torch.autograd` is never invoked.
- **Reactive UI** — all shared state lives in `application/logic/session_store.py` as `solara.reactive` variables. UI components subscribe implicitly and re-render on change.
- **Thread safety** — the auto runner runs in a `daemon` thread. Solara reactive variables are thread-safe for assignment; the thread signals shutdown by checking `store.auto_mode.value`.
- **Parametric scaling** — the entire architecture is determined by the `NODES` list. Change one line in `.env` to get a completely different network.

---

## Saving and loading weights

### Save

Click the **Save** button. The current weights are written to:

```
outputs/weights_YYYYMMDD_HHMMSS.pt
```

This directory is Docker-mounted at `./outputs` on the host, so files survive container restarts.

### Load

1. Click **Load…** to open the file browser pointing at `/outputs`.
2. Click a `.pt` file to select it.
3. Click **Load selected**. The weights are restored and inference runs immediately on the current digit.

Weight files are standard PyTorch state dicts (`torch.save` / `torch.load`). They are architecture-specific: a file saved with `NODES=784,300,100,10` cannot be loaded into a `NODES=784,10` network.

---

## Running without Docker

Requires Python 3.11+ and a local PyTorch installation with CUDA support.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install solara plotly anywidget matplotlib numpy

# set env vars (copy from .env and export them, or use a tool like direnv)
export NODES=784,300,100,10
export DEVICE=cuda
# ... other vars ...

solara run application/app.py --host 0.0.0.0 --port 8765
```

Or via the package entry point:

```bash
python -m application
```

