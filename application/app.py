"""Solara application entry point.

Run with:  solara run application/app.py --host 0.0.0.0 --port 8765
"""
import logging

import solara
import torch

from application.config import Config
from application.logic.auto_runner import AutoRunner
from application.logic.evaluator import Evaluator
from application.logic.mnist_dataset import MnistDataset
from application.logic import session_store as store
from application.logic.session_store import SessionStore
from application.models.activations import build_activation
from application.models.pc_network import PcNetwork
from application.models.pc_optimizer import AdamOptimizer
from application.ui.accuracy_loss_chart import AccuracyLossChart
from application.ui.auto_mode_panel import AutoModePanel
from application.ui.digit_display import DigitDisplay
from application.ui.hyperparams_panel import HyperparamsPanel
from application.ui.label_input import LabelInput
from application.ui.weights_panel import WeightsPanel
from application.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous init — runs once before Solara accepts any browser connections.
# CUDA kernels and MNIST are cached after the first run, so this is fast.
# ---------------------------------------------------------------------------

_cfg = Config.from_env()
_device = torch.device(_cfg.device if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", _device)

store.status_message.value = (
    "Initialising CUDA and building network"
    " (first run compiles CUDA kernels — may take a few minutes)…"
)
_activation = build_activation(_cfg.activation)
_network = PcNetwork(_cfg.nodes, _activation, _device)
_optimizer = AdamOptimizer(lr=_cfg.lr)
logger.info("Network ready on %s.", _device)

store.status_message.value = "Loading MNIST dataset…"
_dataset = MnistDataset(_device)
logger.info("Dataset ready.")

_session = SessionStore(
    _network, _dataset, _device, _cfg.n_infer_iters, _cfg.mu_dt
)
_evaluator = Evaluator(_network, _dataset, _device)
_auto_runner = AutoRunner(
    _session, _evaluator, _optimizer, _cfg.eval_interval,
    batch_size=_cfg.auto_batch_size,
)

_session.advance_sample()
store.status_message.value = (
    f"Ready \u2014 device: {_device} | architecture: {_cfg.nodes}"
)
logger.info("App ready.")


# ---------------------------------------------------------------------------
# Solara Page component
# ---------------------------------------------------------------------------

@solara.component
def Page() -> None:
    """Root page: digit display, label input, auto toggle, chart, weights."""
    solara.use_reactive(store.current_image)

    def ensure_first_sample():
        """Runs once after first render to populate image if still empty."""
        if store.current_image.value is None and _session is not None:
            _session.advance_sample()

    solara.use_effect(ensure_first_sample, [])
    with solara.Column(style={"max-width": "900px", "margin": "0 auto",
                               "padding": "16px", "gap": "16px"}):
        solara.Title("Predictive Coding \u2014 MNIST")

        with solara.Card("Current digit"):
            DigitDisplay()

        with solara.Card("Label it"):
            LabelInput(
                session=_session,
                evaluator=_evaluator,
                optimizer=_optimizer,
            )
            AutoModePanel(
                session=_session,
                auto_runner=_auto_runner,
                optimizer=_optimizer,
            )

        with solara.Card("Learning progress"):
            AccuracyLossChart()

        WeightsPanel(network=_network, optimizer=_optimizer, session=_session)

        HyperparamsPanel(
            session=_session,
            optimizer=_optimizer,
            auto_runner=_auto_runner,
        )
