import logging
import threading

from application.logic import session_store as store
from application.logic.evaluator import Evaluator

logger = logging.getLogger(__name__)


class AutoRunner:
    """Runs the PC learning loop in a background thread.

    When active, samples a random digit, runs inference with the true label,
    updates weights, and periodically evaluates test-set accuracy.
    """

    def __init__(
        self,
        session: "store.SessionStore",
        evaluator: Evaluator,
        optimizer,
        eval_interval: int,
        batch_size: int = 1,
    ) -> None:
        self._session = session
        self._evaluator = evaluator
        self._optimizer = optimizer
        self._eval_interval = eval_interval
        self._batch_size = batch_size
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background learning thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AutoRunner"
        )
        self._thread.start()
        logger.info("AutoRunner started.")

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("AutoRunner stopped.")

    def is_running(self) -> bool:
        """Return True if the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Background thread main loop."""
        since_eval = 0
        since_display = 0
        # Update the displayed digit every N samples to avoid reactive overhead
        DISPLAY_INTERVAL = 20

        while not self._stop_event.is_set():
            try:
                if not store.auto_mode.value:
                    self._stop_event.set()
                    break

                self._session.learn_batch(self._batch_size, self._optimizer)
                since_eval += self._batch_size
                since_display += self._batch_size

                if since_display >= DISPLAY_INTERVAL * self._batch_size:
                    since_display = 0
                    self._session.advance_sample()

                if since_eval >= self._eval_interval:
                    since_eval = 0
                    accuracy, avg_loss = self._evaluator.evaluate()
                    entry = {
                        "samples_seen": store.samples_seen.value,
                        "accuracy": accuracy,
                        "avg_loss": avg_loss,
                    }
                    # Reassign to trigger reactive re-render
                    store.history.value = [*store.history.value, entry]
                    logger.info(
                        "samples=%d  acc=%.3f  loss=%.4f",
                        store.samples_seen.value, accuracy, avg_loss,
                    )
            except Exception:
                logger.exception("Error in AutoRunner loop.")
                self._stop_event.set()
