import solara

from application.logic import session_store as store


@solara.component
def LabelInput(session, evaluator, optimizer) -> None:
    """Digit buttons 0-9 for human labelling, plus a Skip button.

    Clicking a digit triggers learning + history update + next sample.
    Skip advances to next sample without any weight update.
    """
    auto = solara.use_reactive(store.auto_mode)
    busy = solara.use_reactive(store.is_busy)

    def on_label(digit: int) -> None:
        if auto.value or busy.value:
            return
        store.is_busy.value = True
        try:
            session.learn_with_label(digit, optimizer)
            accuracy, avg_loss = evaluator.evaluate()
            entry = {
                "samples_seen": store.samples_seen.value,
                "accuracy": accuracy,
                "avg_loss": avg_loss,
            }
            store.history.value = [*store.history.value, entry]
            store.status_message.value = (
                f"Labelled as {digit} | "
                f"acc={accuracy:.1%} | "
                f"loss={avg_loss:.4f}"
            )
        finally:
            store.is_busy.value = False

    def on_skip() -> None:
        if auto.value or busy.value:
            return
        session.skip_current()
        store.status_message.value = "Skipped — no learning."

    disabled = auto.value or busy.value

    with solara.Row():
        for digit in range(10):
            # Capture loop variable
            d = digit
            solara.Button(
                str(d),
                on_click=lambda d=d: on_label(d),
                disabled=disabled,
                style={"min-width": "42px", "font-size": "1.1rem"},
            )
        solara.Button(
            "Skip",
            on_click=on_skip,
            disabled=disabled,
            color="secondary",
            style={"margin-left": "12px"},
        )
