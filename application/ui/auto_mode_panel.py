import solara

from application.logic import session_store as store


@solara.component
def AutoModePanel(session, auto_runner, optimizer) -> None:
    """Checkbox that toggles auto (background) learning mode."""
    auto = solara.use_reactive(store.auto_mode)

    def on_toggle(value: bool) -> None:
        store.auto_mode.value = value
        if value:
            auto_runner.start()
            store.status_message.value = "Auto mode active — learning…"
        else:
            auto_runner.stop()
            # Reset Adam moments: auto mode uses batch-averaged gradients
            # (scale ~1/64); manual mode uses raw single-sample gradients
            # (~64x larger). Without a reset, Adam applies a mismatched
            # huge step and overwrites learned weights after just one label.
            optimizer.reset()
            session.advance_sample()
            store.status_message.value = "Auto mode stopped."

    with solara.Row(style={"align-items": "center", "gap": "12px"}):
        solara.Checkbox(
            label="Auto mode (remove human from loop)",
            value=auto,
            on_value=on_toggle,
        )
