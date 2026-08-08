import solara

from application.logic import session_store as store


@solara.component
def AutoModePanel(session, auto_runner) -> None:
    """Checkbox that toggles auto (background) learning mode."""
    auto = solara.use_reactive(store.auto_mode)

    def on_toggle(value: bool) -> None:
        store.auto_mode.value = value
        if value:
            auto_runner.start()
            store.status_message.value = "Auto mode active — learning…"
        else:
            auto_runner.stop()
            # Load a fresh sample so manual controls have something to show
            session.advance_sample()
            store.status_message.value = "Auto mode stopped."

    with solara.Row(style={"align-items": "center", "gap": "12px"}):
        solara.Checkbox(
            label="Auto mode (remove human from loop)",
            value=auto,
            on_value=on_toggle,
        )
