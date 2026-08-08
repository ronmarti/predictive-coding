import datetime
from pathlib import Path

import solara
from safetensors.torch import load_file, save_file

from application.logic import session_store as store


OUTPUTS_DIR = Path("/outputs")


@solara.component
def WeightsPanel(network, session) -> None:
    """Save and load network weights."""
    selected_path = solara.use_reactive(store.weights_file_path)
    show_browser, set_show_browser = solara.use_state(False)
    status = solara.use_reactive(store.status_message)

    def on_save() -> None:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTPUTS_DIR / f"weights_{ts}.safetensors"
        flat = {
            f"layer_{i}_{k}": v.contiguous().cpu()
            for i, d in enumerate(network.state_dict())
            for k, v in d.items()
        }
        save_file(flat, path)
        store.status_message.value = f"Saved \u2192 {path.name}"

    def on_load() -> None:
        path = selected_path.value
        if path is None or not path.is_file():
            store.status_message.value = "No file selected."
            return
        flat = load_file(path, device=str(network._device))
        i = 0
        states = []
        while f"layer_{i}_weights" in flat:
            states.append({
                "weights": flat[f"layer_{i}_weights"],
                "biases": flat[f"layer_{i}_biases"],
            })
            i += 1
        network.load_state_dict(states)
        session.advance_sample()
        store.status_message.value = f"Loaded \u2190 {path.name}"
        set_show_browser(False)

    def on_file_selected(path: Path) -> None:
        store.weights_file_path.value = path

    with solara.Card("Weights"):
        with solara.Row():
            solara.Button("Save", on_click=on_save, color="primary")
            solara.Button(
                "Load\u2026",
                on_click=lambda: set_show_browser(not show_browser),
                color="secondary",
            )

        if show_browser:
            solara.FileBrowser(
                directory=OUTPUTS_DIR,
                can_select=True,
                on_path_select=on_file_selected,
                filter=lambda p: p.is_dir() or p.suffix == ".safetensors",
            )
            if selected_path.value is not None:
                solara.Text(f"Selected: {selected_path.value.name}")
                solara.Button("Load selected", on_click=on_load, color="success")
