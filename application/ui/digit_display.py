from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import solara

from application.logic import session_store as store


@solara.component
def DigitDisplay() -> None:
    """Shows the current MNIST digit and a horizontal confidence bar chart."""
    # All hooks must be called unconditionally on every render
    image = solara.use_reactive(store.current_image)
    scores = solara.use_reactive(store.confidence_scores)
    pred = solara.use_reactive(store.predicted_label)
    truth = solara.use_reactive(store.true_label)
    status = solara.use_reactive(store.status_message)

    def make_figure() -> Optional[plt.Figure]:
        if image.value is None:
            return None
        fig, axes = plt.subplots(
            1, 2, figsize=(7, 3),
            gridspec_kw={"width_ratios": [1, 2]},
        )
        # Left: digit image
        axes[0].imshow(image.value, cmap="gray", interpolation="nearest")
        axes[0].axis("off")
        label_txt = (
            f"True: {truth.value}" if truth.value >= 0 else ""
        )
        axes[0].set_title(label_txt, fontsize=11)

        # Right: confidence bars
        digit_labels = [str(d) for d in range(10)]
        colours = [
            "#1f77b4" if i != pred.value else "#d62728"
            for i in range(10)
        ]
        axes[1].barh(digit_labels, scores.value, color=colours)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Confidence")
        axes[1].set_title(
            f"Prediction: {pred.value}" if pred.value >= 0 else "—",
            fontsize=11,
        )
        fig.tight_layout()
        return fig

    fig = make_figure()
    if fig is not None:
        solara.FigureMatplotlib(fig)
        plt.close(fig)
    else:
        solara.Text(status.value, style={"color": "gray", "font-size": "1rem"})
