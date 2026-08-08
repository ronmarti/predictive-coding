import solara
import plotly.graph_objects as go

from application.logic import session_store as store


@solara.component
def AccuracyLossChart() -> None:
    """Dual-axis Plotly chart: accuracy (left) and loss (right) vs samples seen."""
    hist = solara.use_reactive(store.history)

    data = hist.value
    x = [d["samples_seen"] for d in data]
    accuracy = [d["accuracy"] * 100.0 for d in data]
    loss = [d["avg_loss"] for d in data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=accuracy,
        name="Accuracy (%)",
        mode="lines+markers",
        line={"color": "#1f77b4"},
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=loss,
        name="Loss",
        mode="lines+markers",
        line={"color": "#d62728", "dash": "dot"},
        yaxis="y2",
    ))
    fig.update_layout(
        xaxis={"title": "Samples seen"},
        yaxis={
            "title": {"text": "Accuracy (%)", "font": {"color": "#1f77b4"}},
            "range": [0, 100],
        },
        yaxis2={
            "title": {"text": "Avg loss", "font": {"color": "#d62728"}},
            "overlaying": "y",
            "side": "right",
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 60, "r": 60, "t": 30, "b": 50},
        height=300,
    )

    if not data:
        solara.Text(
            "No evaluations yet — label some digits or enable auto mode.",
            style={"color": "gray"},
        )
    else:
        solara.FigurePlotly(fig)
