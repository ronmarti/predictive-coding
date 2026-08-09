import solara

_HINT_STYLE = {"color": "gray", "font-size": "0.78rem", "margin-top": "-6px",
               "margin-bottom": "6px"}


@solara.component
def HyperparamsPanel(session, optimizer, auto_runner) -> None:
    """Live-editable hyperparameter controls; changes apply immediately."""
    lr, set_lr = solara.use_state(float(optimizer._lr))
    batch_size, set_batch_size = solara.use_state(int(auto_runner._batch_size))
    n_iters, set_n_iters = solara.use_state(int(session._n_infer_iters))
    mu_dt, set_mu_dt = solara.use_state(float(session._mu_dt))

    def on_lr(value: float) -> None:
        clamped = max(1e-6, min(float(value), 1.0))
        optimizer._lr = clamped
        set_lr(clamped)

    def on_batch_size(value: int) -> None:
        clamped = max(1, min(int(value), 512))
        auto_runner._batch_size = clamped
        set_batch_size(clamped)

    def on_n_iters(value: int) -> None:
        clamped = max(5, min(int(value), 500))
        session._n_infer_iters = clamped
        set_n_iters(clamped)

    def on_mu_dt(value: float) -> None:
        clamped = max(1e-4, min(float(value), 0.5))
        session._mu_dt = clamped
        set_mu_dt(clamped)

    with solara.Card("Hyperparameters"):
        with solara.Column():
            solara.SliderInt(
                "Auto batch size",
                value=batch_size,
                min=1,
                max=512,
                on_value=on_batch_size,
            )
            solara.Text(
                "Number of samples processed per weight update in auto mode. "
                "Larger batches give a more stable gradient estimate and better "
                "GPU utilisation, but reduce the number of weight updates per "
                "unit time.",
                style=_HINT_STYLE,
            )

            solara.SliderInt(
                "Inference iterations (T)",
                value=n_iters,
                min=5,
                max=500,
                on_value=on_n_iters,
            )
            solara.Text(
                "How many gradient-descent steps the network takes on its "
                "internal latent states μ before applying a weight update. "
                "More iterations → errors converge closer to zero → more "
                "accurate Hebbian gradient, but slower throughput. "
                "Typical range: 20–100.",
                style=_HINT_STYLE,
            )

            solara.SliderFloat(
                "Inference step size (μ_dt)",
                value=mu_dt,
                min=0.0001,
                max=0.5,
                step=0.0001,
                on_value=on_mu_dt,
            )
            solara.Text(
                "Learning rate for the inner inference loop that updates "
                "latent states μ. Larger values converge faster but can "
                "overshoot; smaller values are more stable. "
                "Typical range: 0.005–0.05.",
                style=_HINT_STYLE,
            )

            # Learning rate spans orders of magnitude — use a text input
            solara.InputFloat(
                "Learning rate",
                value=lr,
                on_value=on_lr,
            )
            solara.Text(
                "Adam step size applied to synaptic weights after each "
                "inference convergence. This is the outer learning rate "
                "(the inner one is μ_dt above). Spans many orders of "
                "magnitude so a text field is used instead of a slider. "
                "Typical range: 1e-5 – 1e-3.",
                style=_HINT_STYLE,
            )
