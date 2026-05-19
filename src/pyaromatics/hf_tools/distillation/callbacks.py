import time
from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState


class TimeStoppingCallback(TrainerCallback, ExportableState):
    """A [`TrainerCallback`] that handles early stopping based on wall-clock time.

    Args:
        max_seconds (`float`):
            Maximum training time in seconds.
    """

    def __init__(self, max_seconds: float = 60 * 60 * 10):
        self.max_seconds = max_seconds
        self.initial_time = time.time()
        self.time_stopped = False

    def time_stop(self, control):
        if time.time() - self.initial_time > self.max_seconds:
            print(f"Time limit reached: {time.time() - self.initial_time:.2f} > {self.max_seconds} seconds")
            control.should_training_stop = True
            control.should_prediction_stop = True
            control.should_epoch_stop = True
            control.should_log = False
            control.should_save = False
            control.should_evaluate = False
            self.time_stopped = True

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.time_stop(control)

    def on_prediction_step(self, args, state, control, **kwargs):
        self.time_stop(control)

    def on_step_end(self, args, state, control, **kwargs):
        self.time_stop(control)

    def on_step_begin(self, args, state, control, **kwargs):
        self.time_stop(control)

    def on_substep_end(self, args, state, control, **kwargs):
        self.time_stop(control)

    def on_train_end(self, args, state, control, **kwargs):
        self.time_stop(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.time_stop(control)

    def state(self) -> dict:
        return {
            "args": {
                "max_seconds": self.max_seconds,
            },
            "attributes": {
                "initial_time": self.initial_time,
            },
        }


class GradientNormCallback(TrainerCallback):
    """Callback to track gradient norm during training."""

    def __init__(self):
        super().__init__()
        self.gradient_norm = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Compute gradient norm after optimizer step."""
        if model is not None:
            total_norm = 0.0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                self.gradient_norm = total_norm ** (1. / 2)
            else:
                self.gradient_norm = 0.0
        return control
