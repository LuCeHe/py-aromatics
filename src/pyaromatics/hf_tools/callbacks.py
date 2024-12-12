import time
from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState


class TimeStoppingCallback(TrainerCallback, ExportableState):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, max_seconds: float = 60 * 60 * 10):
        self.max_seconds = max_seconds
        self.initial_time = time.time()

    def time_stop(self, control):
        if time.time() - self.initial_time > self.max_seconds:
            print(f"Time limit reached: {time.time() - self.initial_time:.2f} > {self.max_seconds} seconds")
            control.should_training_stop = True
            control.should_prediction_stop = True
            control.should_epoch_stop = True
            control.should_log = False
            control.should_save = False
            control.should_evaluate = False

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
