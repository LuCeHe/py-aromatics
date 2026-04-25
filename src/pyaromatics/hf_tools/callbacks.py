import time
import numpy as np
import torch
import torch.distributed as dist

from transformers import TrainerCallback, IntervalStrategy, logging
from transformers.trainer_callback import ExportableState

logger = logging.get_logger(__name__)


class EarlyStoppingCallback(TrainerCallback, ExportableState):

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        self.early_stopped = False

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
                args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
                args.eval_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
            self.early_stopped = True

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }


class TimeStoppingCallback(TrainerCallback, ExportableState):
    """
    Stops training when a wall-time budget is exceeded.

    In distributed (DDP/Accelerate) runs, the stop condition is **synchronized** across processes:
    each process shares the same `initial_time` (broadcast from rank 0 at train start) and uses the
    **maximum** elapsed time across ranks before comparing to `max_seconds`. Otherwise ranks can
    disagree by milliseconds (clock skew) or process startup delay and exit the step loop on
    different steps, which deadlocks NCCL collectives.
    """

    def __init__(self, max_seconds: float = 60 * 60 * 10):
        self.max_seconds = max_seconds
        self.initial_time = time.time()
        self.time_stopped = False

    @staticmethod
    def _device_for_comm(args) -> torch.device:
        if torch.cuda.is_available():
            if args is not None and getattr(args, "local_rank", -1) is not None and int(getattr(args, "local_rank", -1)) >= 0:
                return torch.device(f"cuda:{int(args.local_rank)}")
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def on_train_begin(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            d = self._device_for_comm(args)
            if dist.get_rank() == 0:
                t0 = time.time()
            else:
                t0 = 0.0
            t_tensor = torch.tensor([t0], device=d, dtype=torch.float64)
            dist.broadcast(t_tensor, src=0)
            self.initial_time = float(t_tensor.item())

    def _max_elapsed_seconds(self, args) -> float:
        local_elapsed = time.time() - self.initial_time
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return local_elapsed
        d = self._device_for_comm(args)
        t = torch.tensor([local_elapsed], device=d, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def time_stop(self, args, state, control):
        if self.time_stopped:
            return
        elapsed = self._max_elapsed_seconds(args)
        if elapsed <= self.max_seconds:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"Time limit reached: {elapsed:.2f} > {self.max_seconds} seconds (distributed sync: max across ranks)"
            )
        self.time_stopped = True
        control.should_training_stop = True
        control.should_prediction_stop = True
        control.should_epoch_stop = True
        control.should_log = False
        control.should_save = False
        control.should_evaluate = False

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.time_stop(args, state, control)

    def on_prediction_step(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def on_step_end(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def on_step_begin(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def on_substep_end(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def on_train_end(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.time_stop(args, state, control)

    def state(self) -> dict:
        return {
            "args": {
                "max_seconds": self.max_seconds,
            },
            "attributes": {
                "initial_time": self.initial_time,
            },
        }






class EmaEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0,
                 ema_lifetime: int = 5):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.ema_lifetime = ema_lifetime
        self.ema_best = None

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less

        if not state.best_metric is None:
            if self.ema_best is None:
                self.ema_best = state.best_metric

            alpha = 2 / (self.ema_lifetime + 1)
            self.ema_best = alpha * metric_value + (1 - alpha) * self.ema_best

        if state.best_metric is None or (
                operator(metric_value, self.ema_best)
                and abs(metric_value - self.ema_best) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1



    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
                "ema_lifetime": self.ema_lifetime,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }
