"""SelfDistillationTrainer: task-only warmup, then freeze-and-GKD cycles with doubling phase length."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from transformers import TrainerCallback
from trl.experimental.gkd import GKDConfig, GKDTrainer

from pyaromatics.hf_tools.trainers import PlusTrainer, _ensure_tied_lm_head_after_checkpoint_load


class SelfDistillationTrainer(PlusTrainer):
    """Train with no external teacher: pure task loss first, then repeated self-distillation via GKD.

    Phase 0 runs for ``initial_restart_steps`` optimizer steps with standard causal-LM loss only.
    At each subsequent boundary the current student is frozen as the teacher, a fresh student is
    reset from the weights captured at ``train()`` start, and training continues with TRL
    ``GKDTrainer``. After every phase ``restart_steps`` doubles (100, 200, 400, ...).

    Total step budget comes from ``args.max_steps`` when set, otherwise from
    ``args.num_train_epochs`` and the dataloader length.
    """

    def __init__(
        self,
        task_data_collator=None,
        gkd_data_collator=None,
        initial_restart_steps: int = 100,
        gkd_args: GKDConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if getattr(self.args, "auto_find_batch_size", False):
            print(
                "[SelfDistillationTrainer] auto_find_batch_size=True desyncs DDP; forcing False."
            )
            self.args.auto_find_batch_size = False

        self.task_data_collator = task_data_collator or self.data_collator
        self.gkd_data_collator = gkd_data_collator or self.data_collator
        self.initial_restart_steps = max(1, int(initial_restart_steps))
        self.gkd_args = gkd_args
        self.teacher_model: nn.Module | None = None
        self._initial_student_state: dict[str, torch.Tensor] | None = None
        self._phase_index = 0
        self._current_restart_steps = self.initial_restart_steps

    @staticmethod
    def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _capture_initial_student_state(self) -> None:
        self._initial_student_state = self._clone_state_dict(self.model)

    def _reset_student_from_initial_state(self) -> None:
        if self._initial_student_state is None:
            raise RuntimeError("Initial student state was not captured before restart.")
        self.model.load_state_dict(self._initial_student_state)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        _ensure_tied_lm_head_after_checkpoint_load(self.model)

    def _promote_student_to_teacher(self) -> None:
        if self.teacher_model is not None:
            del self.teacher_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _phase_callbacks_list(self) -> list[TrainerCallback]:
        """Callbacks registered on this trainer (HF uses ``callback_handler``, not ``callbacks``)."""
        handler = getattr(self, "callback_handler", None)
        if handler is not None and hasattr(handler, "callbacks"):
            return list(handler.callbacks)
        return []

    def _resolve_total_step_budget(self) -> int:
        if self.args.max_steps and self.args.max_steps > 0:
            return int(self.args.max_steps)

        n_gpu = getattr(self.args, "n_gpu", 0) or (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        batch_size = (
            self.args.per_device_train_batch_size
            * max(1, n_gpu)
            * getattr(self.args, "gradient_accumulation_steps", 1)
        )
        num_examples = len(self.train_dataset)
        steps_per_epoch = max(1, num_examples // batch_size)
        return max(1, int(self.args.num_train_epochs * steps_per_epoch))

    def _build_gkd_config(self) -> GKDConfig:
        if self.gkd_args is not None:
            return copy.deepcopy(self.gkd_args)
        if isinstance(self.args, GKDConfig):
            return copy.deepcopy(self.args)
        return GKDConfig(**self.args.to_dict())

    def _shared_trainer_kwargs(self, callbacks: list[TrainerCallback] | None) -> dict[str, Any]:
        return {
            "model": self.model,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "processing_class": self.processing_class,
            "callbacks": callbacks,
            "tokenizer": getattr(self, "tokenizer", None),
        }

    def _run_task_phase(
        self,
        phase_steps: int,
        callbacks: list[TrainerCallback] | None,
        resume_from_checkpoint,
        trial,
        ignore_keys_for_eval,
        train_kwargs,
    ):
        print(
            f"[SelfDistillationTrainer] Phase {self._phase_index}: "
            f"task-only training for {phase_steps} steps"
        )
        original_max_steps = self.args.max_steps
        original_num_train_epochs = self.args.num_train_epochs
        self.args.max_steps = phase_steps
        self.args.num_train_epochs = 1
        self.data_collator = self.task_data_collator

        try:
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **train_kwargs,
            )
        finally:
            self.args.max_steps = original_max_steps
            self.args.num_train_epochs = original_num_train_epochs

    def _run_gkd_phase(
        self,
        phase_steps: int,
        callbacks: list[TrainerCallback] | None,
        ignore_keys_for_eval,
        train_kwargs,
    ):
        if self.teacher_model is None:
            raise RuntimeError("GKD phase requires a frozen teacher model.")

        gkd_config = self._build_gkd_config()
        original_max_steps = gkd_config.max_steps
        original_num_train_epochs = gkd_config.num_train_epochs
        gkd_config.max_steps = phase_steps
        gkd_config.num_train_epochs = 1

        print(
            f"[SelfDistillationTrainer] Phase {self._phase_index}: "
            f"GKD from frozen self-teacher for {phase_steps} steps "
            f"(next restart_steps={self._current_restart_steps * 2})"
        )

        trainer = GKDTrainer(
            teacher_model=self.teacher_model,
            args=gkd_config,
            data_collator=self.gkd_data_collator,
            **self._shared_trainer_kwargs(callbacks),
        )
        try:
            output = trainer.train(
                resume_from_checkpoint=None,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **train_kwargs,
            )
        finally:
            gkd_config.max_steps = original_max_steps
            gkd_config.num_train_epochs = original_num_train_epochs

        self.model = trainer.model
        self.state = trainer.state
        return output

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        if resume_from_checkpoint is not None:
            print(
                "[SelfDistillationTrainer] resume_from_checkpoint is not supported; "
                "starting self-distillation schedule from scratch."
            )

        self._capture_initial_student_state()
        total_budget = self._resolve_total_step_budget()
        original_max_steps = self.args.max_steps
        original_num_train_epochs = self.args.num_train_epochs

        print(
            f"[SelfDistillationTrainer] Starting schedule: initial_restart_steps="
            f"{self.initial_restart_steps}, total_step_budget={total_budget}"
        )

        steps_completed = 0
        final_output = None
        self._phase_index = 0
        self._current_restart_steps = self.initial_restart_steps

        while steps_completed < total_budget:
            phase_steps = min(self._current_restart_steps, total_budget - steps_completed)
            phase_callbacks = self._phase_callbacks_list()

            if self._phase_index == 0:
                phase_output = self._run_task_phase(
                    phase_steps=phase_steps,
                    callbacks=phase_callbacks,
                    resume_from_checkpoint=None,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    train_kwargs=kwargs,
                )
            else:
                self._promote_student_to_teacher()
                self._reset_student_from_initial_state()
                phase_output = self._run_gkd_phase(
                    phase_steps=phase_steps,
                    callbacks=phase_callbacks,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    train_kwargs=kwargs,
                )

            final_output = phase_output
            steps_completed += phase_steps
            self._phase_index += 1
            self._current_restart_steps *= 2

            print(
                f"[SelfDistillationTrainer] Phase {self._phase_index - 1} complete: "
                f"{steps_completed}/{total_budget} steps done."
            )

        self.args.max_steps = original_max_steps
        self.args.num_train_epochs = original_num_train_epochs
        print("[SelfDistillationTrainer] Self-distillation schedule complete.")
        return final_output
