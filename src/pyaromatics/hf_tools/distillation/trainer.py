"""DistillationTrainer: Knowledge distillation via KL divergence with alternating AdaLoRA/full training."""

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForCausalLM,
    Trainer,
)
from peft import AdaLoraConfig, get_peft_model, TaskType, PeftModel

from pyaromatics.hf_tools.distillation.helpers import enhance_logs_with_custom_metrics


# PEFT rejects these for Mamba-based models (see peft.tuners.tuners_utils._check_lora_target_modules_mamba)
MAMBA_MODEL_TYPES = {"falcon_h1", "mamba", "mamba2", "falcon_mamba"}
MAMBA_INCOMPATIBLE_MODULES = {"out_proj", "conv1d"}


def infer_adalora_target_modules(model: nn.Module) -> list[str]:
    """Infer AdaLoRA target_modules from the model by finding nn.Linear layers with 2D weight matrices."""
    skip_substrings = ('embed_tokens', 'embed_out', 'lm_head', 'wte', 'wpe', 'embedding')
    model_type = getattr(getattr(model, 'config', None), 'model_type', None) or ''
    is_mamba = model_type in MAMBA_MODEL_TYPES
    seen: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = getattr(module, 'weight', None)
        if weight is None or weight.dim() != 2:
            continue
        if any(s in name.lower() for s in skip_substrings):
            continue
        last_part = name.split('.')[-1]
        if is_mamba and last_part in MAMBA_INCOMPATIBLE_MODULES:
            continue
        seen.add(last_part)
    seen = sorted(seen)
    print("Inferred AdaLoRA target_modules:", seen)
    return seen


class DistillationTrainer(Trainer):
    """Custom Trainer for knowledge distillation with alternating AdaLoRA/full training."""

    def __init__(
            self,
            teacher_model: AutoModelForCausalLM | None = None,
            temperature: float = 3.0,
            alpha: float = 0.5,
            training_max_length: int | None = None,
            peft_type: Literal["adalora_full", "adalora_alternating"] | None = None,
            adalora_samples_per_phase: int = 1000,
            adalora_config: dict | None = None,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()

        self.temperature = temperature
        self.alpha = alpha
        self.training_max_length = training_max_length
        self.compute_loss = self.compute_loss_training

        self.current_distillation_loss = None
        self.current_task_loss = None

        self.peft_type = peft_type
        self.adalora_samples_per_phase = adalora_samples_per_phase
        self.adalora_config = adalora_config or {
            'target_modules': ['q_proj', 'v_proj'],
            'init_r': 12,
            'target_r': 8,
            'beta1': 0.85,
            'beta2': 0.85,
            'tinit': 200,
            'tfinal': 1000,
            'deltaT': 10,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        self.is_adalora_phase = False
        self.samples_trained_in_phase = 0
        self.base_model_state = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.teacher_model is not None:
            self.teacher_model.to(self.accelerator.device)
        return control

    def evaluate(self, *args, **kwargs):
        self.compute_loss = self.compute_loss_evaluation

        if self.teacher_model is None:
            metrics = super().evaluate(*args, **kwargs)
            self.compute_loss = self.compute_loss_training
            return metrics

        torch.cuda.empty_cache()
        print("Evaluating student model with gradient checkpointing enabled...")
        self.model.gradient_checkpointing_enable()
        metrics = super().evaluate(*args, **kwargs)
        self.model.gradient_checkpointing_disable()

        self.compute_loss = self.compute_loss_training
        return metrics

    def compute_loss_training(
            self,
            model: AutoModelForCausalLM,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
            ignore_index: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute distillation loss."""
        if self.training_max_length is not None:
            inputs = {
                key: value[:, :self.training_max_length].detach()
                for key, value in inputs.items()
            }
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask", None)

        # Pretrained-student path: no teacher, just causal LM loss
        if self.teacher_model is None:
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous().bool()
                logits_flat = shift_logits.view(-1, shift_logits.size(-1))
                labels_flat = shift_labels.view(-1)
                logits_masked = logits_flat[shift_mask.view(-1)]
                labels_masked = labels_flat[shift_mask.view(-1)]
            else:
                logits_masked = shift_logits.view(-1, shift_logits.size(-1))
                labels_masked = shift_labels.view(-1)
            loss = F.cross_entropy(logits_masked, labels_masked, ignore_index=-100)
            self.current_distillation_loss = None
            self.current_task_loss = loss.detach().item()
            return (loss, outputs) if return_outputs else loss

        # Teacher forward pass
        with torch.no_grad():
            with autocast(enabled=False):
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits.float()

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        if student_logits.size(-1) != teacher_logits.size(-1):
            raise ValueError(
                "Teacher and student logits have different vocabulary sizes "
                f"(student: {student_logits.size(-1)}, teacher: {teacher_logits.size(-1)}). "
                "Ensure both models share the tokenizer and the student embeddings are resized."
            )

        # 1. DISTILLATION LOSS
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        student_lp_flat = student_log_probs.view(-1, student_log_probs.size(-1))
        teacher_p_flat = teacher_probs.view(-1, teacher_probs.size(-1))
        mask_flat = attention_mask.reshape(-1).bool()

        student_lp_masked = student_lp_flat[mask_flat]
        teacher_p_masked = teacher_p_flat[mask_flat]

        distillation_loss = F.kl_div(
            student_lp_masked, teacher_p_masked,
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # 2. TASK LOSS
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        mask_flat = shift_mask.view(-1).bool()

        logits_masked = logits_flat[mask_flat]
        labels_masked = labels_flat[mask_flat]
        task_loss = F.cross_entropy(logits_masked, labels_masked, ignore_index=ignore_index)

        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss

        self.current_distillation_loss = distillation_loss.detach().item()
        self.current_task_loss = task_loss.detach().item()

        return (loss, student_outputs) if return_outputs else loss

    def compute_loss_evaluation(
            self,
            model: AutoModelForCausalLM,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: int | None = None,
            ignore_index: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute evaluation loss (student-only CE)."""
        if self.training_max_length is not None:
            inputs = {
                key: value[:, :self.training_max_length]
                for key, value in inputs.items()
            }
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        mask_flat = shift_mask.view(-1).bool()

        logits_masked = logits_flat[mask_flat]
        labels_masked = labels_flat[mask_flat]
        task_loss = F.cross_entropy(logits_masked, labels_masked, ignore_index=ignore_index)

        return (task_loss, outputs) if return_outputs else task_loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = enhance_logs_with_custom_metrics(
            logs,
            self.current_distillation_loss,
            self.current_task_loss,
            self,
        )
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def _setup_adalora(self, total_step: int):
        """Setup AdaLoRA adapters on the model."""
        if total_step <= 0:
            raise ValueError("AdaLoRA requires total_step > 0 for this phase.")
        if isinstance(self.model, PeftModel):
            print("Model already has PEFT adapters, skipping setup.")
            self.is_adalora_phase = True
            return

        print("\nSetting up AdaLoRA adapters...")
        print(f"  total_step for this phase: {total_step}")

        if self.base_model_state is None:
            self.base_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        default_tinit = self.adalora_config['tinit']
        default_tfinal = self.adalora_config['tfinal']
        default_deltaT = self.adalora_config['deltaT']

        tinit = max(1, min(default_tinit, total_step // 10))
        tfinal = max(1, min(default_tfinal, total_step // 10))
        if tinit + tfinal >= total_step:
            tinit = max(1, total_step // 20)
            tfinal = max(1, (total_step - tinit - 1) // 2)
        if tinit + tfinal >= total_step:
            tinit = 1
            tfinal = max(1, (total_step - 2) // 2)
        deltaT = min(default_deltaT, max(1, (total_step - tfinal - tinit) // 5))

        target_modules = infer_adalora_target_modules(self.model)
        print(f"  AdaLoRA target_modules: {target_modules}")

        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            init_r=self.adalora_config['init_r'],
            target_r=self.adalora_config['target_r'],
            beta1=self.adalora_config['beta1'],
            beta2=self.adalora_config['beta2'],
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            lora_alpha=self.adalora_config['lora_alpha'],
            lora_dropout=self.adalora_config['lora_dropout'],
            total_step=total_step,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.is_adalora_phase = True
        self.samples_trained_in_phase = 0
        print("AdaLoRA adapters setup complete!")

    def _merge_adalora(self):
        """Merge AdaLoRA adapters back into the base model."""
        if not isinstance(self.model, PeftModel):
            return
        print("Merging AdaLoRA adapters into base model...")
        self.model = self.model.merge_and_unload()
        self.is_adalora_phase = False
        self.samples_trained_in_phase = 0
        print("AdaLoRA adapters merged successfully!")

    def _switch_to_full_training(self):
        """Switch from AdaLoRA to full training mode."""
        print("Switching to full training mode...")
        if isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
        for param in self.model.parameters():
            param.requires_grad = True
        self.is_adalora_phase = False
        self.samples_trained_in_phase = 0
        print("Full training mode activated!")

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """Custom train method: peft_type in "adalora_full", None, "adalora_alternating"."""
        if self.peft_type == "adalora_full":
            n_gpu = getattr(self.args, 'n_gpu', 0) if hasattr(self.args, 'n_gpu') else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
            batch_size = self.args.per_device_train_batch_size * max(1, n_gpu) * getattr(self.args, 'gradient_accumulation_steps', 1)
            num_examples = len(self.train_dataset)
            num_update_steps_per_epoch = max(1, num_examples // batch_size)
            total_step = int(self.args.num_train_epochs * num_update_steps_per_epoch) if self.args.max_steps <= 0 else self.args.max_steps
            total_step = max(1, total_step)
            print(f"AdaLoRA full run (single phase, no alternating). Total steps: {total_step}")
            self._setup_adalora(total_step=total_step)
            return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        if self.peft_type != "adalora_alternating":
            return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        # Alternating training: AdaLoRA -> Merge -> Full -> Repeat
        print(f"Starting alternating AdaLoRA/Full training. Samples per phase: {self.adalora_samples_per_phase}")
        total_samples = len(self.train_dataset)
        n_gpu = getattr(self.args, 'n_gpu', 0) if hasattr(self.args, 'n_gpu') else (torch.cuda.device_count() if torch.cuda.is_available() else 0)
        batch_size = self.args.per_device_train_batch_size * max(1, n_gpu)
        steps_per_phase = max(1, self.adalora_samples_per_phase // batch_size)

        current_sample_idx = 0
        phase_num = 1
        final_output = None

        original_max_steps = self.args.max_steps
        original_num_train_epochs = self.args.num_train_epochs

        while current_sample_idx < total_samples:
            is_adalora = (phase_num % 2 == 1)
            samples_to_train = min(self.adalora_samples_per_phase, total_samples - current_sample_idx)

            if is_adalora:
                print(f"PHASE {phase_num}: AdaLoRA Training ({samples_to_train} samples)")
                if not isinstance(self.model, PeftModel):
                    self._setup_adalora(total_step=steps_per_phase)
            else:
                print(f"PHASE {phase_num}: Full Training ({samples_to_train} samples)")
                if isinstance(self.model, PeftModel):
                    self._merge_adalora()
                self._switch_to_full_training()

            phase_dataset = self.train_dataset.select(
                range(current_sample_idx, current_sample_idx + samples_to_train)
            )
            original_train_dataset = self.train_dataset
            self.train_dataset = phase_dataset
            self.args.max_steps = steps_per_phase
            self.args.num_train_epochs = 1

            try:
                phase_output = super().train(
                    resume_from_checkpoint=None,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    **kwargs
                )
                final_output = phase_output
            except Exception as e:
                print(f"Error in phase {phase_num}: {e}")
                import traceback
                traceback.print_exc()
                break

            self.train_dataset = original_train_dataset

            if is_adalora and isinstance(self.model, PeftModel):
                self._merge_adalora()

            current_sample_idx += samples_to_train
            phase_num += 1
            print(f"Phase {phase_num - 1} complete. Total samples trained: {current_sample_idx}/{total_samples}")

        self.args.max_steps = original_max_steps
        self.args.num_train_epochs = original_num_train_epochs

        if isinstance(self.model, PeftModel):
            self._merge_adalora()

        print("Alternating training complete!")
        return final_output if final_output is not None else super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
