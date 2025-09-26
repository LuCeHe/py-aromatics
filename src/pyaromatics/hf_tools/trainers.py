from typing import List, Any, Optional, Union

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from trl import SFTTrainer

from transformers.trainer import (
    nn,
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size,
    deepspeed_init,
    logging,
    OptimizerNames,
    is_sagemaker_mp_enabled,
    # smp_forward_backward,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    DistributedType,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)


class TimeInterruptTrainer(SFTTrainer):

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                   or (
                               self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8" and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            if hasattr(self.control, 'should_prediction_stop') and self.control.should_prediction_stop:
                break

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
                self.compute_metrics is not None
                and all_preds is not None
                and all_labels is not None
                and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)




class TimeOOMSafeTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sequence_length = 16  # Minimum length to try
        self.min_docs = 2 # assuming (n_docs, batch_size, seq_len) for the encoder
        self.length_reduction_factor = 0.75  # Reduce by 25% each time
        self.docs_or_length = 'docs'  # 'length' or 'docs' to reduce sequence length or number of documents
        assert self.docs_or_length in ('length', 'docs')

    def training_step(self, *args, **kwargs):
        """Override training step with automatic length reduction on OOM."""
        try:
            # manually make it fail
            raise RuntimeError('cuda out of memory')
            return super().training_step(*args, **kwargs)
        except RuntimeError as e:
            if self._is_oom_error(e):
                print("🔄 OOM detected, attempting length reduction...")
                if self.docs_or_length == 'length':
                    return self._retry_with_reduced_length(*args, **kwargs)
                else:
                    return self._retry_with_reduced_docs(*args, **kwargs)

            else:
                raise e

    def _is_oom_error(self, error):
        """Check if error is OOM-related."""
        error_msg = str(error).lower()
        return any(phrase in error_msg for phrase in [
            'cuda out of memory',
            'no executable batch size found',
            'out of memory'
        ])

    def _retry_with_reduced_length(self, *args, **kwargs):
        """Retry training step with progressively reduced sequence length."""
        # Get current max length from inputs
        model, inputs, num_items_in_batch = args
        max_length = self._get_max_sequence_length(inputs)
        original_length = max_length

        while max_length >= self.min_sequence_length:
            try:
                # Reduce length
                max_length = int(max_length * self.length_reduction_factor)
                max_length = max(max_length, self.min_sequence_length)

                # Truncate inputs
                reduced_inputs = self._truncate_inputs(inputs, max_length)

                print(f"⚠️  Trying with reduced length: {max_length} (original: {original_length})")

                # Clear cache and retry
                torch.cuda.empty_cache()

                args = (model, reduced_inputs, num_items_in_batch)
                return super().training_step(*args, **kwargs)

            except RuntimeError as e:
                if self._is_oom_error(e):
                    continue  # Try with even smaller length
                else:
                    raise e

        # If we get here, even minimum length failed
        raise RuntimeError(f"Failed even at minimum length {self.min_sequence_length}. "
                           f"Consider reducing batch size or model size.")

    def _get_max_sequence_length(self, inputs):
        """Get the maximum sequence length from input tensors."""
        max_length = 0
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                max_length = max(max_length, value.shape[-1])
        return max_length



    def _get_max_docs(self, inputs):
        """Get the maximum sequence length from input tensors."""
        max_docs = 0
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 2:
                max_docs = max(max_docs, value.shape[0])
        return max_docs



    def _retry_with_reduced_docs(self, *args, **kwargs):
        """Retry training step with progressively reduced sequence length."""
        # Get current max length from inputs
        model, inputs, num_items_in_batch = args
        max_docs = self._get_max_docs(inputs)
        original_docs = max_docs

        while max_docs >= self.min_docs:
            try:
                # Reduce length
                max_docs = int(max_docs * self.length_reduction_factor)
                max_docs = max(max_docs, self.min_sequence_length)

                # Truncate inputs
                reduced_inputs = self._truncate_inputs_docs(inputs, max_docs)

                print(f"⚠️  Trying with reduced docs: {max_docs} (original: {original_docs})")

                # Clear cache and retry
                torch.cuda.empty_cache()

                args = (model, reduced_inputs, num_items_in_batch)
                return super().training_step(*args, **kwargs)

            except RuntimeError as e:
                if self._is_oom_error(e):
                    continue  # Try with even smaller length
                else:
                    raise e

        # If we get here, even minimum length failed
        raise RuntimeError(f"Failed even at minimum docs {self.min_docs}. "
                           f"Consider reducing batch size or model size.")



    def _truncate_inputs(self, inputs, max_length):
        print('truncating to', max_length)
        """Truncate all sequence tensors in inputs to max_length with padding-side deduction."""
        truncated = {}

        # Deduce padding side at batch level (fallbacks to tokenizer if available)
        padding_side = self._deduce_padding_side(inputs)

        for key, value in inputs.items():
            print('', key, value.shape if isinstance(value, torch.Tensor) else None)
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                if value.shape[-1] > max_length:
                    if padding_side == "left":
                        truncated[key] = value[..., -max_length:]
                    else:
                        truncated[key] = value[..., :max_length]
                else:
                    truncated[key] = value
            else:
                truncated[key] = value

        return truncated


    def _truncate_inputs_docs(self, inputs, max_docs):
        print('truncating to', max_docs)
        """Truncate all sequence tensors in inputs to max_length with padding-side deduction."""
        truncated = {}

        # Deduce padding side at batch level (fallbacks to tokenizer if available)
        padding_side = self._deduce_padding_side(inputs)

        for key, value in inputs.items():
            print('', key, value.shape if isinstance(value, torch.Tensor) else None)
            if isinstance(value, torch.Tensor) and len(value.shape) > 2:
                if value.shape[0] > max_docs:
                    truncated[key] = value[:max_docs]
                else:
                    truncated[key] = value
            else:
                truncated[key] = value

        return truncated

    def _deduce_padding_side(self, inputs) -> str:
        """Infer padding side ('left' or 'right') from batch tensors.

        Priority:
          4) fallback to tokenizer.padding_side if known, else 'right'
          1) attention_mask (1=real, 0=pad)
          2) labels with ignore_index -100
          3) input_ids with tokenizer.pad_token_id (if available)
        """

        # 4) Fallbacks
        tk = getattr(self, "tokenizer", None)
        if tk is not None and getattr(tk, "padding_side", None) in ("left", "right"):
            return tk.padding_side

        # 1) Use attention_mask if available
        attn = inputs.get("attention_mask", None)
        if isinstance(attn, torch.Tensor) and attn.ndim >= 2:
            # Count left vs right padding occurrences across batch
            bsz, seqlen = attn.shape[0], attn.shape[-1]
            # left pad if there are leading zeros before the first one
            first_one = (attn == 1).float().argmax(dim=-1)
            has_left_pad = (first_one > 0).sum().item()
            # right pad if there are trailing zeros after the last one
            # compute last_one index as seqlen-1 - argmax on reversed
            rev = torch.flip(attn, dims=[-1])
            last_one_from_end = (rev == 1).float().argmax(dim=-1)
            last_one = (seqlen - 1) - last_one_from_end
            has_right_pad = (last_one < seqlen - 1).sum().item()
            return "left" if has_left_pad > has_right_pad else "right"

        # 2) Use labels with ignore_index=-100
        labels = inputs.get("labels", None)
        if isinstance(labels, torch.Tensor) and labels.ndim >= 2:
            ignore = (labels == -100)
            if ignore.any():
                bsz, seqlen = labels.shape[0], labels.shape[-1]
                first_valid = (~ignore).float().argmax(dim=-1)
                has_left_pad = (first_valid > 0).sum().item()
                rev = torch.flip(~ignore, dims=[-1])
                last_valid_from_end = (rev == 1).float().argmax(dim=-1)
                last_valid = (seqlen - 1) - last_valid_from_end
                has_right_pad = (last_valid < seqlen - 1).sum().item()
                return "left" if has_left_pad > has_right_pad else "right"

        # 3) Use input_ids + tokenizer.pad_token_id
        input_ids = inputs.get("input_ids", None)
        pad_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
        if pad_id is not None and isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2:
            pads = (input_ids == pad_id)
            if pads.any():
                bsz, seqlen = input_ids.shape[0], input_ids.shape[-1]
                first_nonpad = (~pads).float().argmax(dim=-1)
                has_left_pad = (first_nonpad > 0).sum().item()
                rev = torch.flip(~pads, dims=[-1])
                last_nonpad_from_end = (rev == 1).float().argmax(dim=-1)
                last_nonpad = (seqlen - 1) - last_nonpad_from_end
                has_right_pad = (last_nonpad < seqlen - 1).sum().item()
                return "left" if has_left_pad > has_right_pad else "right"

        return "right"





class PlusTrainer(TimeInterruptTrainer, TimeOOMSafeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


