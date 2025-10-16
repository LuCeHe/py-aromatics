import time, GPUtil, psutil, traceback, gc, tempfile

from multiprocessing import get_context
from typing import Optional
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


def check_performance(tensors):
    print('\n\n')
    print('=' * 80)

    print('-' * 50)
    traceback.print_exc()
    print('-' * 50)

    print('Performance Check:')
    print('\nTensor Info:')
    for i, tensor in enumerate(tensors):
        if tensor is None:
            continue
        print(f'Tensor {i}: dtype={tensor.dtype}, device={tensor.device}, shape={tensor.shape}, '
              f'memory={tensor.element_size() * tensor.nelement() / 1024 ** 2:.2f} MB')

    print("\nGPU Info:")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
    else:
        print("No GPU available")

    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Load: {gpu.load * 100:.1f}%")
        print(f"  Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        print(f"  Temperature: {gpu.temperature} °C")

    # --- CPU Info ---
    print("\nCPU Info:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Total cores: {psutil.cpu_count(logical=True)}")
    print(f"  CPU usage per core: {psutil.cpu_percent(percpu=True, interval=1)}")
    print(f"  Total CPU usage: {psutil.cpu_percent()}%")

    # --- Memory Info ---
    mem = psutil.virtual_memory()
    print("\nMemory Info:")
    print(f"  Total: {mem.total / (1024 ** 3):.2f} GB")
    print(f"  Available: {mem.available / (1024 ** 3):.2f} GB")
    print(f"  Used: {mem.used / (1024 ** 3):.2f} GB")
    print(f"  Memory Usage: {mem.percent}%")

    print('\n\n')


def _run_training_step(fn, args, kwargs):
    """Helper to run in subprocess."""
    return fn(*args, **kwargs)


def _run_training_step_in_worker(
        training_step_fn,
        model_path, cpu_inputs, num_items_in_batch, kwargs):
    import torch
    from accelerate import Accelerator

    # Clear cache and retry
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 1️⃣ Load the full model (architecture + weights)
    model = torch.load(model_path, map_location="cpu")  # loaded on CPU first

    # 2️⃣ Move model to GPU
    # model.to("cuda")
    accelerator = Accelerator()
    model = accelerator.prepare_model(model, training=True)

    # 3️⃣ Move inputs to GPU
    gpu_inputs = {k: v.to("cuda") if torch.is_tensor(v) else v
                  for k, v in cpu_inputs.items()}

    # 4️⃣ Run one training step (forward + backward)
    args = (model, gpu_inputs, num_items_in_batch)
    output = training_step_fn(*args, **kwargs)

    # 5️⃣ Clean up
    del model

    # Clear cache and retry
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return output


class OOMSaferTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assuming encoder batch shape (docs, batch, length)
        self.shape_mins = {
            'docs': 2,
            'batch': 2,
            'length': 16,
        }

        self.reduction_factor = 0.75  # Reduce by 25% each time
        self.axis_to_oom_resize = 'docs'  # 'length' or 'docs' to reduce sequence length or number of documents
        assert self.axis_to_oom_resize in ('length', 'docs', 'batch', 'random', 'rotate')
        if self.axis_to_oom_resize in ['random', 'rotate']:
            raise NotImplementedError(f"{self.axis_to_oom_resize} not implemented yet")

    def training_step(self, *args, **kwargs):
        print('are kwargs consistent?', kwargs)
        """Override training step with automatic length reduction on OOM."""
        try:
            # manually make it fail
            # raise RuntimeError('cuda out of memory')
            output = super().training_step(*args, **kwargs)
            # output = self.safe_training_step(args, kwargs)
            return output
        except RuntimeError as e:
            print(e)
            if self._is_oom_error(e):
                print("🔄 OOM detected, attempting reduction...")
                return self._retry_with_reduced(*args, **kwargs)

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

    def _retry_with_reduced(self, *args, **kwargs):
        """Retry training step with progressively reduced sequence length."""
        # Get current max length from inputs

        reduce_axis = self.axis_to_oom_resize
        model, inputs, num_items_in_batch = args

        maxs = self._get_shape_maxs(inputs)
        mins = self.shape_mins
        reducible = [ax_name for ax_name in ['docs', 'batch', 'length'] if maxs[ax_name] > mins[ax_name]]
        max_axis = maxs[reduce_axis]
        min_axis = mins[reduce_axis]
        original_axis = max_axis

        reduced_inputs = inputs
        while max_axis >= min_axis:
            # Reduce length
            max_axis = int(max_axis * self.reduction_factor)

            new_maxs = self._get_shape_maxs(reduced_inputs)

            print(f"\n\n⚠️  Trying with reduced {reduce_axis} to {max_axis}")
            print(f"                   original: {original_axis} of {maxs}")
            print(f"                   previous: {new_maxs[reduce_axis]} of {new_maxs}")

            if max_axis < min_axis:
                break

            reduced_inputs = self._truncate_inputs(reduced_inputs, max_axis, reduce_axis=reduce_axis)

            args = (model, reduced_inputs, num_items_in_batch)

            # output = super().training_step(*args, **kwargs)
            output = self.safe_training_step(*args, **kwargs)
            if not output is None:
                return output

        # if reduce_axis was docs, then try batch, then try length
        if reduce_axis in ['docs', 'batch']:
            other_axis = 'batch' if reduce_axis == 'docs' else 'length'
            if other_axis in reducible:
                print(f"🔄 Switching to reducing {other_axis} axis.")
                self.axis_to_oom_resize = other_axis
                return self._retry_with_reduced(*args, **kwargs)

        # If we get here, even minimum length failed
        raise RuntimeError(f"Failed to run training step even with all axes at minimum: {mins} (original {maxs})")

    def safe_training_step(self, args, kwargs):
        ctx = get_context("spawn")

        # Clear cache and retry
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        model, inputs, num_items_in_batch = args

        # 1️⃣ Save the full model (architecture + weights) to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            torch.save(model, tmp.name)
            model_path = tmp.name

        # 2️⃣ Move inputs to CPU (only small batch, not full model)
        cpu_inputs = {k: v.detach().cpu() if torch.is_tensor(v) else v
                      for k, v in inputs.items()}

        output = None
        with ctx.Pool(1) as pool:
            # async_result = pool.apply_async(_run_training_step, (super().training_step, args, kwargs))
            async_result = pool.apply_async(
                _run_training_step_in_worker,
                (super().training_step,
                model_path,
                cpu_inputs,
                num_items_in_batch,
                kwargs

            )

            try:
                output = async_result.get(timeout=None)

            except (RuntimeError, TimeoutError) as e:
                reduced_inputs = args[1]
                check_performance(tensors=list(reduced_inputs.values()))

                if self._is_oom_error(e):
                    time.sleep(0.5)
                else:
                    raise e

        return output

    def _truncate_inputs(self, inputs, max_axis, reduce_axis='batch'):
        """Truncate all sequence tensors in inputs to max_length with padding-side deduction."""
        truncated = {}

        # Deduce padding side at batch level (fallbacks to tokenizer if available)
        padding_side = None
        if reduce_axis == 'length':
            padding_side = self._deduce_padding_side(inputs)

        # slices = [slice(None)] * 3  # [:, :, :, ...] dynamically
        # slices[axis] = slice(-max_axis, None)  # only modify the desired axis
        # slices = tuple(slices)
        # print(slices)

        for key, value in inputs.items():
            truncated[key] = value
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                print('      ', key, value.shape)
                if reduce_axis == 'length':
                    if padding_side == "left":
                        truncated[key] = value[..., -max_axis:]
                    else:
                        truncated[key] = value[..., :max_axis]

                elif reduce_axis == 'docs' and len(value.shape) > 2:
                    truncated[key] = value[:max_axis]

                elif reduce_axis == 'batch':
                    if len(value.shape) > 2:
                        truncated[key] = value[:, :max_axis]
                    else:
                        truncated[key] = value[:max_axis]
                key_spaces = ' ' * len(key)
                print('      ', key_spaces, truncated[key].shape)

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

    def _get_shape_maxs(self, inputs):
        """Get the maximum sequence length from input tensors."""
        max_docs = 0
        max_batch_size = 0
        max_length = 0

        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 2:
                max_docs = max(max_docs, value.shape[0])
                max_batch_size = max(max_batch_size, value.shape[1])
                max_length = max(max_length, value.shape[2])
        # return max_docs, max_batch_size, max_length
        return {'docs': max_docs, 'batch': max_batch_size, 'length': max_length}


# class PlusTrainer(TimeInterruptTrainer):
class PlusTrainer(TimeInterruptTrainer, OOMSaferTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
