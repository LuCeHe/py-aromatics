from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from trl.trainer.utils import pad
import torch

from transformers.data.data_collator import DataCollatorMixin
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


class DataCollatorForOnlineLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch if
    they are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForLanguageModeling
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [
    ...     {"input_ids": [1, 2, 3]},
    ...     {"input_ids": [4, 5]}
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[   1,   2,   3],
                          [   4,   5,   0]]),
     'attention_mask': tensor([[  1,   1,   1],
                               [  1,   1,   0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side="right")
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side="right")
        output["labels"] = pad(labels, padding_value=-100, padding_side="right")

        return output


class PackingOnlineCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        original_batch_size = len(batch)
        # Flatten all texts in the batch
        texts = [example["text"] for example in batch]
        encodings = self.tokenizer(texts, add_special_tokens=False).input_ids

        # Flatten all tokens into one long list
        all_tokens = [token for sequence in encodings for token in sequence]

        # Pack into chunks of max_length
        chunks = [all_tokens[i:i + self.max_length] for i in range(0, len(all_tokens), self.max_length)]

        # Pad all chunks to max_length
        input_ids = [chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk)) for chunk in chunks]
        attention_mask = [[1] * len(chunk) + [0] * (self.max_length - len(chunk)) for chunk in chunks]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        print('max value of input_ids:', input_ids.max())
        print('min value of input_ids:', input_ids.min())

        # Since bounding will lose information, let's try to show more data randomly in another epoch
        shuffling = torch.randperm(input_ids.size(0))
        input_ids = input_ids[shuffling]
        attention_mask = attention_mask[shuffling]

        # Ensure a bounded batch size
        input_ids = input_ids[:2 * original_batch_size]
        attention_mask = attention_mask[:2 * original_batch_size]
        print('shapes of input_ids and attention_mask:', input_ids.shape, attention_mask.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # Labels are the same as input_ids for LM
        }
