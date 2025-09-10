import random

from typing import Any, Dict, List, Union

import torch
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import pad

from bridge_official.neural_models.in_batch_docs import build_in_batch_docs


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
    def __init__(self, tokenizer, dataset_text_field='text'):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.dataset_text_field = dataset_text_field

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        original_batch_size = len(batch)
        # Flatten all texts in the batch
        texts = [example[self.dataset_text_field] for example in batch]
        encodings = self.tokenizer(texts, add_special_tokens=False).input_ids

        # Flatten all tokens into one long list
        all_tokens = [token for sequence in encodings for token in sequence]

        # Pack into chunks of max_length
        chunks = [all_tokens[i:i + self.max_length] for i in range(0, len(all_tokens), self.max_length)]

        # Pad all chunks to max_length

        # print('self.max_length - len(chunk)', self.max_length - len(chunks[0]))
        input_ids = [chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk)) for chunk in chunks]
        attention_mask = [[1] * len(chunk) + [0] * (self.max_length - len(chunk)) for chunk in chunks]

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)

        # Since bounding will lose information, let's try to show more data randomly in another epoch
        shuffling = torch.randperm(input_ids.size(0))
        input_ids = input_ids[shuffling]
        attention_mask = attention_mask[shuffling]

        # Ensure a bounded batch size
        input_ids = input_ids[:2 * original_batch_size]
        attention_mask = attention_mask[:2 * original_batch_size]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # Labels are the same as input_ids for LM
        }


def time_fold_tensor(x, n_folds, pad_id):
    b, t = x.shape
    remainder = t % n_folds
    if remainder != 0:
        pad_len = n_folds - remainder
        pad_tensor = x.new_full((b, pad_len), pad_id)
        x = torch.cat([x, pad_tensor], dim=1)
        t = x.shape[1]

    # Reshape: (batch, n_folds, time // n_folds)
    x = x.view(b, n_folds, t // n_folds)

    # Move folds axis to the front
    x = x.permute(1, 0, 2).contiguous()
    return x


class TwoTokenizersCollator:
    def __init__(
            self,
            tokenizer_encoder,
            text_field_encoder='text',
            tokenizer_decoder=None,
            text_field_decoder=None,
            truncation_encoder=False,
            truncation_decoder=False,
            random_encoder_folding=True,
            in_batch_docs=False,
            encoder_docs_axis=True
    ):
        """
        Data collator that uses two different tokenizers for two different text fields in the dataset.
        Args:
            tokenizer_encoder: The tokenizer for the encoder input (e.g., the article text).
            text_field_encoder: The field name in the dataset for the encoder input text.
            tokenizer_decoder: The tokenizer for the decoder input (e.g., the summary text). If None, uses tokenizer_encoder.
            text_field_decoder: The field name in the dataset for the decoder input text. If None, uses text_field_encoder.
            truncation_encoder: Whether to truncate the encoder input to the model's maximum length.
            truncation_decoder: Whether to truncate the decoder input to the model's maximum length.
            random_encoder_folding: Whether to randomly fold the encoder input in time dimension for data augmentation.
        """

        self.tokenizer_encoder = tokenizer_encoder
        self.text_field_encoder = text_field_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.text_field_decoder = text_field_decoder if text_field_decoder is not None else text_field_encoder
        self.truncation_encoder = truncation_encoder
        self.truncation_decoder = truncation_decoder
        self.random_encoder_folding = random_encoder_folding
        self.in_batch_docs = in_batch_docs
        self.encoder_docs_axis = encoder_docs_axis

        self.encoder_max_length = self.tokenizer_encoder.model_max_length

    def do_random_encoder_folding(self, encodings):
        if not self.random_encoder_folding:
            return encodings
        possible_folds = list(range(11))
        n_folds = random.choice(possible_folds)

        if n_folds == 0:
            encodings['input_ids'] = encodings['input_ids'].unsqueeze(0)
            encodings['attention_mask'] = encodings['attention_mask'].unsqueeze(0)
            return encodings

        encodings['input_ids'] = time_fold_tensor(
            encodings['input_ids'],
            n_folds + 1, self.tokenizer_encoder.pad_token_id
        )
        encodings['attention_mask'] = time_fold_tensor(
            encodings['attention_mask'],
            n_folds + 1, 0
        )

        return encodings

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:

        texts_decoder = [example[self.text_field_decoder] for example in examples]
        ids_decoder = self.tokenizer_decoder(
            texts_decoder,
            add_special_tokens=True,
            padding=True,  # same as "longest"
            truncation=self.truncation_decoder,
            return_tensors="pt"
        )

        text_field_encoder = self.text_field_encoder
        same_encdec_text = not self.text_field_encoder in examples[0]
        if same_encdec_text:
            text_field_encoder = self.text_field_decoder

        texts_encoder = [example[text_field_encoder] for example in examples]
        ids_encoder = self.tokenizer_encoder(
            texts_encoder,
            add_special_tokens=True,
            padding=True,  # same as "longest"
            truncation=False,
            return_tensors="pt"
        )

        if not same_encdec_text:
            ids_encoder = self.do_random_encoder_folding(ids_encoder)

        input_ids_encoder = ids_encoder["input_ids"]
        attention_mask_encoder = ids_encoder["attention_mask"]

        if self.in_batch_docs and same_encdec_text:
            input_ids_encoder = build_in_batch_docs(
                batch=input_ids_encoder, mode='auto', batch_first=True,
                vocab_size=self.tokenizer_encoder.vocab_size
            )
            attention_mask_encoder = (input_ids_encoder != self.tokenizer_encoder.pad_token_id).long()

        if self.encoder_docs_axis and len(input_ids_encoder.shape) == 2:
            input_ids_encoder = input_ids_encoder.unsqueeze(0)
            attention_mask_encoder = attention_mask_encoder.unsqueeze(0)

        input_ids_encoder = input_ids_encoder[..., -self.encoder_max_length:]
        attention_mask_encoder = attention_mask_encoder[..., -self.encoder_max_length:]

        return {
            "input_ids": ids_decoder["input_ids"],
            "input_ids_encoder": input_ids_encoder,
            "attention_mask": ids_decoder["attention_mask"],
            "attention_mask_encoder": attention_mask_encoder,

            "labels": ids_decoder["input_ids"].clone(),
        }


def get_collator(dataset_name, tokenizer_encoder, tokenizer_decoder, notes):

    collator = TwoTokenizersCollator(
        tokenizer_encoder=tokenizer_encoder,
        tokenizer_decoder=tokenizer_decoder,
        text_field_encoder="article",
        text_field_decoder="text",
        truncation_encoder=True,
        truncation_decoder=True,
        in_batch_docs=True
    )

    if 'dolma' in dataset_name and 'oldclltr' in notes:
        collator = PackingOnlineCollator(tokenizer_decoder)

    return collator
