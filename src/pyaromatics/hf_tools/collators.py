import random
from transformers import PreTrainedTokenizerBase
from typing import Optional, Dict, List, Union

import torch

from bridge_official.neural_models.in_batch_docs import build_in_batch_docs
from pyaromatics.stay_organized.utils import str2val


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


def get_masked_indices(tensor, tokenizer, mlm_probability=0.15):
    # Create a mask for 15% of the tokens
    probability_matrix = torch.full(tensor.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
        tensor.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    return masked_indices


class TwoTokenizersCollator:
    def __init__(
            self,
            tokenizer_encoder: PreTrainedTokenizerBase,
            text_field_encoder: str = 'text',
            tokenizer_decoder: Optional[PreTrainedTokenizerBase] = None,
            text_field_decoder: Optional[str] = None,
            truncation_encoder: bool = True,
            truncation_decoder: bool = True,
            encoder_folds: Union[bool, str, int] = False,
            mlm_probability: float = 0.0,
            in_batch_docs: bool = False,
            encoder_docs_axis: bool = False,
            shift_labels: bool = False,
            padding_side: str = "left",
            truncation_side: str = "right",
            call_max_length_encoder: int = 50_000,
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
            encoder_folds: Whether to randomly fold the encoder input in time dimension for data augmentation. Can be
            False, 'random', 'max', 'halfmax' or an integer.
            mlm_probability: Whether to apply masked language modeling to the decoder input. P stands for probability.
        """

        self.tokenizer_encoder = tokenizer_encoder
        self.text_field_encoder = text_field_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.text_field_decoder = text_field_decoder if text_field_decoder is not None else text_field_encoder
        self.truncation_encoder = truncation_encoder
        self.truncation_decoder = truncation_decoder
        self.encoder_folds = encoder_folds
        self.mlm_probability = mlm_probability
        self.in_batch_docs = in_batch_docs
        self.encoder_docs_axis = encoder_docs_axis
        self.shift_labels = shift_labels
        self.padding_side = padding_side

        self.vocab_size_encoder = self.tokenizer_encoder.vocab_size
        self.vocab_size_decoder = self.tokenizer_decoder.vocab_size

        self.tokenizer_encoder.padding_side = padding_side
        self.tokenizer_decoder.padding_side = padding_side
        self.tokenizer_encoder.truncation_side = truncation_side
        self.tokenizer_decoder.truncation_side = truncation_side

        self.call_max_length_encoder = call_max_length_encoder
        self.final_max_length_encoder = tokenizer_encoder.model_max_length

    def do_encoder_folds(self, encodings):
        if self.encoder_folds == False:
            return encodings
        possible_folds = list(range(11))

        n_folds = 0
        if self.encoder_folds == 'random':
            n_folds = random.choice(possible_folds)

        elif self.encoder_folds == 'max':
            n_folds = encodings['input_ids'].shape[1] // self.final_max_length_encoder

        elif self.encoder_folds == 'halfmax':
            n_folds = max(1, encodings['input_ids'].shape[1] // (self.final_max_length_encoder // 2))

        elif isinstance(self.encoder_folds, int):
            n_folds = self.encoder_folds

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

    def do_masked_language_modeling(self, batch):
        labels = batch["labels"].clone()
        inputs = batch["input_ids"].clone()
        input_ids_encoder = batch["input_ids_encoder"]

        # for the decoder --------------------------
        masked_indices = get_masked_indices(labels, self.tokenizer_decoder, self.mlm_probability)

        # 80% of the times we pass the clean labels
        if random.random() < 0.8:
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            masked_indices = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices

        # 50% of masked tokens are replaced with random words
        # The other 50% of masked tokens are left unchanged
        random_words = torch.randint(len(self.tokenizer_decoder), labels.shape, dtype=torch.long)
        inputs[masked_indices] = random_words[masked_indices]

        # for the encoder --------------------------
        original_shape_enc = list(input_ids_encoder.shape)
        iie_reshaped = input_ids_encoder.reshape(-1, input_ids_encoder.shape[-1])
        masked_indices_enc = get_masked_indices(iie_reshaped, self.tokenizer_encoder, self.mlm_probability)

        random_words_enc = torch.randint(len(self.tokenizer_encoder), iie_reshaped.shape, dtype=torch.long)
        iie_reshaped[masked_indices_enc] = random_words_enc[masked_indices_enc]
        input_ids_encoder = iie_reshaped.view(*original_shape_enc)

        batch["labels"] = labels
        batch["input_ids"] = inputs
        batch["input_ids_encoder"] = input_ids_encoder
        return batch

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

        texts_encoder = [
            ' '.join(example[text_field_encoder]) if isinstance(example[text_field_encoder], list)
            else example[text_field_encoder]
            for example in examples
        ]


        # temporarily adjust for this call
        self.tokenizer_encoder.model_max_length = self.call_max_length_encoder
        ids_encoder = self.tokenizer_encoder(
            texts_encoder,
            add_special_tokens=True,
            padding=True,
            truncation=self.truncation_encoder,
            return_tensors="pt"
        )
        self.tokenizer_encoder.model_max_length = self.final_max_length_encoder

        if not same_encdec_text:
            ids_encoder = self.do_encoder_folds(ids_encoder)

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

        if self.padding_side == 'left':
            input_ids_encoder = input_ids_encoder[..., -self.final_max_length_encoder:]
            attention_mask_encoder = attention_mask_encoder[..., -self.final_max_length_encoder:]
        else:
            input_ids_encoder = input_ids_encoder[..., :self.final_max_length_encoder]
            attention_mask_encoder = attention_mask_encoder[..., :self.final_max_length_encoder]

        output = {
            "input_ids": ids_decoder["input_ids"],
            "input_ids_encoder": input_ids_encoder,
            "attention_mask": ids_decoder["attention_mask"],
            "attention_mask_encoder": attention_mask_encoder,
            "labels": ids_decoder["input_ids"].clone(),
        }

        if 0 < self.mlm_probability < 1:
            output = self.do_masked_language_modeling(output)

        if self.shift_labels:
            input_ids = output['input_ids'][..., :-1]
            labels = output['labels'][..., 1:]
            output['input_ids'], output['labels'] = input_ids, labels

        return output


def test_double_collator():
    enc_sentences = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Transformers are great for natural language processing tasks."
    ]
    enc_sentences = ["Hello, how are you?sadcascdddddddddddddd Hello, how are you?"] * 100
    dec_sentence = "I am fine, thank you!"

    from transformers import AutoTokenizer
    tokenizer_encoder = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_decoder = AutoTokenizer.from_pretrained("gpt2")
    # padding
    tokenizer_encoder.pad_token = tokenizer_encoder.eos_token
    tokenizer_decoder.pad_token = tokenizer_decoder.eos_token
    collator = TwoTokenizersCollator(
        tokenizer_encoder=tokenizer_encoder,
        text_field_encoder="passage_list",
        tokenizer_decoder=tokenizer_decoder,
        text_field_decoder="old_instruction",
        encoder_folds='halfmax',
    )

    examples = [
        {"passage_list": enc_sentences, "old_instruction": dec_sentence}
    ]

    output = collator(examples)
    for k, v in output.items():
        print(f"{k}: {v.shape}")


def get_collator(tokenizer_encoder, tokenizer_decoder, notes='', dataset_name='', eval=False):
    mlm_probability = str2val(notes, 'mlmprob', default=0.0, output_type=float)
    if eval:
        mlm_probability = 0.0

    shift_labels = 'manualshift' in notes
    if 'trainmanualshift' in notes and eval:
        shift_labels = False

    padding_side = 'right' if 'rightpad' in notes else 'left'
    truncation_side = 'left' if padding_side == 'right' else 'right'

    collator = TwoTokenizersCollator(
        tokenizer_encoder=tokenizer_encoder,
        tokenizer_decoder=tokenizer_decoder,
        text_field_encoder="article",
        text_field_decoder="text",
        truncation_encoder=True,
        truncation_decoder=True,
        encoder_folds='max',
        mlm_probability=mlm_probability,
        in_batch_docs=True,
        shift_labels=shift_labels,
        padding_side=padding_side,
        truncation_side=truncation_side,
    )

    if 'dolma' in dataset_name and 'oldclltr' in notes:
        collator = PackingOnlineCollator(tokenizer_decoder)

    return collator


if __name__ == "__main__":
    test_double_collator()
