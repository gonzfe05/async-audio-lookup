from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial
import json
from typing import List
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from lookup.dataset import SegmentDataset, CustomDataset
import torch
from dataclasses import dataclass, field
import numpy as np
from datasets import load_metric


def prepare_dataset(_input: dict, processor: Wav2Vec2Processor) -> dict:
    """Process input and target values separately.

    Parameters
    ----------
    _input : dict
        Contains audio array key for input and text key for target.
    processor : Wav2Vec2Processor
        Prepares input array for wav2vec and text target for CTC

    Returns
    -------
    dict
        Processed input for wav2vecforctc
    """
    audio = _input["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    _input["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        _input["labels"] = processor(_input["text"]).input_ids
    return _input

def w2v_data_loader(uris: List[str], in_sr: int, out_sr: int) -> Tuple[CustomDataset, Wav2Vec2Processor]:
    """Creates CustomDataset with Wav2VecProcessor for using with Wav2VecForCTC

    Parameters
    ----------
    uris : List[str]
        Data to load into the dataset
    ndim : int
        Used to query DocArray's DocStore
    in_sr : int
        Raw audio sample rate
    out_sr : int
        Dataset sample rate

    Returns
    -------
    Tuple[CustomDataset, Wav2Vec2Processor]
        To be used for training Wav2VecForCTC
    """
    segment_dataset = SegmentDataset(uris, in_sr, out_sr)
    with open('/tmp/vocab.json', 'w') as f:
        json.dump(segment_dataset.vocab, f)
    tokenizer = Wav2Vec2CTCTokenizer("/tmp/vocab.json",
                                     unk_token=segment_dataset.unk_token,
                                     pad_token=segment_dataset.pad_token,
                                     word_delimiter_token=segment_dataset.word_delimiter_token)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=out_sr, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    _func = partial(prepare_dataset, processor=processor)
    dataset = CustomDataset(uris, in_sr, out_sr, _func)
    return dataset, processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_i
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_metric = load_metric("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def load_w2v(processor: Wav2Vec2Processor, checkpoint: str = "facebook/wav2vec2-base") -> Wav2Vec2ForCTC:
    """Loads Wav2VecForCTC with processor pad_token_id

    Parameters
    ----------
    processor : Wav2Vec2Processor
        Processor used in the dataset creation
    checkpoint : str, optional
        Path to model, by default "facebook/wav2vec2-base"

    Returns
    -------
    Wav2Vec2ForCTC
        Model
    """
    return Wav2Vec2ForCTC.from_pretrained(
        checkpoint, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id)

def get_data_collator(processor):
    """Load data collator that will dynamically pad the inputs received."""
    return DataCollatorCTCWithPadding(processor=processor, padding=True)

def load_trainer(model: Wav2Vec2ForCTC,
                 data_collator: DataCollatorCTCWithPadding,
                 train_dataset: CustomDataset,
                 eval_dataset: CustomDataset,
                 processor: Wav2Vec2Processor,
                 epochs: int = 10,
                 freeze_features: bool = True,
                 use_cuda: bool = False,
                 output_dir: int ='checkpoint/') -> Trainer:
    """Loads a trainer for Wav2VecForCTC from a CustomDataset"""
    if freeze_features:
        model.freeze_feature_extractor()
    fp16=False
    if use_cuda:
        model.cuda()
        fp16=True
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        fp16=fp16,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        )

    return Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

def map_to_result(batch: dict, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor) -> dict:
    """Predict target text and add to batch

    Parameters
    ----------
    batch : dict
        Batch of inputs to predict
    model : Wav2Vec2ForCTC
        For prediction
    processor : Wav2Vec2Processor
        Used to decode model output

    Returns
    -------
    dict
        Batch with predictions as pred_str
    """
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    return batch
