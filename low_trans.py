from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from typing import Any, Dict, List, Union
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import gc
import evaluate
import torch

login("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")

pretrained_model = "openai/whisper-large-v2"
source_data = "tonyc729/CoVoST2-ja-to-en-cv-15_0"
language_short = "ja"
language_long = "Japanese"
task = "translate"

common_voice = DatasetDict()
common_voice["train"] = load_dataset(source_data, language_short, split="train", token=True)
common_voice["test"] = load_dataset(source_data, language_short, split="test", token=True)

print(common_voice)

feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model)

tokenizer = WhisperTokenizer.from_pretrained(pretrained_model, language=language_long, task=task)

processor = WhisperProcessor.from_pretrained(pretrained_model, language=language_long, task=task)
print(common_voice["train"][0])

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["translation"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def make_inputs_require_grad(_, __, output):
    output.requires_grad_(True)


model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_kbit_training(model)
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v2-CoVoST2-ja-to-en-cv-15",  # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
processor.save_pretrained(training_args.output_dir)
gc.collect()
trainer.train()

kwargs = {
    "dataset_tags": source_data,
    "dataset": "Common Voice 15.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: ja, split: test",
    "language": language_short,
    "model_name": "Whisper-large-v2_cv15_CoVoST2-ja-to-en",  # a 'pretty' name for our model
    "finetuned_from": pretrained_model,
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)
