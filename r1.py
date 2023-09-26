from huggingface_hub import login
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState,
                          TrainerControl)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
import gc
import torch
import evaluate


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # to first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in the previous tokenization step,
        # cut bos token here as it's appended later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


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


def make_inputs_require_grad(_, __, output):
    output.requires_grad_(True)


def main():
    login("")
    model_name = "openai/whisper-large-v2"
    task = "transcribe"
    dataset_name = "mozilla-foundation/common_voice_13_0"
    language = "Japanese"
    language_abbr = "ja"
    output_directory = "tonyc729/whisper-large-v2-custom-ja"
    upload_directory = "tonyc729/whisper-large-v2-custom-ja"
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    #os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from the input audio array
        batch["input_features"] = \
        feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", token=True)
    common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", token=True)

    print(common_voice)

    common_voice = common_voice.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
    )

    print(common_voice)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    print(common_voice["train"][0])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    print(common_voice["train"][0])

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=8)
    print(common_voice["train"])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

    model = prepare_model_for_kbit_training(model)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_directory,  # change to a repo name of your choice
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
        gradient_checkpointing = True,
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=1,
        evaluation_strategy="steps",
        fp16=True,
        # bf16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        push_to_hub = True,
        # max_steps=100,  # only for testing purposes, remove this from your final run :)
        remove_unused_columns=False,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above

    )

    # This callback helps to save only the adapter weights and remove the base model weights.

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = True  # Silence the warnings. Please re-enable (make true) for inference!
    gc.collect()
    torch.cuda.empty_cache()
    print("Beginning Training")
    trainer.train()

    print("Finished training")
    peft_model_id = upload_directory
    print("Uploading to hf")
    model.push_to_hub(peft_model_id)
    print("Uploaded to hf")
    print("Completed, Exiting...")


if __name__ == '__main__':
    main()
