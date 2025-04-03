from huggingface_hub import notebook_login
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig
import subprocess


def login_to_huggingface():
    """Logs in to the Hugging Face Hub from a notebook environment."""
    notebook_login()


def load_model_and_tokenizer(
    model_name: str,
    save_path: str,
    use_quantization: bool = False,
    quant_threshold: float = 6.0
):
    """
    Load and optionally quantize a model and tokenizer from Hugging Face.

    Args:
        model_name (str): HF model repo name or path.
        save_path (str): Local directory to cache/save model and tokenizer.
        use_quantization (bool): Whether to load model in 8-bit mode.
        quant_threshold (float): Threshold for quantization if enabled.

    Returns:
        model, tokenizer
    """
    if use_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant_threshold
        )
    else:
        quant_config = None

    if not os.path.exists(save_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            model.resize_token_embeddings(len(tokenizer))
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForCausalLM.from_pretrained(save_path, quantization_config=quant_config)

    return model, tokenizer


def add_lora_adapters(model, r=16, alpha=32, dropout=0.1):
    """
    Wraps a model with LoRA adapters for fine-tuning.

    Args:
        model: The base transformer model.
        r (int): Rank of LoRA update matrices.
        alpha (int): Scaling factor.
        dropout (float): Dropout for LoRA layers.

    Returns:
        Model with LoRA applied.
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, peft_config)


def prepare_dataset(csv_path: str, tokenizer, max_length: int = 512):
    """
    Load and tokenize a dataset from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        tokenizer: Hugging Face tokenizer.
        max_length (int): Max token length.

    Returns:
        tokenized train and evaluation datasets.
    """
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        tokenized = tokenizer(inputs, truncation=True, padding='max_length', max_length=max_length)
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    split = tokenized_dataset.train_test_split(test_size=0.1)
    return split['train'], split['test']


def get_training_args(output_dir: str, use_fp16: bool = True):
    """
    Constructs training arguments for the Trainer.

    Args:
        output_dir (str): Directory to save trained model and logs.
        use_fp16 (bool): Whether to use mixed-precision training.

    Returns:
        TrainingArguments instance.
    """
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        num_train_epochs=7,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=use_fp16,
    )


def run_training(model, tokenizer, train_dataset, eval_dataset, training_args):
    """
    Trains a model using the Hugging Face Trainer API.

    Args:
        model: The transformer model (with LoRA).
        tokenizer: The tokenizer.
        train_dataset: Tokenized training dataset.
        eval_dataset: Tokenized validation dataset.
        training_args: Hugging Face TrainingArguments.

    Returns:
        Trained model.
    """
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    torch.cuda.empty_cache()
    trainer.train()
    return trainer


def save_and_push_model(model, tokenizer, merged_dir: str, hf_repo: str):
    """
    Saves LoRA-merged model and pushes to Hugging Face Hub.

    Args:
        model: The LoRA model after training.
        tokenizer: The tokenizer.
        merged_dir (str): Directory to save merged model.
        hf_repo (str): Hugging Face repo to push to (e.g., org/model-name).
    """
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)

def convert_to_gguf(input_dir: str, output_dir: str, dtype: str = "bf16"):
    """
    Converts a Hugging Face model to GGUF format for use with llama.cpp.

    Args:
        input_dir (str): Path to the HF model directory.
        output_dir (str): Path to store the GGUF converted files.
        dtype (str): Precision format (e.g., bf16, f16, q4_0, q8_0).
    """
    command = [
        "python", "convert-hf-to-gguf.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--dtype", dtype
    ]
    subprocess.run(command, check=True)