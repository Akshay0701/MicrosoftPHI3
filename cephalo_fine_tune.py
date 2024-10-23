

# pip install numpy
# pip install pandas
# pip install Pillow
# pip install matplotlib
# pip install scikit-learn
# pip install tqdm
# pip install datasets
# pip install ipython
# pip install huggingface_hub
# pip install transformers
# pip install peft
# pip install torch
# pip install seaborn
# pip install bitsandbytes

import os
#If you want to limit visible CUDA devices
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import io
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm.notebook import tqdm
from io import BytesIO
import base64
import datetime
import seaborn as sns

from datasets import Dataset, DatasetDict, Features, Value, load_dataset, concatenate_datasets
from datasets import Image as DatasetsImage

from IPython.display import display, Image, HTML, Markdown

from huggingface_hub import HfFolder, HfApi

from transformers import (
    AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, TrainerCallback,
    TrainingArguments, Trainer
)
from transformers.image_utils import load_image

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel

from textwrap import wrap
import math

import torch


# Define CUDA device
DEVICE='cuda:0'

# Define base model to start from
base_model_id='microsoft/Phi-3-vision-128k-instruct'

# Define output directories and file names
output_dir = 'DSC_Microsoft_PHIV3'
FT_model_name = output_dir
loss_file = output_dir + '.csv'

# Training hyperparameters
num_train_epochs = 4
gradient_accumulation_steps = 4
learning_rate = 1e-5
max_grad_norm = 0.5
lr_scheduler_type = "cosine"
optim = "paged_adamw_8bit"
push_model_to_hub = False

# Additional configuration
warmup_ratio = 0.01

# Number of samples to use for training (set to None to use all)
N_training_samples_select = 'all' # Use all
#N_training_samples_select = 512


model = AutoModelForCausalLM.from_pretrained( base_model_id, torch_dtype=torch.bfloat16,
                                              _attn_implementation="eager",
                                              trust_remote_code=True,
                                            ).to (DEVICE )

processor = AutoProcessor.from_pretrained(
    base_model_id, trust_remote_code=True,
)

#Only target layers in FF block (for downstream MoE construction)
target_modules=[
        "gate_up_proj",
        "down_proj", #add others if you want to use other linear layers for LoRA
     ]
lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        bias="none",
        lora_dropout=0.1,
        target_modules=target_modules,
        use_dora=False,
        init_lora_weights="gaussian",
        task_type="CAUSAL_LM",
    )
model=get_peft_model(model, lora_config)


from datasets import load_dataset

train_dataset = load_dataset("akshay0701/DSCDataSet", split="train")
eval_dataset = load_dataset("akshay0701/DSCDataSet", split="test")





class CephaloDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        assert len (examples)==1, "Batch size must be 1."
        for example in examples:
            image = example["image"]
            question = example["query"]
            answer = example["answers"]
            try:
                text=example["text"]
            except:
                text=None

            if image!=None:
                messages = [ {
                            "role": "user",  "content": '<|image_1|>\n'+question},
                              {"role": "assistant", "content": f"{answer}"},
                            ]

                text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                batch = processor(text=text, images=[image], return_tensors="pt", padding=True
                         )
        labels = batch["input_ids"].clone()
        labels = torch.clamp(labels, min=0, )

        batch["labels"] = labels
        return batch

collator = CephaloDataCollator(processor)





training_args = TrainingArguments(
    output_dir = "DSC_Microsoft_PHIV3",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm = max_grad_norm,
    optim=optim,
    learning_rate=learning_rate,
    output_dir=f"{FT_model_name}",
    lr_scheduler_type=lr_scheduler_type,
    eval_steps = 10,
    save_steps = 25,
    max_steps = 25,
    remove_unused_columns=False,
    report_to="none",
    save_total_limit=50,
    bf16=True, #check compute ability of your GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset
)


trainer.train()


from huggingface_hub import login

# Replace 'your_api_token' with your actual Hugging Face API token
login(token='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')


model.push_to_hub("akshay0701/MicrosoftPHI3")
processor.push_to_hub("akshay0701/MicrosoftPHI3")