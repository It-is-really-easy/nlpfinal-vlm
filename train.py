import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'
import torch
from typing import Optional
from functools import partial
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from trainer import MultiModalTrainer
from model.model import MMultiModal, LanguageConfig, VisualConfig, MultiModalConfig
from dataset.my_image_caption_dataset import ImageCaptionDataset, data_collate

import transformers
from transformers import HfArgumentParser, AutoTokenizer, ChineseCLIPProcessor
from dataclasses import dataclass, field

from qwen.modeling_qwen import QWenLMHeadModel
from qwen.qwen_generation_utils import make_context

@dataclass
class FinetuneArguments:
    lora_rank: int = field(default=8)
    lora_dropout: float = field(default=0.1)
    previous_lora_weights: str = field(default=None) 
    target_modules: str = field(default="W_pack") 
    image_dir: str = field(default='./dataset/images/train', metadata={"help": "训练集路径"})
    captions_file: str = field(default="./dataset/LevirCCcaptions.json", metadata={"help": "数据集描述"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    feature_proj_lr: Optional[float] = None

def train():
    # print("cuda is available:", torch.cuda.is_available())
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    base_language_model = "./Qwen-7B-Chat"
    base_value_model = "./clip-vit-large-patch14"
    # base_language_model = "F:/huggingface_model/qwen/Qwen-7B-Chat"
    # base_value_model = "F:/huggingface_model/clip-vit-large-patch14"

    tokenizer = AutoTokenizer.from_pretrained(base_language_model, trust_remote_code=True)
    replace_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
    
    # print(replace_token_id)

    if torch.cuda.is_available():
        torch.cuda.init()
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("training on:", device)
    model = MMultiModal(LanguageConfig(model_path=base_language_model), VisualConfig(model_path=base_value_model), 
                        MultiModalConfig(replace_token_id=replace_token_id), finetune_args, train=True)
    model.to(device)
    model.train()
    model.LLM.config.use_cache = False

    

    dataset = ImageCaptionDataset(tokenizer, finetune_args.image_dir, finetune_args.captions_file, VisualConfig(model_path=base_value_model), max_train_data_item=300000)

    print(training_args)

    trainer = MultiModalTrainer(model=model,
                                data_collator=partial(data_collate, tokenizer=tokenizer, black_token_length = MultiModalConfig.image_context_length),
                                train_dataset=dataset,
                                args=training_args)
    trainer.train()

def main():
    train()

if __name__ == "__main__":
    main()

