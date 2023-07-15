import copy
import math
import os
from turtle import forward
import warnings
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as T
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
import pytorch_lightning as pl

#from datasets import load_from_disk
import time
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrefixModelBLIP2(pl.LightningModule):
    '''
    Blip2 and INstructBLIP model class
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config

        if device == "cpu":
            self.float_type = torch.float32
        else:
            self.float_type = torch.float16

        if self.config.model_config.UseBLIP2:
        
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            self.generator = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=self.float_type)
            print('Uses full checkpoint from Salesforce/blip2-flan-t5-xl')

            self.r=8
            print('r value for LoRA is', self.r)
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.r, lora_alpha=32, lora_dropout=0.1)
            self.generator.language_model = PeftModelForSeq2SeqLM(self.generator.language_model, peft_config)
            self.generator.language_model.print_trainable_parameters()
            
        elif self.config.model_config.UseInstructBLIP:

            self.generator = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            print('Uses full checkpoint from Salesforce/instructblip-flan-t5-xl')
            
            self.r=8
            print('r value for LoRA is', self.r)
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.r, lora_alpha=32, lora_dropout=0.1)
            self.generator.language_model = PeftModelForSeq2SeqLM(self.generator.language_model, peft_config)
            self.generator.language_model.print_trainable_parameters()

        else:
            print("No model selected, please choose between InstructBLIP and BLIP2")
            exit(0)

    def forward(self, questions,
                      gold_answer,
                      pixel_values,
                    **kwargs):

        if self.config.model_config.UseBLIP2:
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            question = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, self.float_type)

            labels = self.processor(
                text=gold_answer, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt")#.to(device, self.float_type)

            generator_outputs = self.generator(
                **inputs, 
                labels=labels.input_ids.to(device))
            return generator_outputs.loss
            generated_ids = self.generator.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids)
            print(generated_text)
            
        elif self.config.model_config.UseInstructBLIP:
            
            batch_images_preprocessed = torch.stack(pixel_values).to(device)

            question = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            labels = self.processor(
                text=gold_answer, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt")

            generator_outputs = self.generator(
                **inputs, #input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values
                labels=labels.input_ids.to(device))

            return generator_outputs.loss

        else:
            print("No model selected, please choose between InstructBLIP and BLIP2")
            exit(0)

    def generate(self, questions,
                      pixel_values,
                      **kwargs):
        if self.config.model_config.UseBLIP2:
        
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            question = []
            for q in questions: 
                question.append('Use the provided image to answer the question: '+q+' Provide your answer as short as possible:')
                
            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, self.float_type)
            generated_ids = self.generator.generate(**inputs)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        else:
            '''
            images=[]
            for i in range(len(pixel_values)):
                images.append(self.vis_processors["eval"](self.transform(pixel_values[i]).to(device)))
            images = torch.stack(images).to(device)
            generated_text = self.model.generate({"image": images, "prompt": questions})
            '''
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            question = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            generator_outputs = self.generator.generate(**inputs)
            generated_text = self.processor.batch_decode(generator_outputs, skip_special_tokens=True)
            
        return generated_text