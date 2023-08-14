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
from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForSeq2SeqLM

# KORISTI NAS MODEL
from models.rag.instructblip_model import InstructBlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrefixModelBLIP2Text(pl.LightningModule):
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
                      text_based_vision,
                      gold_answer,
                      pixel_values,
                    **kwargs):

        if self.config.model_config.UseInstructBLIP:
            
            batch_images_preprocessed = torch.stack(pixel_values).to(device)

            # question = []
            # for q in questions:
            #     question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=questions, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            text_based_vision = self.processor(
                text=text_based_vision, 
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
                pixel_values=inputs.pixel_values,
                qformer_input_ids=inputs.qformer_input_ids,
                qformer_attention_mask=inputs.qformer_attention_mask,
                input_ids=text_based_vision.input_ids,
                attention_mask=text_based_vision.attention_mask,
                labels=labels.input_ids.to(device))
            return generator_outputs.loss

        else:
            print("No model selected, please choose between InstructBLIP and BLIP2")
            exit(0)

    def generate(self, questions,
                      text_based_vision,
                      pixel_values,
                      **kwargs):
    
        batch_images_preprocessed = torch.stack(pixel_values).to(device)
        question = []
        # for q in questions:
        #     question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

        inputs = self.processor(
            images=batch_images_preprocessed, 
            text=questions, 
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt").to(device, torch.float32)
                    
                    
        text_based_vision = self.processor(
            text=text_based_vision, 
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt").to(device, torch.float32)

        generator_outputs = self.generator.generate(
            pixel_values=inputs.pixel_values,
            qformer_input_ids=inputs.qformer_input_ids,
            qformer_attention_mask=inputs.qformer_attention_mask,
            input_ids=text_based_vision.input_ids,
            attention_mask=text_based_vision.attention_mask)

        generated_text = self.processor.batch_decode(generator_outputs, skip_special_tokens=True)
            
        return generated_text


class PrefixModelBLIP2TextROI(pl.LightningModule):
    '''
    Blip2 and INstructBLIP model class
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.generator_tokenizer = data_loader.decoder_tokenizer

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
            # self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
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
                      text_based_vision,
                      gold_answer,
                      pixel_values,
                    **kwargs):

        if self.config.model_config.UseInstructBLIP:
            
            split_sizes = [len(img_list) for img_list in pixel_values]
            flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
            batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)

            # question = []
            # for q in questions:
            #     question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')
            extended_questions = []
            for idx, q in enumerate(questions):
                extended_questions.extend([q] * split_sizes[idx])

            # inputs = self.generator_tokenizer( 
            #     text=questions, 
            #     padding='longest',
            #     max_length=self.config.data_loader.additional.max_decoder_source_length,
            #     truncation=True,
            #     return_tensors="pt").to(device, torch.float32)

            inputs = self.generator_tokenizer(
                text=text_based_vision, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            qformer_inputs = self.generator_tokenizer(
                images=batch_images_preprocessed, 
                text=extended_questions, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            language_model_inputs, language_model_attention_mask = self.generator.get_qformer_features(
            pixel_values=qformer_inputs.pixel_values,
            qformer_input_ids=qformer_inputs.qformer_input_ids,
            qformer_attention_mask=qformer_inputs.qformer_attention_mask)

            # split_pixel_values = torch.split(qformer_inputs.pixel_values, split_sizes, dim=0)
            split_qformer_input_ids = torch.split(language_model_inputs, split_sizes, dim=0)
            split_qformer_attention_masks = torch.split(language_model_attention_mask, split_sizes, dim=0)
            # shapes = [tensor.shape for tensor in split_qformer_attention_masks]
            # print(f"Number of tensors: {len(shapes)}, Shapes: {shapes}")


            num_rois, num_tokens, lm_dim = split_qformer_input_ids[0].shape 

            reshaped_ids = [t.reshape(1, num_rois*num_tokens, lm_dim) for t in split_qformer_input_ids]
            concat_qformer_input_ids = torch.cat(reshaped_ids, dim=0)

            # print("concat: ", concat_qformer_input_ids.size)

            reshaped_mask = [t.reshape(1, num_rois*num_tokens) for t in split_qformer_attention_masks]
            concat_qformer_attention_masks = torch.cat(reshaped_mask, dim=0)


            labels = self.generator_tokenizer(
                text=gold_answer, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            generator_outputs = self.generator(
                language_model_inputs=concat_qformer_input_ids,
                language_model_attention_mask=concat_qformer_attention_masks,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels.input_ids.to(device),
                return_dict=True)
            return generator_outputs.loss

        else:
            print("No model selected, please choose between InstructBLIP and BLIP2")
            exit(0)

    def generate(self, questions,
                      text_based_vision,
                      pixel_values,
                      **kwargs):
    
        split_sizes = [len(img_list) for img_list in pixel_values]
        flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
        batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)
        question = []
        # for q in questions:
        #     question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')

        extended_questions = []
        for idx, q in enumerate(questions):
            extended_questions.extend([q] * split_sizes[idx])
            
        # inputs = self.generator_tokenizer(
        #     images=batch_images_preprocessed, 
        #     text=questions, 
        #     padding='longest',
        #     max_length=self.config.data_loader.additional.max_decoder_source_length,
        #     truncation=True,
        #     return_tensors="pt").to(device, torch.float32)
                    
        inputs = self.generator_tokenizer(
            text=text_based_vision, 
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt").to(device, torch.float32)

        qformer_inputs = self.generator_tokenizer(
            images=batch_images_preprocessed, 
            text=extended_questions, 
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt").to(device, torch.float32)

        language_model_inputs, language_model_attention_mask = self.generator.get_qformer_features(
        pixel_values=qformer_inputs.pixel_values,
        qformer_input_ids=qformer_inputs.qformer_input_ids,
        qformer_attention_mask=qformer_inputs.qformer_attention_mask)

        # split_pixel_values = torch.split(qformer_inputs.pixel_values, split_sizes, dim=0)
        split_qformer_input_ids = torch.split(language_model_inputs, split_sizes, dim=0)
        split_qformer_attention_masks = torch.split(language_model_attention_mask, split_sizes, dim=0)
        shapes = [tensor.shape for tensor in split_qformer_attention_masks]
        # print(f"Number of tensors: {len(shapes)}, Shapes: {shapes}")


        num_rois, num_tokens, lm_dim = split_qformer_input_ids[0].shape 

        reshaped_ids = [t.reshape(1, num_rois*num_tokens, lm_dim) for t in split_qformer_input_ids]
        concat_qformer_input_ids = torch.cat(reshaped_ids, dim=0)

        # print("concat: ", concat_qformer_input_ids.size)

        reshaped_mask = [t.reshape(1, num_rois*num_tokens) for t in split_qformer_attention_masks]
        concat_qformer_attention_masks = torch.cat(reshaped_mask, dim=0)

        generator_outputs = self.generator.generate(
            language_model_inputs=concat_qformer_input_ids,
            language_model_attention_mask=concat_qformer_attention_masks,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            # decoder_input_ids=generator_inputs.generator_decoder_input_ids,
            return_dict=True)

        generated_text = self.generator_tokenizer.batch_decode(generator_outputs, skip_special_tokens=True)    
            
        return generated_text