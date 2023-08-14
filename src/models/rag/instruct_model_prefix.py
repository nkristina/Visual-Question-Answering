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
            
            split_sizes = [len(img_list) for img_list in pixel_values]
            flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
            batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)

            print("batch_images_preprocessed.size: ", batch_images_preprocessed.size)

            question = []
            caption_prompt = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')
                # caption_prompt.append("Caption this image.")

            extended_questions = []
            for idx, q in enumerate(questions):
                extended_questions.extend(['Based on the image, respond to this question with a short answer: ' + q + '. Answer:'] * split_sizes[idx])
            
            print("extended:", len(extended_questions))
            if self.config.ROI:
                inputs = self.processor( 
                    text=question, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

                qformer_inputs = self.processor(
                    images=batch_images_preprocessed, 
                    text=extended_questions, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

                split_pixel_values = torch.split(qformer_inputs.pixel_values, split_sizes, dim=0)
                split_qformer_input_ids = torch.split(qformer_inputs.qformer_input_ids, split_sizes, dim=0)
                split_qformer_attention_masks = torch.split(qformer_inputs.qformer_attention_mask, split_sizes, dim=0)
                print("slipt: ", split_pixel_values.size)
                concat_pixel_values = torch.cat(split_pixel_values, dim=1)
                concat_qformer_input_ids = torch.cat(split_qformer_input_ids, dim=1)
                concat_qformer_attention_masks = torch.cat(split_qformer_attention_masks, dim=1)
                print("concat: ", concat_pixel_values.size)

            else:
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
            
            if self.config.ROI:
                # print("q-inputid", qformer_inputs.qformer_input_ids.shape)
                # print("q-atn", qformer_inputs.qformer_attention_mask.shape)
                generator_outputs = self.generator(
                    pixel_values=concat_pixel_values,
                    qformer_input_ids=concat_qformer_input_ids,
                    qformer_attention_mask=concat_qformer_attention_masks,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels.input_ids.to(device))
            else:
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
            split_sizes = [len(img_list) for img_list in pixel_values]
            flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
            batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)

            print("batch_images_preprocessed.size: ", batch_images_preprocessed.shape)

            question = []
            caption_prompt = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')
                # caption_prompt.append("Caption this image.")

            extended_questions = []
            for idx, q in enumerate(questions):
                extended_questions.extend(['Based on the image, respond to this question with a short answer: ' + q + '. Answer:'] * split_sizes[idx])
            
            print("extended:", len(extended_questions))
            if self.config.ROI:
                print("Usao:")
                inputs = self.processor( 
                    text=question, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

                qformer_inputs = self.processor(
                    images=batch_images_preprocessed, 
                    text=extended_questions, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

                print("Zavrsio procesor")    
                split_pixel_values = torch.split(qformer_inputs.pixel_values, split_sizes, dim=0)
                reshaped_tensors = [torch.cat(torch.unbind(tensor, dim=0), dim=2) for tensor in split_pixel_values]
                concat_pixel_values = torch.cat(reshaped_tensors, dim=0)

                split_qformer_input_ids = torch.split(qformer_inputs.qformer_input_ids, split_sizes, dim=0)
                split_qformer_attention_masks = torch.split(qformer_inputs.qformer_attention_mask, split_sizes, dim=0)
                shapes = [tensor.shape for tensor in split_qformer_input_ids]
                print(f"Number of tensors: {len(shapes)}, Shapes: {shapes}")

                # concat_pixel_values = torch.cat(split_pixel_values, dim=1)
                concat_qformer_input_ids = torch.cat(split_qformer_input_ids, dim=1)
                concat_qformer_attention_masks = torch.cat(split_qformer_attention_masks, dim=1)
                print("concat: ", concat_pixel_values.shape)

                # print("q-inputid", qformer_inputs.qformer_input_ids.shape)
                # print("q-atn", qformer_inputs.qformer_attention_mask.shape)
                generator_outputs = self.generator.generate(
                    pixel_values=concat_pixel_values,
                    qformer_input_ids=concat_qformer_input_ids,
                    qformer_attention_mask=concat_qformer_attention_masks,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask)
            else:
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

class PrefixModelBLIP2ROI(pl.LightningModule):
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
            
            split_sizes = [len(img_list) for img_list in pixel_values]
            flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
            batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)

            # print("batch_images_preprocessed.size: ", batch_images_preprocessed.size)

            question = []
            caption_prompt = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')
                # caption_prompt.append("Caption this image.")

            extended_questions = []
            for idx, q in enumerate(questions):
                extended_questions.extend(['Based on the image, respond to this question with a short answer: ' + q + '. Answer:'] * split_sizes[idx])
            
            # print("extended:", len(extended_questions))
            if self.config.ROI:
                inputs = self.generator_tokenizer( 
                    text=question, 
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

                # print("Zavrsio procesor")    
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

                # print("concat: ", concat_qformer_attention_masks.size)

                labels = self.generator_tokenizer(
                    text=gold_answer, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

            else:
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
            
            if self.config.ROI:
                # print("q-inputid", qformer_inputs.qformer_input_ids.shape)
                # print("q-atn", qformer_inputs.qformer_attention_mask.shape)
                # generator_decoder_input_ids = self.generator.language_model._shift_right(labels.input_ids)
                generator_outputs = self.generator(
                    language_model_inputs=concat_qformer_input_ids,
                    language_model_attention_mask=concat_qformer_attention_masks,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels.input_ids.to(device),
                    return_dict=True)
            else:
                generator_outputs = self.generator(
                    **inputs, #input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values
                    labels=labels.input_ids.to(device))


            # print("generator_outputs", generator_outputs)
            # print("generator_outputs.loss", generator_outputs.loss)
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
            split_sizes = [len(img_list) for img_list in pixel_values]
            flattened_pixel_values = [img for sublist in pixel_values for img in sublist]
            batch_images_preprocessed = torch.stack(flattened_pixel_values).to(device)

            print("batch_images_preprocessed.size: ", batch_images_preprocessed.shape)

            question = []
            caption_prompt = []
            for q in questions:
                question.append('Based on the image, respond to this question with a short answer: '+q+'. Answer:')
                # caption_prompt.append("Caption this image.")

            extended_questions = []
            for idx, q in enumerate(questions):
                extended_questions.extend(['Based on the image, respond to this question with a short answer: ' + q + '. Answer:'] * split_sizes[idx])
            
            # print("extended:", len(extended_questions))
            if self.config.ROI:
                # print("Usao:")
                inputs = self.generator_tokenizer( 
                    text=question, 
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

                # print("Zavrsio procesor")    
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

                # print("concat: ", concat_qformer_attention_masks.size)

                # print("q-inputid", qformer_inputs.qformer_input_ids.shape)
                # print("q-atn", qformer_inputs.qformer_attention_mask.shape)
                generator_outputs = self.generator.generate(
                    language_model_inputs=concat_qformer_input_ids,
                    language_model_attention_mask=concat_qformer_attention_masks,
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    # decoder_input_ids=generator_inputs.generator_decoder_input_ids,
                    return_dict=True)
            else:
                inputs = self.processor(
                    images=batch_images_preprocessed, 
                    text=question, 
                    padding='longest',
                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                    truncation=True,
                    return_tensors="pt").to(device, torch.float32)

                generator_outputs = self.generator.generate(**inputs)

            generated_text = self.generator_tokenizer.batch_decode(generator_outputs, skip_special_tokens=True)    

        return generated_text