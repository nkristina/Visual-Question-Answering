

import copy
import math
import os
from turtle import forward
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers import VisualBertModel, VisualBertConfig, BertTokenizer
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
from transformers import BertModel, BertConfig
from transformers.models.rag.retrieval_rag import CustomHFIndex, CanonicalHFIndex
import pytorch_lightning as pl
from pathlib import Path

# from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForSeq2SeqLM

from models.rag.vct0_model import VCT0Prefix

import time

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class RagModelNoDPRLora(pl.LightningModule):
    '''
    Class for RAG, re-implementation
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()
        print("MODEL KREIRAN")
        self.config = config
        self.data_loader = data_loader
        # self.retriever_tokenizer = data_loader.tokenizer
        self.generator_tokenizer = data_loader.tokenizer

        
        # Initialising question encoder - ostaje isto !
        # QueryEncoderModelClass = globals()[self.config.model_config.QueryEncoderModelClass]
        # QueryEncoderConfigClass = globals()[self.config.model_config.QueryEncoderConfigClass]
        # question_encoder_model_config = QueryEncoderConfigClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion)
        # self.question_encoder = QueryEncoderModelClass.from_pretrained(self.config.model_config.QueryEncoderModelVersion,
        #                                             config=question_encoder_model_config)
        # self.retiever_hidden_size = question_encoder_model_config.hidden_size

        # Initialising generator
        # GeneratorModelClass = globals()[self.config.model_config.ModelClass] # T5ForConditionalGeneration ! (imported class)
        # GeneratorConfigClass = globals()[self.config.model_config.ConfigClass]
        # generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.ModelVersion)
        # self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.ModelVersion,
        #                                             config=generator_model_config)

        GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.model_config.ConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.ModelVersion)
        self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.ModelVersion,config=generator_model_config)
        
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.generator = PeftModelForSeq2SeqLM(self.generator, peft_config)
        self.generator.print_trainable_parameters()

        # self.generator = GeneratorModelClass.from_pretrained("google/flan-t5-large", config=generator_model_config)
        # self.question_encoder.resize_token_embeddings(len(self.retriever_tokenizer))
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        
        # self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        self.lm_embedding_size = self.generator.model_dim # dimesnion of hidden state of lm model !
        self.use_img_embd = False
        self.use_ROI_embd = config.model_config.mlp.include_ROI_image_embeddings == 1


        if config.model_config.mlp.include_image_embeddings == 1:
            self.use_img_embd = True
            self.prefix_length = self.config.model_config.mlp.prefix_length

            print("\n\n Using MLP \n\n")
            self.clip_project = MLP(
                (
                    self.config.model_config.mlp.prefix_size,
                    (self.lm_embedding_size * self.prefix_length) // 2,
                    self.lm_embedding_size * self.prefix_length,
                )
            )
            print(self.clip_project)

            if  not config.model_config.mlp.checkpoint_path == None:
                checkpoint_path = config.model_config.mlp.checkpoint_path
                if not os.path.exists(checkpoint_path):
                    print("No checkpoint exists from '{}'. Skipping...".format(checkpoint_path))
                else:
                    print("Loading checkpoint from '{}'".format(checkpoint_path))
                    print('Weights before loading')
                    for n, p in self.clip_project.named_parameters():
                        print(n,p)
                    # checkpoint = torch.load(checkpoint_path)
                    # print(checkpoint.keys())
                    # self.vct0 = VCT0Prefix(prefix_length = self.prefix_length, model_version = self.config.model_config.ModelVersion)
                    # self.vct0.load_state_dict(checkpoint["state_dict"], strict=False)
                    # self.clip_project.load_state_dict(self.vct0.clip_project.state_dict())
                    # print("Loaded?")
                    checkpoint_path = Path(config.model_config.mlp.checkpoint_path)
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                    with torch.no_grad():
                        self.clip_project.model[0].weight.copy_(checkpoint['state_dict']["model.clip_project.model.0.weight"])
                        self.clip_project.model[0].bias.copy_(checkpoint['state_dict']["model.clip_project.model.0.bias"])
                        self.clip_project.model[2].weight.copy_(checkpoint['state_dict']["model.clip_project.model.2.weight"])
                        self.clip_project.model[2].bias.copy_(checkpoint['state_dict']["model.clip_project.model.2.bias"])
                    print('Weights after loading')
                    for n, p in self.clip_project.named_parameters():
                        print(n,p)
                    print('Loadeed??')

    def prepare_inputs_for_generator(self, 
                input_text_sequences, retrieved_docs, labels, n_docs=None):
        
        if n_docs is None:
            n_docs = self.config.data_loader.additional.num_knowledge_passages
        
        batch_size = len(input_text_sequences)

        extended_input_text_sequences = []

        for index, input_text_sequence in enumerate(input_text_sequences):
            scores = []
            for doc in retrieved_docs[index]:
                extended_input_text_sequences.append(
                    ' '.join([input_text_sequence, doc['content']])
                )
                # print("Tekst")
                # print(extended_input_text_sequences[-1])
                scores.append(doc['score'])

        targets = labels

        encoding = self.generator_tokenizer([sequence for sequence in extended_input_text_sequences],
                                    padding='longest',
                                    max_length=self.config.data_loader.additional.max_decoder_source_length,
                                    truncation=True,
                                    return_tensors="pt")
        generator_input_ids, generator_attention_mask = encoding.input_ids, encoding.attention_mask
        generator_input_ids = generator_input_ids.to(labels.device)
        generator_attention_mask = generator_attention_mask.to(labels.device)
        generator_decoder_input_ids = self.generator._shift_right(targets)

        return EasyDict(
            generator_input_text_sequences=extended_input_text_sequences,
            generator_input_ids=generator_input_ids,
            generator_attention_mask=generator_attention_mask,
            generator_decoder_input_ids=generator_decoder_input_ids,
            generator_labels=targets,
        )

    def insert_prefix_into_inputs(self, batch_size, no_documents, prefix_embd, input_text_ids, input_text_att_mask, labels):
        img_embd_projection_len = prefix_embd.shape[1] # koliko tokena od img embedinga
        # print("Tokeni od embedinga slike: ", img_embd_projection_len)
        input_text_ids_len = input_text_ids.shape[1] # koliko tokena za ostatak teksta
        # print("Broj tokena za sav ostali tekst: ", input_text_ids_len)
        output_seq_length = img_embd_projection_len + input_text_ids_len
        # print("input_text_ids_shape: ", input_text_ids.shape)
        # print("prefix_embd_shape: ", prefix_embd.shape)

        # Generate text embeddings
        input_text_embd = self.generator.shared(input_text_ids) # projekcija id-a u text embedinge
        # print("input_text_embd_shape: ", input_text_embd.shape)

        # print("POCINJEMO: ")
        embedding_out = torch.ones((no_documents*batch_size, output_seq_length, self.lm_embedding_size), device=labels.device) * -100
        # print("embedding_out_shape", embedding_out.shape)
        attention_mask_out = torch.ones((no_documents*batch_size, output_seq_length), dtype=int, device=labels.device) * -100
        text_mask = torch.zeros((no_documents*batch_size, output_seq_length), dtype=int, device=labels.device)

        text_mask[:, img_embd_projection_len + torch.arange(input_text_ids_len)] = 1 # set 1 to text indexes

        text_embedding_inds = text_mask.bool()
        prefix_embedding_inds = ~text_mask.bool()

        embedding_out[text_embedding_inds] = input_text_embd.view(-1, self.lm_embedding_size)
        # print("input_text_embd_shape: ", input_text_embd.shape)

        for i in range(batch_size):
            embedding_out[i*no_documents:(i+1)*no_documents, torch.arange(img_embd_projection_len)] = prefix_embd[i].view(-1, self.lm_embedding_size)

        # embedding_out[prefix_embedding_inds] = prefix_embd.view(-1, self.lm_embedding_size)
        # print("prefix_embd: ", prefix_embd.view(-1, self.lm_embedding_size).shape)

        attention_mask_out[text_embedding_inds] = input_text_att_mask.view(-1)
        attention_mask_out[prefix_embedding_inds] = 1

        # print("attention_mask_out[text_embedding_inds]", attention_mask_out[text_embedding_inds])
        # print("attention_mask_out[prefix_embedding_inds]", attention_mask_out[prefix_embedding_inds])

        return embedding_out, attention_mask_out

    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      prefix: torch.Tensor,
                      ROIprefix: torch.Tensor,
                    **kwargs):
        
        batch_size = input_ids.shape[0]

        # # Retrieve docs for given question inputs
        # retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences)
        # retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        
        # answers = kwargs.get('answers', None)
        # assert answers is not None
        # get_retrieval_labels_results = self.get_retrieval_labels(
        #     batch_answers=answers,
        #     batch_retrieved_docs=retrieved_docs,
        # )
        # retrieval_labels = get_retrieval_labels_results.retrieval_labels


        n_docs = 1
        # if 'add_null_document' in self.config.model_config.modules:
        #     n_docs += 1
        
        # if 'force_existence' in self.config.model_config.modules:
        #     # Force the label to be in the retrieved document
        #     selected_answers = get_retrieval_labels_results.selected_answers
        #     target_encoding = self.generator_tokenizer(selected_answers,
        #             padding='longest',
        #             max_length=self.config.data_loader.additional.max_target_length,
        #             truncation=True)
        #     labels = target_encoding.input_ids
        #     labels = torch.LongTensor(labels).type_as(input_ids)
        # else:
        #     labels = labels.repeat_interleave(n_docs, 0)


        # prepare inputs for generator
        # generator_inputs = self.prepare_inputs_for_generator(input_text_sequences=input_text_sequences,
        #                                     retrieved_docs=retrieved_docs,
        #                                     labels=labels, n_docs=n_docs)

        # print("GENERATOR INPUT: ", generator_inputs)

        if self.use_img_embd:
            if not self.use_ROI_embd:
                prefix_projections = self.clip_project(prefix).view(
                    -1, self.prefix_length, self.lm_embedding_size
                )
            else:
                # print("ROIprefix shape before: ", ROIprefix.shape)
                batch_size_, ROI_number_, _, input_dim_ = ROIprefix.shape
                # print("PRE: ", ROIprefix[0:2])

                ROIprefix = ROIprefix.reshape(batch_size_ * ROI_number_, 1, input_dim_)
                # print("ROIprefix shape after: ", ROIprefix.shape)

                # print("POSLE: ", ROIprefix[0:2])

                prefix_projections = self.clip_project(ROIprefix).view(
                    -1, self.prefix_length, self.lm_embedding_size
                )

                # print("PROSAO")
                # print(prefix_projections.shape)
                # print("Pre: ", prefix_projections[0:6])
                # prefix_projections = prefix_projections.reshape(batch_size_, ROI_number_, self.prefix_length, self.lm_embedding_size)
                # prefix_projections = prefix_projections.transpose(1, 2).contiguous().view(batch_size, -1, self.lm_embedding_size)
                prefix_projections = prefix_projections.view(batch_size, -1, self.lm_embedding_size)
                # print("OPTION 2")
                # print(prefix_projections.shape)
                # print("Posle: ", prefix_projections[0:6])


            joint_embeddings, joint_attention_masks = self.insert_prefix_into_inputs(
                                                    batch_size=batch_size, 
                                                    no_documents=1, 
                                                    input_text_ids=input_ids, 
                                                    input_text_att_mask=attention_mask, 
                                                    prefix_embd=prefix_projections,
                                                    labels=labels)

            generator_outputs = self.generator(
                                inputs_embeds=joint_embeddings,
                                attention_mask=joint_attention_masks,
                                labels=labels)

        else:
            generator_outputs = self.generator(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
        
        return EasyDict(loss=generator_outputs.loss)


    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      prefix: torch.Tensor,
                      ROIprefix: torch.Tensor,
                      **kwargs):

        batch_size = input_ids.shape[0]
        
        # Retrieve docs for given question inputs
        # retrieval_results = self.retrieve(input_ids, attention_mask, labels, question_ids, input_text_sequences)
        # retrieved_docs, doc_scores = retrieval_results.retrieved_docs, retrieval_results.doc_scores
        

        # if n_docs is None:
        #     n_docs = self.config.data_loader.additional.num_knowledge_passages
        #     if 'add_null_document' in self.config.model_config.modules:
        #         n_docs += 1

        # populate labels
        # labels = labels.repeat_interleave(n_docs, 0)

        # # prepare inputs for generator
        # generator_inputs = self.prepare_inputs_for_generator(input_text_sequences=input_text_sequences,
        #                                     retrieved_docs=retrieved_docs,
        #                                     labels=labels,
        #                                     n_docs=n_docs)
        
        if self.use_img_embd:
            if not self.use_ROI_embd:
                prefix_projections = self.clip_project(prefix).view(
                    -1, self.prefix_length, self.lm_embedding_size
                )
            else:
                # print("ROIprefix shape before: ", ROIprefix.shape)
                batch_size_, ROI_number_, _, input_dim_ = ROIprefix.shape
                # print("PRE: ", ROIprefix[0:2])

                ROIprefix = ROIprefix.reshape(batch_size_ * ROI_number_, 1, input_dim_)
                # print("ROIprefix shape after: ", ROIprefix.shape)

                # print("POSLE: ", ROIprefix[0:2])

                prefix_projections = self.clip_project(ROIprefix).view(
                    -1, self.prefix_length, self.lm_embedding_size
                )

                # print("PROSAO")
                # print(prefix_projections.shape)
                # print("Pre: ", prefix_projections[0:6])
                # prefix_projections = prefix_projections.reshape(batch_size_, ROI_number_, self.prefix_length, self.lm_embedding_size)
                # prefix_projections = prefix_projections.transpose(1, 2).contiguous().view(batch_size, -1, self.lm_embedding_size)
                prefix_projections = prefix_projections.view(batch_size, -1, self.lm_embedding_size)
                # print("OPTION 2")
                # print(prefix_projections.shape)
                # print("Posle: ", prefix_projections[0:6])            
            # print("Input text", generator_inputs.generator_input_text_sequences[0])
            joint_embeddings, joint_attention_masks = self.insert_prefix_into_inputs(
                                                    batch_size=batch_size, 
                                                    no_documents=1, 
                                                    input_text_ids=input_ids, 
                                                    input_text_att_mask=attention_mask, 
                                                    prefix_embd=prefix_projections,
                                                    labels=input_ids)

            # Get encoder outputs first
            test_batch = EasyDict({
                'inputs_embeds': joint_embeddings,
                'attention_mask': joint_attention_masks,
            })

        else:
            # Get encoder outputs first
            test_batch = EasyDict({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })

        encoder_outputs = self.generator.encoder(
            **test_batch
        )

        # Get decoder outputs from encoder_outputs
        test_batch = {
            'encoder_outputs': encoder_outputs,
            "max_length": self.config.data_loader.additional.max_target_length,
        }
        generation_outputs = self.generator.generate(**test_batch)
        
        return generation_outputs