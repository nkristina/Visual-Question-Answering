import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging
logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data

from data_loader_manager.data_loader_wrapper import DataLoaderWrapper
from data_loader_manager.datasets import *

from torchvision.utils import make_grid, save_image
from torchvision.transforms import Compose, ToTensor, Resize, PILToTensor
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class DataLoaderBLIP2_ROI(DataLoaderWrapper):
    '''
    Data loader for OKVQA dataset
    '''

    def __init__(self, config):
        DataLoaderWrapper.__init__(self, config)



    def LoadOKVQAData(self, module_config):
        '''
        Load vqa data into self.data.okvqa_data
        {
          "type": "LoadOKVQAData", "option": "default",
          "config": {
            "vqa_data_path": {
                "question_files":{
                    "train": "..",
                    "test": "..",
                },
                "annotation_files": {
                    "train": "..",
                    "test": "..",
                },
            },
            "image_data_path": {
                "train": "..",
                "valid": "..",
            },
        },
        '''
        ######################
        #   Read OK-VQA data
        ######################
        def most_frequent(List):
            return max(set(List), key = List.count)
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        def _transform():
            return Compose([
                Resize((400,400), interpolation=InterpolationMode.BICUBIC),
                _convert_image_to_rgb,
                PILToTensor(),
                ])
        
        answer_candidate_list = []
        vqa_helpers = EasyDict({
            'train': VQA(module_config.config.vqa_data_path.annotation_files.train, 
                            module_config.config.vqa_data_path.question_files.train),
            'test': VQA(module_config.config.vqa_data_path.annotation_files.test, 
                            module_config.config.vqa_data_path.question_files.test),
        })
        
        self.data.okvqa_data = EasyDict({
            'train': {},
            'test': {},
            'lookup': {},
            'vqa_helpers': vqa_helpers,
        })
        image_preprocessor = _transform()
        for data_split, vqa_helper in vqa_helpers.items():
            vqa_helper.createIndex()
            vqa_helper.info()

            # For each data split, prepare dataset
            self.data.okvqa_data[data_split] = load_cached_data(self.config, '{}_data_preprocessed_ROI2_half'.format(data_split))
            if not self.data.okvqa_data[data_split]:
                # This split data is not cached
                self.data.okvqa_data[data_split] = EasyDict({}) # re-initialise
                # Create list of images from helper
                img_data_path = module_config.config.image_data_path[data_split]
                ROI_data_path = module_config.config.ROI_data_path[data_split]
                img_list = []
                for imgId in vqa_helper.imgToQA.keys():
                    dataSubType = vqa_helper.dataSubType
                    imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
                    img_path = os.path.join(img_data_path, imgFilename)
                    img_list.append((imgId, img_path))
                    if self.config.data_loader.dummy_dataloader:
                        # Load only a few samples for testing
                        if len(img_list) > 20:
                            break
                
                # Create entries for each question and related answers
                self.data.okvqa_data[data_split].data_items = []
                for imgId, img_path in tqdm(img_list):
                    # avoid error in splitting: must remove ".." in "../path/to/file"
                    # img_key = img_p.replace('..', '').split('.')[0].split('_')[-1]
                    img_key = imgId
                    img_key_str = str(img_key)
                    #removed since oscar captions are not loaded
                    #img_caption = self.data.caption_features.get(img_key_str, None)
                    #if img_caption is not None: 
                    #    img_caption = img_caption[0] 
                    #else: 
                    #    logger.debug('No caption found for {}!'.format(img_key))
                    
                    img_key_full = str(img_key).zfill(12)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = image_preprocessor(Image.fromarray(img))                    
                    
                    related_question_ids = vqa_helper.getQuesIds(imgIds=[imgId])
                    related_answers = vqa_helper.loadQA(ids=related_question_ids)
                    related_question_and_answers = vqa_helper.returnQA(related_answers)
                    
                    for question_and_answer in related_question_and_answers:
                        # For each question and related answers, create an entry
                        entry_data = EasyDict()
                        entry_data.answers = list(question_and_answer['answers'].values())
                        entry_data.answers = [answer for answer in entry_data.answers if answer != '']
                        entry_data.gold_answer = most_frequent(entry_data.answers)
                        entry_data.question = question_and_answer['question']
                        entry_data.question_id = question_and_answer['question_id']
                        entry_data.img_path = img_path
                        entry_data.img_key_full = img_key_full
                        entry_data.img_key = img_key
                        # entry_data.img = img

                        # process related ROIs

                        # Locate and stack related images from another folder based on naming convention
                        # related_imgs_folder = 'path_to_related_images_folder'  # Modify with your path
                        related_imgs = [f for f in os.listdir(ROI_data_path) 
                                        if f.startswith(f"{str(imgId).zfill(12)}_{str(entry_data.question_id)}")]
                        
                        images_list = [img]  # Start with the main image

                        for related_img_name in related_imgs:
                            related_img_path = os.path.join(ROI_data_path, related_img_name)
                            related_img = cv2.imread(related_img_path)
                            related_img = cv2.cvtColor(related_img, cv2.COLOR_BGR2RGB)
                            related_img = image_preprocessor(Image.fromarray(related_img))
                            images_list.append(related_img)

                        entry_data.img = images_list  # This will be a list of images

                        #entry_data.img_caption = img_caption
                        self.data.okvqa_data[data_split].data_items.append(entry_data)

                        # Collect answer candidates for evaluation
                        for ans in list(question_and_answer['answers'].values()):
                            if ans not in answer_candidate_list:
                                answer_candidate_list.append(ans)
                                # if data_split == 'test':
                                #     print(ans, 'is added from test set!')
                
                # After building the data split, save to cache
                save_cached_data(self.config, self.data.okvqa_data[data_split], '{}_data_preprocessed_ROI2_half'.format(data_split))

            for entry_data in self.data.okvqa_data[data_split].data_items:
                self.data.okvqa_data['lookup'][str(entry_data.question_id)] = entry_data

            

            # Report statistics
            logger.info('[Data statistics] split: {}  entries: {}'.format(
                data_split,
                len(self.data.okvqa_data[data_split].data_items)))

        # Save answer candidate list
        self.data.okvqa_data.answer_candidate_list = answer_candidate_list

        self.data.vqa_data = self.data.okvqa_data

    def set_dataloader(self):
        """
        This function wraps datasets into dataloader for trainers
        """
        train_dataset_dict = {
            'data': self.data.vqa_data.train,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'train',
        }
        self.train_dataset = globals()[self.config.data_loader.dataset_type](self.config, train_dataset_dict)
        # for i in self.train_dataset:
        #     pprint(i)
        #     input()
        train_sampler = RandomSampler(self.train_dataset)
        # train_sampler = SequentialSampler(self.train_dataset)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.config.train.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            # num_workers=8,
        )
        # for i in self.train_dataloader:
        #     print(i)
        #     input()
        
        test_dataset_dict = {
            'data': self.data.vqa_data.test,
            'answer_candidate_list': self.data.vqa_data.answer_candidate_list,
            'tokenizer': self.tokenizer,
            'decoder_tokenizer': self.decoder_tokenizer,
            'feature_extractor': self.feature_extractor,
            'mode': 'test',
        }
        self.test_dataset = globals()[self.config.data_loader.dataset_type](self.config, test_dataset_dict)

        test_sampler = SequentialSampler(self.test_dataset)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            batch_size=self.config.valid.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            # num_workers=4,
        )
    
        logger.info('[Data Statistics]: training data loader: {};  test data loader: {}'.format(
                                len(self.train_dataloader), 
                                len(self.test_dataloader)))