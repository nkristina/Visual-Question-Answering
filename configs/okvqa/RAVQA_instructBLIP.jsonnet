local base_env = import 'base_env.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local MLP_lr = 1e-4;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;
local mlp_lr = 5e-4;

local seed=2021;

local RAG_data_config_full = {
  "index_passages_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset",
  "index_path": "../data/ok-vqa/pre-extracted_features/faiss/ok-vqa-passages-full-caption-pretrained-NewRun/my_knowledge_dataset_hnsw_index.faiss",
};

local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "RAG",
    "ModelClass": "RagModelInstructBLIP", // general class
    "TokenizerClass": "DPRQuestionEncoderTokenizer",  // question encoder tokenizer
    "TokenizerModelVersion": "facebook/dpr-question_encoder-single-nq-base", // question encoder tokenizer version
    
    "DecoderTokenizerClass": "InstructBlipProcessor",  // generator tokenizer
    "DecoderTokenizerModelVersion": "Salesforce/instructblip-flan-t5-xl", // generator tokenizer version

    "QueryEncoderModelClass": "DPRQuestionEncoder", // question encoder
    "QueryEncoderConfigClass": "DPRConfig", // question encoder
    // "QueryEncoderModelVersion": "facebook/dpr-question_encoder-single-nq-base",
    "QueryEncoderModelVersion": "/home/kn413/rds/rds-cvnlp-hirYTW1FQIw/wl356/projects/Retrieval-Augmented-Visual-Question-Answering/Experiments/Knowledge_Retriever_DPR_dim_768_inbatch_negative_caption_FullCorpus_NewRun/train/saved_model/epoch6/query_encoder",
    //"QueryEncoderModelVersion": "/additional_data/projects/RAVQA/Experiments/OKVQA_DPR_FullCorpus/train/saved_model/epoch6/query_encoder",
    
    "GeneratorModelClass": "InstructBlipForConditionalGeneration", // answer generator
    "GeneratorModelVersion": "Salesforce/instructblip-flan-t5-xl",
    "pretrained": 1,
    "UseBLIP2": 1,
    "UseInstructBLIP": 0,

    "modules": [
    ],
    "RAVQA_loss_type": "Approach6", // NoPR, Approach[1-6]

    "loss_ratio":{
      "nll_loss": 1,
      "additional_loss": 0,
      "rag_loss": 0,
    },
    "mlp": {
      "checkpoint_path": "nan", # "../data/ok-vqa/mlp_models/model_08.ckpt",
      "prefix_length": 10,
      "prefix_size": 768,  # dimensions of clip embedding
      "include_image_embeddings": 0,
    },
    "SPECIAL_TOKENS":{  // for query encoder
      // "bos_token": "<PAD>",
      // "pad_token": "<PAD>",
      "additional_special_tokens": [],
    },
    "DECODER_SPECIAL_TOKENS":{ // for answer generator
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": [],
    },
    "input_modules": {
      "module_list":[ # Promeni ovo u objasnjenje
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {'start': 'Question:', 'end': ''}},
        {"type": "TextBasedVisionInput",  "option": "caption",
                  "separation_tokens": {'start': 'Caption:', 'end': ''}},
        {"type": "TextBasedVisionInput",  "option": "object", 
                  "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
                  "separation_tokens": {'start': 'Objects:', 'sep': ',', 'end': ''}},  
      ],
      "postprocess_module_list": [
        {"type": "PostProcessInputTokenization", "option": "default"},
      ],
    },
    "decoder_input_modules": {
      "module_list":[],
      "postprocess_module_list": [],
    },
    "output_modules": {
      "module_list":[
        {"type": "GenerationOutput", "option": "default"},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessOutputTokenizationInstructBLIP", "option": "default"},
      ],
    },
  },
  "cache":{
    "regenerate":{
      "train_data_preprocessed_BLIP2_text": 0,
      "test_data_preprocessed_BLIP2_text": 0,
      "train_data_preprocessed_BLIP2": 0,
      "test_data_preprocessed_BLIP2": 0,
      "vinvl_feature_preprocessed": 0,
      "ocr_feature_preprocessed": 0,
      "train_data_preprocessed_txt": 0,
      "test_data_preprocessed_txt": 0,
      "train_data_preprocessed_dpr": 0,
      "test_data_preprocessed_dpr": 0,
      "clip_embeddings": 0,
      "qformer_embeddings": 0,
      "instructBLIP_embeddings": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQAWithKnowledge",
    "dataset_type": "OKVQADatasetBLIP2Text",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':10,
      'num_knowledge_passages': 5,
    },
    "dataset_modules": {
      "module_list": [
        "LoadVinVLFeatures",
        "LoadGoogleOCRFeatures",
        "LoadOscarCaptionFeatures",
        "LoadOKVQAData",
        "LoadGoogleSearchPassageData",
      ],
      "module_dict":{
      },
    },
    "index_files": RAG_data_config_full,
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "RagExecutorInstructBLIP",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "retriever_lr": retriever_lr,
    "MLP_lr": MLP_lr,
    "adam_epsilon": adam_epsilon,
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "save_interval":save_interval,
    "scheduler": "none",
    "additional": {
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": warmup_steps,
        "gradient_clipping": gradient_clipping,
    }
  },
  "valid": {
    "batch_size":valid_batch_size,
    "step_size":valid_step_size,
    "break_interval": break_interval,
    "additional": {
    },
  },
  "test": {
    "evaluation_name": "test_evaluation",
    "load_epoch": -1,
    "load_model_path": "",
    "load_best_model": 0,
    "batch_size": test_batch_size,
    "num_evaluation": 0,
    "additional": {
        "multiprocessing": 4,
    },
  },
  "metrics": [
    {'name': 'compute_exact_match'},
    {'name': 'compute_retrieval_metrics'},
    {'name': 'compute_okvqa_scores'},
  ],
};

std.mergePatch(base_env, override)
