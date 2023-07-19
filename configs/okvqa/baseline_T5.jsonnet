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
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;
local mlp_lr = 1e-4;

local seed=2021;



local override = {
  "platform_type": "pytorch",
  "ignore_pretrained_weights": [],
  "experiment_name": "default_test",
  "seed": seed,
  "model_config": {
    "base_model": "T5",
    "ModelClass": "RagModelNoDPR",
    "TokenizerClass": "T5Tokenizer",
    "TokenizerModelVersion": "t5-large",
    "GeneratorModelClass": "T5ForConditionalGeneration",
    "ConfigClass": "T5Config",
    "ModelVersion": "t5-large",
    "pretrained": 1,
    "modules": [
    ],
    "SPECIAL_TOKENS":{
      "bos_token": "<PAD>",
      "pad_token": "<PAD>",
      "additional_special_tokens": ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>"],
    },
    "mlp": {
      "checkpoint_path": "nan", # "../data/ok-vqa/mlp_models/model_08.ckpt",
      "prefix_length": 10,
      "prefix_size": 768,  # dimensions of clip embedding
      "include_image_embeddings": 1,
    },
    "input_modules": {
      "module_list":[
        {"type": "QuestionInput",  "option": "default", 
                  "separation_tokens": {'start': '<BOQ>', 'end': '<EOQ>'}},
        // {"type": "TextBasedVisionInput",  "option": "caption",
        //           "separation_tokens": {'start': '<BOC>', 'end': '<EOC>'}},
        // {"type": "TextBasedVisionInput",  "option": "object", 
        //           "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
        //           "separation_tokens": {'start': '<BOV>', 'sep': '<SOV>', 'end': '<EOV>'}},
        {"type": "EmbeddingInput",  "option": "default"},
      ],
      "postprocess_module_list": [
        {"type": "PostProcessClipEmbeddings", "option": "default"},
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
        {"type": "PostProcessOutputTokenization", "option": "default"},
      ],
    },
  },
  "cache":{
    "regenerate":{
      "vinvl_feature_preprocessed": 0,
      "ocr_feature_preprocessed": 0,
      "train_data_preprocessed": 0,
      "test_data_preprocessed": 0,
      "clip_embeddings": 0,
    },
  },
  "data_loader": {
    "type": "DataLoaderOKVQA",
    "dataset_type": "OKVQADataset",
    "dummy_dataloader": 0,
    "additional":{
      'max_source_length':512,
      'max_decoder_source_length': 512,
      'max_target_length':10,
    },
    "dataset_modules": {
      "module_list": [
        "LoadVinVLFeatures",
        "LoadGoogleOCRFeatures",
        "LoadOscarCaptionFeatures",
        "LoadOKVQAData",
        "LoadClipEmbeddings",
      ],
      "module_dict":{
      },
    },
  },
  "cuda": 0,
  "gpu_device":0,
  "train": {
    "type": "T5Executor",
    "epochs":train_epochs,
    "batch_size":train_batch_size,
    "lr": lr,
    "mlp_lr": mlp_lr,
    "retriever_lr": retriever_lr,
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
    {'name': 'compute_okvqa_scores'},
  ],
};

std.mergePatch(base_env, override)
