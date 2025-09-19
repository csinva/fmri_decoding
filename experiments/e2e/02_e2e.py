import joblib 
import random
import numpy as np
import torch
import pickle
import os
import json
import copy
from decoding.e2e.end2end_model import End2End_model
import sys
from decoding.e2e.config import get_config, REPO_DIR
from decoding.language_generation.data import FMRI_dataset
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
if __name__ == '__main__':
    args = get_config()
    print(args)
    save_name = args['results_path']
    for key in args.keys():
        if key not in ['cuda']:
            save_name += key+'('+str(args[key])+')_'
    save_name = save_name[:-1]
    dataset_class = FMRI_dataset
    dataset_name = args['task_name'].split('_')[0]
    subject_name = args['task_name'].split('_')[1]
    if 'example' not in args['task_name']:
        args['dataset_path'] = os.path.join(args['dataset_path'], dataset_name)
    dataset_path = args['dataset_path']
    # setup model path
    args['llm_model_path'] = args['checkpoint_path']
    args['word_rate_model_path'] = os.path.join(
        # REPO_DIR,
        # f'word_rate/results/{args["task_name"]}_{args["model_name"]}') # only use huth model for word rate
        args['results_path'],
        f'{args["task_name"]}_huth_encoding') # only use huth model for word rate
    
    if 'Huth' in args['task_name']:
        input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
        decoding_model = End2End_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Pereira' in args['task_name']:
        input_dataset = pickle.load(open(f'{dataset_path}/{subject_name}.wq.pkl','rb'))
        decoding_model = End2End_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)
    elif 'Narratives' in args['task_name']:
        u2s = json.load(open(os.path.join(REPO_DIR, 'dataset_info', 'u2s.json')))
        args['Narratives_stories'] = u2s[f'sub-{subject_name}']
        input_dataset = {}
        for story_name in args['Narratives_stories']:
            input_dataset[story_name] = pickle.load(open(f'{dataset_path}/{story_name}.wq.pkl','rb'))
        decoding_model = End2End_model(args)
        dataset = dataset_class(input_dataset, args, tokenizer = decoding_model.tokenizer, decoding_model = decoding_model)

    print('dataset initialized')

    if args['mode'] in ['end2end',]:
        decoding_model.args['load_check_point'] = True
        decoding_model.load_check_point()
        # decoding_model.test(dataset.test_dataset, args['output'])
        decoding_model.test_beam(dataset.test_dataset, file_name='e2e_' + args['output'])
        

