import pandas as pd
import csv
import random
import sys
import numpy as np
import torch
import json

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

import os
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(os.path.join(PROJECT_DIR,"code/features"))
import feature_builder as fb
import eval_utils as eu
import utils

def train_model(code, args):
    identifier = args.identifier
    data_dir = os.path.join(PROJECT_DIR,"data/1977")
    model_dir = os.path.join(PROJECT_DIR,"models/separate",identifier,code)
    os.makedirs(model_dir,exist_ok=True)
    output_dir = os.path.join(PROJECT_DIR,"output/separate")

    processor = fb.DOTProcessor()
    classification = args.output_mode == 'classification'
    labels = processor.get_labels(code,classification)
    num_labels = len(labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.lower_case)
    train_examples = processor.get_train_examples(data_dir,code)
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
    model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=cache_dir,
                  num_labels=num_labels)

    num_train_optimization_steps = int(len(train_examples)*args.num_train_epochs)

    train_features = processor.convert_examples_to_features(train_examples, labels, args.max_seq_length, tokenizer, args.output_mode)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    if args.output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            if args.output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif args.output_mode == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(model_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_dir)

    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args.lower_case)

    model.to(device)

    eval_examples = processor.get_dev_examples(data_dir,code)
    eval_features = processor.convert_examples_to_features(
        eval_examples, labels, args.max_seq_length, tokenizer, args.output_mode)


    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if args.output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        if args.output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif args.output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
        preds = np.rint(preds)
    df = pd.Series(preds)

    df.to_csv(os.path.join(output_dir,'_'.join([identifier,code,'preds.csv'])),
                                                                    index=False)

def combine_predictions(identifier):
    output_dir = os.path.join(PROJECT_DIR,"output/separate")
    results_dir = os.path.join(PROJECT_DIR,"results/separate")
    data_dir = os.path.join(PROJECT_DIR,"data/1977")
    df = pd.read_csv(data_dir+'/dev.csv',header=None)
    df.columns = ['Code','Title','Description']
    codes = ['data','people','things']
    for code in codes:
        pred_path = os.path.join(output_dir,'_'.join([identifier,code,'preds.csv']))
        code_preds = pd.read_csv(pred_path,header=None)[0]
        df.loc[:,code] = np.maximum(code_preds,0)

    df.drop(columns='Description',inplace=True)
    df['DPT'] = df['data'].map('{0:g}'.format)+df['people'].map('{0:g}'.format)+df['things'].map('{0:g}'.format)
    df[['Title','Code','DPT']].to_csv(output_dir+'/'+identifier+'_preds.csv')

def evaluate_model(identifier):
    results_dir = os.path.join(PROJECT_DIR,"results/separate")
    output_dir = os.path.join(PROJECT_DIR,"output/separate")
    pred_df = pd.read_csv(output_dir+'/'+identifier+'_preds.csv')
    data_dir = os.path.join(PROJECT_DIR,"data/1977")
    df = pd.read_csv(data_dir+'/dev.csv',header=None)
    df.columns = ['Code','Title','Description']

    labels = df['Code'].str.slice(start=4,stop=7)
    preds = pred_df['DPT']

    result = eu.evaluate_predictions(preds,labels)
    output_eval_file = os.path.join(results_dir, identifier+"_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

parser = utils.model_options_parser()
args = parser.parse_args()
settings = vars(args)
model_dir = os.path.join(PROJECT_DIR,"models/separate",args.identifier)
os.makedirs(model_dir,exist_ok=True)
json.dump(settings, open(model_dir+'/settings.txt', 'w'), indent=0)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_model('data',args)
train_model('people',args)
train_model('things',args)
combine_predictions(args.identifier)
evaluate_model(args.identifier)
