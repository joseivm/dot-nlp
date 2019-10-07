import pandas as pd
import csv
import random
import sys
import numpy as np
import torch
import json
import math

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

import os
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(os.path.join(PROJECT_DIR,"code/features"))
import feature_builder as fb
import eval_utils as eu
import utils

def train_model(args):
    train_data_dir = os.path.join(PROJECT_DIR,"data",args.train_year)
    eval_data_dir = os.path.join(PROJECT_DIR,"data",args.eval_year)
    model_dir = os.path.join(PROJECT_DIR,"models/joint",args.identifier)
    results_dir = os.path.join(PROJECT_DIR,"results/joint")
    output_dir = os.path.join(PROJECT_DIR,"output/joint")

    processor = fb.DOTProcessor()
    labels = processor.get_labels()
    num_labels = len(labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.lower_case)
    train_examples = processor.get_train_examples(train_data_dir)
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

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

    eval_examples = processor.get_dev_examples(eval_data_dir)
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

    df = pd.read_csv(eval_data_dir+'/dev.csv',header=None)
    df.columns = ['Code','Title','Description']
    df['DPT'] = preds
    df = df[['Title','Code','DPT']]
    labels = df['Code'].str.slice(start=4,stop=7)
    df.to_csv(os.path.join(output_dir,args.identifier+'_preds.csv'),index=False)
    result = eu.evaluate_predictions(df['DPT'],labels)
    output_eval_file = os.path.join(results_dir,args.identifier+ "_eval_results.txt")

    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

def train_model2(args):
    train_data_dir = os.path.join(PROJECT_DIR,"data",args.train_year)
    eval_data_dir = os.path.join(PROJECT_DIR,"data",args.eval_year)
    model_dir = os.path.join(PROJECT_DIR,"models/joint",args.identifier)
    results_dir = os.path.join(PROJECT_DIR,"results/joint")
    output_dir = os.path.join(PROJECT_DIR,"output/joint")

    processor = fb.DOTProcessor()
    labels = processor.get_labels()
    num_labels = len(labels)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.lower_case)
    train_examples = processor.get_train_examples(train_data_dir)
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
    min_eval_loss = math.inf

    device = torch.device("cuda:"+args.cuda_device if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
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

        else:
            eval_examples = processor.get_dev_examples(eval_data_dir)
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

            if eval_loss < min_eval_loss:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(model_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(model_dir)

                epochs_no_improve = 0
                min_eval_loss = eval_loss
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f'Finished training in {epoch} epochs')

def evaluate_model2(identifier,eval_year,cuda_device='2',test=True):
    device = 'cuda:' + cuda_device
    eval_type = 'test' if test else 'eval'
    results_dir = os.path.join(PROJECT_DIR,"results/joint")
    output_dir = os.path.join(PROJECT_DIR,"output/joint")
    model_dir = os.path.join(PROJECT_DIR,"models/joint",identifier)
    with open(os.path.join(model_dir,'settings.txt'),'r') as f:
        model_args = json.loads(f.read())

    eval_data_dir = os.path.join(PROJECT_DIR,"data",eval_year)

    processor = fb.DOTProcessor()
    labels = processor.get_labels()
    num_labels = len(labels)

    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_args['lower_case'])

    model.to(device)

    if test:
        eval_examples = processor.get_test_examples(eval_data_dir)
    else:
        eval_examples = processor.get_dev_examples(eval_data_dir)
    eval_features = processor.convert_examples_to_features(
        eval_examples, labels, model_args['max_seq_length'], tokenizer, model_args['output_mode'])


    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if model_args['output_mode'] == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif model_args['output_mode'] == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=model_args['eval_batch_size'])

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

        if model_args['output_mode'] == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif model_args['output_mode'] == "regression":
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
    if model_args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif model_args['output_mode'] == "regression":
        preds = np.squeeze(preds)
        preds = np.rint(preds)

    df = pd.read_csv(eval_data_dir+'/'+eval_type+ '.csv',header=None)
    df.columns = ['Code','Title','Description']
    df['DPT'] = preds
    df = df[['Title','Code','DPT']]
    labels = df['Code'].str.slice(start=4,stop=7)
    df.to_csv(os.path.join(output_dir,identifier+'_'+eval_year+'_'+eval_type+'_preds.csv'),index=False)
    result = eu.evaluate_predictions(df['DPT'],labels)
    output_eval_file = os.path.join(results_dir,identifier+'_' + eval_year + "_"+eval_type +"_results.txt")

    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

def evaluate_model(identifier,eval_year):
    results_dir = os.path.join(PROJECT_DIR,"results/joint")
    output_dir = os.path.join(PROJECT_DIR,"output/joint")
    pred_df = pd.read_csv(output_dir+'/'+identifier+'_preds.csv')
    results_dir = os.path.join(PROJECT_DIR,"results")
    data_dir = os.path.join(PROJECT_DIR,"data",eval_year)
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
model_dir = os.path.join(PROJECT_DIR,"models/joint",args.identifier)
os.makedirs(model_dir,exist_ok=True)
json.dump(settings, open(model_dir+'/settings.txt', 'w'), indent=0)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_model2(args)
evaluate_model2(args.identifier,'1977',args.cuda_device)
evaluate_model2(args.identifier,'1965',args.cuda_device)
