# coding=utf-8
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os
import random
import sys
import math
import pandas as pd

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,AdamW,RobertaConfig,
  RobertaForSequenceClassification, RobertaTokenizer,get_linear_schedule_with_warmup)

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")

sys.path.append(os.path.join(PROJECT_DIR,"code/features"))
import roberta_feature_builder as rfb
import eval_utils as eu
import utils

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

PROCESSORS = {'DPT' : rfb.DPTProcessor, 'Attr' : rfb.AttributesProcessor}
TASK_YEARS = {'DPT': ['1965','1977'],'Attr':['1965','1991']}
TASK_CODES = {'DPT': ['data','people','things'], 'Attr': ['GED','EHFCoord','FingerDexterity','DCP','STS']}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_examples(args, code, processor, tokenizer, year,type='train'):
    output_mode = args.output_mode
    data_dir = os.path.join(PROJECT_DIR,'data',args.task_name,year)

    label_list = processor.get_labels(code)

    examples = processor.get_examples(data_dir,type,code)
    features = processor.convert_examples_to_features(examples,tokenizer,max_length=args.max_seq_length)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train_model(args,code):
    model_dir = os.path.join(PROJECT_DIR,"models",args.task_name,'separate',args.identifier,code)
    # Set seed
    set_seed(args)

    processor = PROCESSORS[args.task_name]()
    label_list = processor.get_labels(code)
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
    config = config_class.from_pretrained(args.model,
                                          num_labels=num_labels,
                                          cache_dir=None)
    tokenizer = tokenizer_class.from_pretrained(args.model,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir= None)
    model = model_class.from_pretrained(args.model,
                                        config=config,
                                        cache_dir=None)

    model.to(args.device)

    train_dataset = load_examples(args, code, processor,tokenizer, args.train_year,type='train')
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)

    print("Done Training")

def train(args, code, train_dataset, model, tokenizer):
    """ Train the model """
    train_data_dir = os.path.join(PROJECT_DIR,"data",args.task_name,args.train_year)
    model_dir = os.path.join(PROJECT_DIR,"models",args.task_name,'separate',args.identifier,code)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(args, os.path.join(model_dir, 'training_args.bin'))

    epochs_no_improve = 0
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    # Train!

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    min_eval_loss = math.inf
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            inputs['token_type_ids'] = None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        else:
            eval_loss = get_eval_loss(args, code, model, tokenizer)
            if eval_loss < min_eval_loss:
                # Save model checkpoint
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                min_eval_loss = eval_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve > args.patience:
                    print(f'Finished training in {epoch} epochs')
                    return

    return global_step, tr_loss / global_step

def get_eval_loss(args, code, model, tokenizer):
    eval_task = args.task_name
    processor = PROCESSORS[args.task_name]()
    eval_dataset = load_examples(args, code, processor, tokenizer, args.train_year,type='dev')

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            inputs['token_type_ids'] = None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    return eval_loss

def evaluate_model(args):
    model_dir = os.path.join(PROJECT_DIR,"models",args.task_name,'separate',args.identifier)

    tokenizer = RobertaTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.to(args.device)
    for year in TASK_YEARS[args.task_name]:
        evaluate(args, model, tokenizer,year,'test')

# this function will write out the model predictions to a file. Think about renaming
def evaluate(args, code, model, tokenizer, eval_year, eval_type):
    eval_output_dir = os.path.join(PROJECT_DIR,"output",args.task_name,'separate')
    eval_output_file = os.path.join(eval_output_dir,args.identifier+'_'+eval_year+'_'+eval_type+'_preds.csv')
    data_dir = os.path.join(PROJECT_DIR,'data',args.task_name, eval_year)

    processor = PROCESSORS[args.task_name]()
    eval_dataset = load_examples(args, code, processor, tokenizer, eval_year, type=eval_type)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}

            inputs['token_type_ids'] = None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)

    if os.path.exists(output_eval_file):
        df = pd.read_csv(output_eval_file)
    else:
        df = pd.read_csv(data_dir+'/'+eval_type+ '.csv')
    preds_name = 'pred_'+code
    df[preds_name] = preds
    df.to_csv(eval_output_file,index=False)

def evaluate_predictions(args):
    eval_results_dir = os.path.join(PROJECT_DIR,"results",args.task_name,'separate')
    eval_output_dir = os.path.join(PROJECT_DIR,"output",args.task_name,'separate')
    eval_output_file = os.path.join(eval_output_dir,args.identifier+'_'+eval_year+'_'+eval_type+'_preds.csv')
    data_dir = os.path.join(PROJECT_DIR,'data',args.task_name, eval_year)

    df = pd.read_csv(output_eval_file)
    preds_name = 'pred_'+args.task_name
    df[preds_name] = list(zip(df.pred_GED.astype(str),df.pred_EHFCoord.astype(str),
                          df.pred_FingerDexterity.astype(str),df.pred_DCP.astype(str),
                          df.pred_STS.astype(str)))
    df[preds_name] = df[preds_name].astype(str)
    df = df[['Title','Code','Definition',preds_name,args.task_name]]
    labels = df[args.task_name]
    df.to_csv(eval_output_file,index=False)
    result = eu.evaluate_predictions(df[preds_name],labels,args.task_name)
    output_eval_file = os.path.join(eval_results_dir,args.identifier+'_' + eval_year + "_"+eval_type +"_results.csv")
    result.to_csv(output_eval_file)

    print(result)

def main():
    parser = utils.roberta_parser()
    args = parser.parse_args()
    task_codes = TASK_CODES[args.task_name]
    for code in task codes:
        if not args.no_train:
            train_model(args,code)

        if not args.no_eval:
            evaluate_model(args,code)
            
    evaluate_predictions(args)

main()
