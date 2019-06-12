import pandas as pd
import os
import csv
import random
import sys
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(os.path.join(PROJECT_DIR,"code/features"))
import feature_builder as fb

def evaluate_predictions(preds,labels):
    code_to_idx = {'data':0,'people':1,'things':2}
    results = {}
    str_preds = preds.astype(str).apply(to_three_digit)
    results['Overall accuracy'] = (labels == str_preds).mean()
    for code, idx in code_to_idx.items():
        code_preds = str_preds.str.slice(start=idx,stop=idx+1).astype(int)
        code_labels = labels.str.slice(start=idx,stop=idx+1).astype(int)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        results[code +' accuracy'] = code_accuracy
        results[code + ' mean distance'] = code_dist

    return results

def to_three_digit(code):
    if len(code) == 1:
        code = '00'+code
    elif len(code) == 2:
        code = '0'+code
    return code
