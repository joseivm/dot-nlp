import pandas as pd
import os
import csv
import random
import sys
sys.path.append(os.path.abspath("/home/joseivm/dot-nlp/code/features"))
import feature_builder as fb

def evaluate_predictions(preds,labels):
    code_to_idx = {'data':0,'people':1,'things':2}
    results = {}
    results['Overall accuracy'] = (labels == preds).mean()
    for code, idx in code_to_idx.iteritems():
        code_preds = preds.str.slice(start=idx,end=idx+1).astype(int)
        code_labels = labels.str.slice(start=idx,end=idx+1).astype(int)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        results[code +' accuracy'] = code_accuracy
        results[code + ' mean distance'] = code_dist

    return results
