from scipy.stats import spearmanr, pearsonr

def evaluate_predictions(preds,labels,task):
    if task == 'DPT':
        return evaluate_dpt_predictions(preds,labels)
    else:
        return evaluate_attr_predictions(preds,labels)

def evaluate_dpt_predictions(preds,labels):
    code_to_idx = {'data':0,'people':1,'things':2}
    results = {}
    str_preds = preds.astype(str).apply(to_three_digit)
    results['Overall accuracy'] = (labels == str_preds).mean()
    for code, idx in code_to_idx.items():
        code_preds = str_preds.str.slice(start=idx,stop=idx+1).astype(int)
        code_labels = labels.str.slice(start=idx,stop=idx+1).astype(int)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        code_corr = pearsonr(code_preds,code_labels)[0]
        code_rank_corr = spearmanr(code_preds,code_labels)[0]
        results[code +' accuracy'] = code_accuracy
        results[code + ' mean distance'] = code_dist
        results[code + ' correlation'] = code_corr
        results[code + ' rank correlation'] = code_rank_corr
    return results

def evaluate_attr_predictions(preds,labels):
    code_to_idx = {'GED':0,'EHFCoord':1,'FingerDexterity':2,'DCP':3,'STS':4}
    results = {}
    str_preds = preds.astype(str)
    results['Overall accuracy'] = (labels == str_preds).mean()
    for code, idx in code_to_idx.items():
        code_preds = str_preds.apply(lambda x: x[idx]).astype(float)
        code_labels = labels.apply(lambda x: x[idx]).astype(float)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        code_corr = pearsonr(code_preds,code_labels)[0]
        code_rank_corr = spearmanr(code_preds,code_labels)[0]
        results[code +' accuracy'] = code_accuracy
        results[code + ' mean distance'] = code_dist
        results[code + ' correlation'] = code_corr
        results[code + ' rank correlation'] = code_rank_corr
    return results

def to_three_digit(code):
    if len(code) == 1:
        code = '00'+code
    elif len(code) == 2:
        code = '0'+code
    return code
