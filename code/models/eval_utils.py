from scipy.stats import spearmanr, pearsonr
import ast

HML_dict = {'FingerDexterity':[3,3.5],'EHFCoord':[4,5]}

def evaluate_predictions(preds,labels,task):
    if task == 'DPT':
        return evaluate_dpt_predictions(preds,labels)
    else:
        return evaluate_attr_predictions(preds,labels)

def evaluate_dpt_predictions(preds,labels):
    code_to_idx = {'data':0,'people':1,'things':2}
    results = []
    str_preds = preds.astype(str).apply(to_three_digit)
    labels = labels.str.slice(start=1)
    results['Overall accuracy'] = (labels == str_preds).mean()
    for code, idx in code_to_idx.items():
        code_preds = str_preds.str.slice(start=idx,stop=idx+1).astype(int)
        code_labels = labels.str.slice(start=idx,stop=idx+1).astype(int)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        code_corr = pearsonr(code_preds,code_labels)[0]
        code_rank_corr = spearmanr(code_preds,code_labels)[0]
        results.append({'Attribute':code,'Accuracy':code_accuracy,
                        'Correlation':code_corr,'Rank Correlation':code_rank_corr})

    return pd.DataFrame(results)

def evaluate_attr_predictions(preds,labels):
    code_to_idx = {'GED':0,'EHFCoord':1,'FingerDexterity':2,'DCP':3,'STS':4}
    results = []
    # preds = preds.apply(lambda x: ast.literal_eval(x))
    labels = labels.apply(lambda x: ast.literal_eval(x))
    results['Overall accuracy'] = (labels == preds).mean()
    for code, idx in code_to_idx.items():
        code_preds = preds.apply(lambda x: x[idx]).astype(float)
        code_labels = labels.apply(lambda x: x[idx]).astype(float)
        code_accuracy = (code_preds == code_labels).mean()
        code_dist = (code_preds - code_labels).abs().mean()
        code_corr = pearsonr(code_preds,code_labels)[0]
        code_rank_corr = spearmanr(code_preds,code_labels)[0]
        if code in HML_dict.keys():
            quantiles = HML_dict[code]
            pred_low = code_preds.apply(lambda x: x < quantiles[0])
            pred_medium = code_preds.apply(lambda x: x >= quantiles[0] and x < quantiles[1])
            pred_high = code_preds.apply(lambda x: x >= quantiles[1])
            pred_hml = code_low + code_medium + code_high

            label_low = code_labels.apply(lambda x: x < quantiles[0])
            label_medium = code_labels.apply(lambda x: x >= quantiles[0] and x < quantiles[1])
            label_high = code_labels.apply(lambda x: x >= quantiles[1])
            label_hml = label_low + label_medium + label_high

            hml_accuracy = (pred_hml == label_hml).mean()
            results.append({'Attribute':code,'Accuracy':code_accuracy,'HML Accuracy':hml_accuracy,
                            'Correlation':code_corr,'Rank Correlation':code_rank_corr})
        else:
            results.append({'Attribute':code,'Accuracy':code_accuracy,'HML Accuracy':0,
                        'Correlation':code_corr,'Rank Correlation':code_rank_corr})
    return pd.DataFrame(results)

def to_three_digit(code):
    if len(code) == 1:
        code = '00'+code
    elif len(code) == 2:
        code = '0'+code
    return code
