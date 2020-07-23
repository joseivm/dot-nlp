import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, pearsonr

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DATA_DIR = os.path.join(PROJECT_DIR,'data','Attr')

# Input files/dirs
PREDS_DIR = os.path.join(PROJECT_DIR,'output','Full')

# Output files/dirs
RESULTS_DIR = os.path.join(PROJECT_DIR,'results','Full')

def save_results(test_year):
    filename = os.path.join(PREDS_DIR,'domain_1965md_'+test_year+'_full_data_preds.csv')
    df = pd.read_csv(filename)
    df = df.loc[df.pred_DPT.notna(),:]
    df['Data'] = df.DPT.str.slice(1,2).astype(int)
    df['People'] = df.DPT.str.slice(2,3).astype(int)
    df['Things'] = df.DPT.str.slice(3,4).astype(int)
    df['pred_Data'] = df['pred_DPT'].str.slice(1,2).astype(int)
    df['pred_People'] = df['pred_DPT'].str.slice(2,3).astype(int)
    df['pred_Things'] = df['pred_DPT'].str.slice(3,4).astype(int)
    results = []
    attributes = ['Data','People','Things','DCP','STS','GED','SVP','EHFCoord','FingerDexterity']
    for attr in attributes:
        pred_attr = 'pred_'+attr
        labels = df.loc[(df[attr].notna()) & (df[pred_attr].notna()),attr]
        preds = df.loc[(df[attr].notna()) & (df[pred_attr].notna()),pred_attr]
        obs = len(preds)
        accuracy = (labels == preds).mean()
        mae = (preds - labels).abs().mean()
        corr = pearsonr(preds,labels)[0]
        rank_corr = spearmanr(preds,labels)[0]
        results.append({'Attribute':attr,'Accuracy':accuracy,
                        'Correlation':corr,'Rank Correlation':rank_corr,'MAE': mae,'Obs':obs})

    results = pd.DataFrame(results)
    results.loc[results.Attribute == 'FingerDexterity','Attribute'] = 'FingerDext'
    outfile = os.path.join(RESULTS_DIR,test_year+'_results.csv')
    results.to_csv(outfile,index=False,float_format='%.3f')

    stable = results.loc[results.Attribute.isin(attributes[:5])]
    stable_outfile = os.path.join(RESULTS_DIR,test_year+'_stable_results.csv')
    stable.to_csv(stable_outfile,index=False,float_format='%.3f')

    unstable = results.loc[~results.Attribute.isin(attributes[:5])]
    unstable_outfile = os.path.join(RESULTS_DIR,test_year+'_unstable_results.csv')
    unstable.to_csv(unstable_outfile,index=False,float_format='%.3f')

def save_1965_results():
    test_filename = os.path.join(DATA_DIR,'1965','test.csv')
    filename = os.path.join(PREDS_DIR,'domain_1965md_1965md_full_data_preds.csv')
    df = pd.read_csv(filename)
    test_df = pd.read_csv(test_filename)
    test_df = test_df[['Title','Industry']]
    df = df.merge(test_df,on=['Title','Industry'])
    df = df.loc[(df.pred_DPT.notna()) & (df.GED.notna()),:]
    df['Data'] = df.DPT.str.slice(1,2).astype(int)
    df['People'] = df.DPT.str.slice(2,3).astype(int)
    df['Things'] = df.DPT.str.slice(3,4).astype(int)
    df['pred_Data'] = df['pred_DPT'].str.slice(1,2).astype(int)
    df['pred_People'] = df['pred_DPT'].str.slice(2,3).astype(int)
    df['pred_Things'] = df['pred_DPT'].str.slice(3,4).astype(int)
    results = []
    attributes = ['Data','People','Things','DCP','STS','GED','SVP','EHFCoord','FingerDexterity']
    for attr in attributes:
        labels = df[attr]
        preds = df['pred_'+attr]
        obs = len(preds)
        accuracy = (labels == preds).mean()
        mae = (preds - labels).abs().mean()
        corr = pearsonr(preds,labels)[0]
        rank_corr = spearmanr(preds,labels)[0]
        results.append({'Attribute':attr,'Accuracy':accuracy,
                        'Correlation':corr,'Rank Correlation':rank_corr,'MAE': mae,'Obs':obs})

    results = pd.DataFrame(results)
    results.loc[results.Attribute == 'FingerDexterity','Attribute'] = 'FingerDext'
    outfile = os.path.join(RESULTS_DIR,'1965_results.csv')
    results.to_csv(outfile,index=False,float_format='%.3f')

    stable = results.loc[results.Attribute.isin(attributes[:5])]
    stable_outfile = os.path.join(RESULTS_DIR,'1965_stable_results.csv')
    stable.to_csv(stable_outfile,index=False,float_format='%.3f')

    unstable = results.loc[~results.Attribute.isin(attributes[:5])]
    unstable_outfile = os.path.join(RESULTS_DIR,'1965_unstable_results.csv')
    unstable.to_csv(unstable_outfile,index=False,float_format='%.3f')

def main():
    save_1965_results()
    save_results('1977')
    save_results('1991')

main()
