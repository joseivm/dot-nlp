import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DATA_DIR = os.path.join(PROJECT_DIR,'data')

def get_HML_dict(year):
    filepath = os.path.join(DATA_DIR,'Attr',year,'full_data.csv')
    df = pd.read_csv(filepath)
    quantile_dict = {}
    for column in ['FingerDexterity','EHFCoord']:
        df[column] = df[column].astype(float)
        cdf = df.groupby(column).size().reset_index(name='N')
        weighted_array = [[value]*count for value, count in zip(cdf[column],cdf.N)]
        weighted_array = [item for sublist in weighted_array for item in sublist]
        quantiles = np.quantile(weighted_array,[.2,.8])
        quantile_dict[column] = quantiles
    return(quantile_dict)
