import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DATA_DIR = os.path.join(PROJECT_DIR,'data')

RANDOM_SEED = 29

def save_data(df,task,year):
    train, val, test = train_val_test_split(df)
    write_set(train,task,year,'train')
    write_set(val,task,year,'dev')
    write_set(test,task,year,'test')

def train_val_test_split(df,train_size=0.6,val_size=0.2,test_size=0.2):
    train_idx = int(train_size*len(df))
    val_idx = int((train_size+val_size)*len(df))
    train, val, test = np.split(df.sample(frac=1,random_state=RANDOM_SEED), [train_idx, val_idx])
    return(train, val, test)

def write_set(df,task,year,set_type):
    outfile = os.path.join(DATA_DIR,task,year,set_type)+'.csv'
    df.to_csv(outfile,index=False)
