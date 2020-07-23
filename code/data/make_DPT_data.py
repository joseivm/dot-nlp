import re
import os
import pandas as pd
import numpy as np
import data_utils as du

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DOT_1977_PATH = os.path.join(PROJECT_DIR,'data','raw','1977_Output.txt')
DATA_DIR = os.path.join(PROJECT_DIR,'data')

##### Parser Classes #####
class DOT77Parser:
    def __init__(self):
        self.full_title_regex = re.compile('^[\d]{2}[\-\.\s\d]{0,11}[A-Z]+[A-Z\-,\s]*')
        self.code_regex = re.compile("\d{3}.\d{3}[-\s]{1,2}\d{3}")
        self.title_regex = re.compile("[A-Z]+[\-\s,A-Z]*")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;\&]+\)")
        self.numeral_regex = re.compile("\)\s{0,2}[IV]{1,4}")
        self.alternate_titles_regex = re.compile("([a-z\-\s,\.]+;*)+$")
        self.reference_regex = re.compile("(Performs|performing) duties as described under ([A-Z]+[\-\s,A-Z]*)(\([a-zA-Z\s\-\.,;\&]+\))")
        self.definitions = {}
        self.current_title = ''
        self.current_definition = ''

    def has_full_title(self,line):
        title_match = self.full_title_regex.search(line)
        return title_match is not None

    def has_reference(self, line):
        reference_match = self.reference_regex.search(line)
        return reference_match is not None

    def has_occ_code(self,title):
        code_match = self.code_regex.search(title)
        return code_match is not None

    def has_title(self,line):
        title_match = self.title_regex.search(line)
        return title_match is not None

    def has_numeral(self,line):
        numeral_match = self.numeral_regex.search(line)
        return numeral_match is not None

    def get_numeral(self,title):
        if self.has_numeral(title):
            numeral_match = self.numeral_regex.search(title)
            numeral = numeral_match.group(0)
            numeral = numeral.strip()
            numeral = re.sub("l",'I',numeral)
            numeral = re.sub("[\)\s]",'',numeral)
        else:
            numeral = ''
        return numeral

    def get_title(self,line):
        if self.has_title(line):
            title_match = self.title_regex.search(line)
            title = title_match.group(0)
            title = title.strip()
            title = re.sub('\s*-\s*',' ',title)
            title = remove_extra_spaces(title)
            return title
        else:
            return None

    def get_alternate_titles(self,line):
        alt = re.sub(self.full_title_regex,'',line)
        alt = re.sub(self.industry_regex,'',alt)
        alt = alt.strip()
        alt = re.sub('\.','',alt)
        # if alt_titles_match is not None:
        #     alt_titles = alt_titles_match.group(0)
        # else:
        #     alt_titles = ''
        return alt

    def get_reference_title(self,line):
        ref_match = self.reference_regex.search(line)
        title = ref_match.group(2)
        title = title.strip()
        title = re.sub('\s*-\s*',' ',title)
        title = remove_extra_spaces(title)
        return title

    def get_reference_industry(self,line):
        ref_match = self.reference_regex.search(line)
        industry = ref_match.group(3)
        industry = clean_industry(industry)
        return industry

    def get_industry(self, title):
        industry_match = self.industry_regex.search(title)
        if industry_match is not None:
            industry = industry_match.group(0)
            industry = clean_industry(industry)
        else:
            industry = ''
        return(industry)

    def get_occ_code(self,title):
        code_match = self.code_regex.search(title)
        code = code_match.group(0)
        code = re.sub('\s*-\s*','-',code)
        code = re.sub('\s*\.\s*','.',code)
        return code

    def parse(self,dot):
        # Go line by line, if line is a title, start saving definition
        # until you come across another title. Once you come across another
        # title, check if current title is an occ title, if it is, add it to
        # dictionary, otherwise, discard
        for line in dot:
            if self.has_full_title(line):
                if self.has_occ_code(self.current_title):
                    code = self.get_occ_code(self.current_title)
                    title = self.get_title(self.current_title)
                    industry = self.get_industry(self.current_title)
                    numeral = self.get_numeral(self.current_title)
                    alt_titles = self.get_alternate_titles(self.current_title)
                    self.definitions[code] = (title,industry,numeral,self.current_definition,alt_titles)
                self.current_title = line
                self.current_definition = ''
            else:
                self.current_definition += line

        df = pd.DataFrame(list(self.definitions.items()),columns=['Code','TitleInfo'])
        df['Title'] = df['TitleInfo'].apply(lambda x: x[0])
        df['Industry'] = df['TitleInfo'].apply(lambda x: x[1])
        df['Numeral'] = df['TitleInfo'].apply(lambda x: x[2])
        df['Definition'] = df['TitleInfo'].apply(lambda x: x[3])
        df['AlternateTitles'] = df['TitleInfo'].apply(lambda x: x[4])
        df['DPT'] = df['Code'].apply(get_DPT)
        df.drop(columns=['TitleInfo'],inplace=True)
        # df = df.loc[df.Definition != '',:]
        return df

##### Cleaning Functions #####
def remove_extra_spaces(title):
    title = str(title)
    return(' '.join(title.split()))

def clean_industry(industry):
    industry = industry.lower()
    industry = re.sub('[\.,]','',industry)
    industry = industry.strip()
    industry = remove_extra_spaces(industry)
    return(industry)

##### Data Loading Functions #####
def load_77_dot():
    with open(DOT_1977_PATH,'rb') as f:
        dot = f.readlines()
    dot = [line.decode('utf-8',errors='ignore') for line in dot]
    dot = [re.sub('\-\n','',line) for line in dot]
    dot = [line.rstrip() for line in dot]
    dot = [line for line in dot if line != '']
    return dot

##### Data Creation Functions #####
def make_1977_data():
    dot = load_77_dot()
    parser = DOT77Parser()
    df = parser.parse(dot)
    df = match_reference_defs(df)
    return(df)

def match_reference_defs(df):
    ph = DOT77Parser()
    df['HasReference'] = df['Definition'].apply(ph.has_reference)
    df.loc[df.HasReference,'ReferenceTitle'] = df.loc[df.HasReference,'Definition'].apply(ph.get_reference_title)
    df.loc[df.HasReference,'ReferenceIndustry'] = df.loc[df.HasReference,'Definition'].apply(ph.get_reference_industry)
    df['ReferenceDefinition'] = ''
    verbose=False
    for idx, row in df.iterrows():
        if idx % 1000 == 0: print(idx)
        if row['HasReference']:
            # if idx > 21000: verbose=True
            definition = find_def(df,row['ReferenceTitle'],row['ReferenceIndustry'],verbose)
            # df.at[idx,'Code'] = code
            df.at[idx,'ReferenceDefinition'] = definition
    df.drop(columns=['HasReference'],inplace=True)
    return df

def find_def(df,title,industry,verbose=False):
    if verbose: print(title+industry)
    if sum((df.Title == title) & (df.Industry == industry)) == 0:
        return ''
    elif df.loc[(df.Title == title) & (df.Industry == industry),'Definition'].notna().to_numpy()[0]:
        definition = df.loc[(df.Title == title) & (df.Industry == industry),'Definition'].to_numpy()[0]
        return definition
    elif df.loc[(df.Title == title) & (df.Industry == industry),'ReferenceTitle'].notna().to_numpy()[0]:
        reference_title = df.loc[(df.Title == title) & (df.Industry == industry),'ReferenceTitle'].to_numpy()[0]
        industry = df.loc[(df.Title == title) & (df.Industry == industry),'ReferenceIndustry'].to_numpy()[0]
        return find_def(df,reference_title,industry,verbose)
    else:
        return ''

def get_DPT(code):
    data = int(code[4])
    people = int(code[5])
    things = int(code[6])
    data = 6 if data > 6 else data
    things = 7 if things > 7 else things
    dpt = 'D'+str(data)+str(people)+str(things)
    return dpt

def main():
    df = make_1977_data()
    df.to_csv(os.path.join(DATA_DIR,'DPT','1977','full_data.csv'),index=False)
    du.save_data(df,'DPT','1977')

main()
