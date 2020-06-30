import os
import re
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DATA_DIR = os.path.join(PROJECT_DIR,'data')

##### Parsing Classes #####
class ParserHelper:
    def __init__(self):
        self.code_regex = re.compile("[\d]{1}[\s\-]{1,4}[\d]{2}[\.\d]*")
        self.starts_with_full_title_regex = re.compile("^[A-Z]{3,}[A-Z \-\,\.']+[\(a-z\s\-\.,;\& \)]*\([a-zA-Z\s\-\.,;\&]+\)")
        self.starts_with_title_regex = re.compile('''^[A-Z"]{3,}[A-Z \-\,\.']+''')
        self.starts_with_industry_regex = re.compile("^\([a-zA-Z\s\-\.,;\&]+\)")
        self.full_title_regex = re.compile("[A-Z]{3,}[A-Z \-\,\.']+[\(a-z\s\-\.,;\& \)]*\([a-zA-Z\s\-\.,;\&]+\)")
        self.title_regex = re.compile('''[A-Z"]{3,}[A-Z \-\,\.']+''')
        self.numeral_regex = re.compile("\)\s{0,2}[IV]{1,4}")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;:\&]+\)")
        self.reference_regex = re.compile('''(^[Ss]ee |^[Aa] [A-Z"]{3,}[A-Z \-\,\.']+ |^[Aa]n |^[Ss]pec[\.,] for )''')
        self.see_under_regex = re.compile("(^see under|^see [A-Z]+[A-Z \-\,]+ under [A-Z]+[A-Z \-\,]+\([a-zA-Z\s\-\.,;\&]+\)+)")
        self.leading_period_regex = re.compile("^\.")
        self.end_regex = re.compile("([\.,]$|\. [A-Z]$)")

    def has_reference(self, line):
        reference_match = self.reference_regex.search(line)
        return reference_match is not None

    def has_see_under(self,line):
        reference_match = self.see_under_regex.search(line)
        return reference_match is not None

    def has_code(self,line):
        code_match = self.code_regex.search(line)
        return code_match is not None

    def starts_with_full_title(self,line):
        title_match = self.starts_with_title_regex.search(line)
        title_match = title_match is not None
        industry_match = self.has_industry(line)
        # title_match = self.starts_with_full_title_regex.search(line)
        return title_match and industry_match

    def starts_with_title_no_industry(self,line):
        title_match = self.starts_with_title_regex.search(line)
        title_match = title_match is not None
        industry_match = self.has_industry(line)
        return title_match and not industry_match

    def has_industry_and_numeral(self,line):
        numeral_match = self.starts_with_industry_regex.search(line)
        numeral_match = numeral_match is not None
        industry_match = self.has_numeral(line)
        # match = self.starts_with_numeral_and_industry_regex.search(line)
        return numeral_match and industry_match

    def starts_with_industry(self,line):
        industry_match = self.starts_with_industry_regex.search(line)
        return industry_match is not None

    def has_industry(self,line):
        industry_match = self.industry_regex.search(line)
        return industry_match is not None

    def has_numeral(self,line):
        numeral_match = self.numeral_regex.search(line)
        return numeral_match is not None

    def has_title(self,line):
        title_match = self.title_regex.search(line)
        return title_match is not None

    def get_industry(self, title):
        industry_match = self.industry_regex.search(title)
        industry = industry_match.group(0)
        industry = industry.strip().lower()
        return(industry)

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

    def get_code(self,line):
        if self.has_code(line):
            code_match = self.code_regex.search(line)
            code = code_match.group(0)
        else:
            code = ''
        return code

    def get_title(self,line):
        if self.has_title(line):
            title_match = self.title_regex.search(line)
            title = title_match.group(0)
            title = title.strip()
            return title
        else:
            return None

    def get_full_title(self,line):
        if self.has_title(line) and self.has_industry(line):
            title = self.get_title(line)
            industry = self.get_industry(line)
            return(title + ' ' + industry)
        else:
            return self.get_title(line)

    def remove_code(self,line):
        line = re.sub(self.code_regex,'',line)
        line = re.sub(self.leading_period_regex,'',line)
        return line

    def remove_trailing_period(self,line):
        line = re.sub(self.end_regex,'',line)
        return(line)

class DOT49Parser:
    # add set_title function
    def __init__(self):
        self.ph = ParserHelper()
        self.title_searching = True
        self.current_title = ''
        self.current_full_title = ''
        self.current_definition = ''
        self.current_industry = ''
        self.current_numeral = ''
        self.definitions = {}
        self.previous_full_title = (None,None,None,None,None)
        self.previous_definition = ''
        self.previous_code = ''

    def update_title(self,line,title=True,numeral=True,industry=True,full_title=True):
        if title:
            self.current_title = self.ph.get_title(line)
        if numeral:
            self.current_numeral = self.ph.get_numeral(line)
        if industry:
            self.current_industry = self.ph.get_industry(line)
        if full_title:
            if not title:
                self.current_full_title = self.current_title + ' '+line
            else:
                self.current_full_title = line

    def set_title(self,title='',numeral='',industry='',code='',full_title=''):
        if not title:
            self.current_title = title
        if not numeral:
            self.current_numeral = numeral
        if not industry:
            self.current_industry = industry
        if not full_title:
            self.full_title = full_title

    def add_current_definition(self):
        if self.ph.has_see_under(self.current_definition):
            self.current_definition = ''
            return

        code = self.ph.get_code(self.current_definition)
        definition = self.ph.remove_code(self.current_definition)
        full_title = (self.current_title, self.current_numeral,
                      self.current_industry, code, self.current_full_title)
        self.definitions[full_title] = definition
        self.current_definition = ''
        self.previous_full_title = full_title
        self.previous_definition = definition
        self.previous_code = code
        return

    def parse(self,dot):
        for line in dot:
            if self.title_searching:
                if self.ph.starts_with_full_title(line):
                    if 'Cont.' in line:
                        if len(re.findall(self.ph.full_title_regex,line)) == 2:
                            # treat second title as new title
                            match = re.search('Cont\.(.+)',line)
                            line = match.group(1)
                            self.update_title(line)
                            self.title_searching = False

                        else:
                            self.set_title(*self.previous_full_title)
                            self.current_definition = self.previous_code +' '+self.previous_definition
                            self.title_searching = False

                    else:
                        self.update_title(line)
                        self.title_searching = False
                else:
                    pass
            else:
                if not line:
                    self.add_current_definition()
                    self.title_searching = True

                elif self.ph.starts_with_industry(line):
                    self.add_current_definition()
                    self.update_title(line,title=False)

                elif self.ph.starts_with_full_title(line):
                    if 'Cont.' in line:
                        if len(re.findall(self.ph.full_title_regex,line)) == 2:
                            # treat second title as new title
                            match = re.search('Cont\.(.+)',line)
                            line = match.group(1)
                            self.add_current_definition()
                            self.update_title(line)

                        else:
                            # ignore
                            pass
                    else:
                        self.add_current_definition()
                        self.update_title(line)

                elif self.ph.starts_with_title_no_industry(line):
                    self.add_current_definition()
                    self.update_title(line,industry=False)

                else:
                    self.current_definition += line

        df = pd.DataFrame(list(self.definitions.items()),columns=['OriginalTitle','Definition'])
        df['Title'] = df['OriginalTitle'].apply(lambda x: x[0])
        df['Numeral'] = df['OriginalTitle'].apply(lambda x: x[1])
        df['Industry'] = df['OriginalTitle'].apply(lambda x: x[2])
        df['Code'] = df['OriginalTitle'].apply(lambda x: x[3])
        df['FullTitle'] = df['OriginalTitle'].apply(lambda x: x[4])

        df.drop(columns=['OriginalTitle'],inplace=True)
        # df = clean_column(df,'Title','CleanedTitle')
        # df = clean_column(df,'Industry','CleanedIndustry')
        return df

##### Data Loading Functions #####
def load_49_dot():
    filepath = os.path.join(DATA_DIR,'raw','1949_Output.txt')
    with open(filepath,'r',errors='ignore') as f:
        lines = f.readlines()
    dot = [line.strip() for line in lines]
    dot = [remove_extra_spaces(line) for line in dot]
    dot = [re.sub('^SEE ','see ',line) for line in dot]
    dot = [re.sub('^see Volume','check Volume',line) for line in dot]
    return(dot)

def clean_column(df,col_to_clean):
    df.loc[:,col_to_clean] = df[col_to_clean].str.replace('\s*-\s*',' ')
    df.loc[:,col_to_clean] = df[col_to_clean].str.strip()
    df.loc[:,col_to_clean] = df[col_to_clean].str.replace('\.',',')
    df.loc[:,col_to_clean] = df[col_to_clean].apply(remove_extra_spaces)
    return(df)

def remove_extra_spaces(title):
    title = str(title)
    return(' '.join(title.split()))

def clean_industry(industry):
    industry = industry.lower()
    industry = re.sub('[\.,]','',industry)
    industry = industry.strip()
    industry = remove_extra_spaces(industry)
    return(industry)

##### Data Creation functions #####
def make_1949_data():
    dot_text = load_49_dot()
    parser = DOT49Parser()
    df = parser.parse(dot_text)
    df = match_reference_defs(df)
    df['DOT Code'] = df.Code.str.slice(stop=4)
    return(df)

def match_reference_defs(df):
    ph = ParserHelper()
    df['HasReference'] = df['Definition'].apply(ph.has_reference)
    df.loc[df.HasReference,'ReferenceTitle'] = df.loc[df.HasReference,'Definition'].apply(ph.get_full_title)
    df.loc[df.ReferenceTitle.notna(),
            'ReferenceTitle'] = df.loc[df.ReferenceTitle.notna(),'ReferenceTitle'].apply(ph.remove_trailing_period)
    df = clean_column(df,'Title')
    df = clean_column(df,'ReferenceTitle')
    df['NumberedTitle'] = df['Title']+ ' '+df['Numeral']
    df = clean_column(df,'NumberedTitle')
    df = clean_column(df,'Industry')
    df['Industry'] = df['Industry'].str.lower()
    df['Industry'] = df['Industry'].str.replace('[\.,]','')
    df['Code'] = df.Code.replace({'': np.nan})
    df['ReferenceDefinition'] = ''
    verbose=False
    for idx, row in df.iterrows():
        if idx % 1000 == 0: print(idx)
        if str(row['Code']) == 'nan':
            # if idx > 21000: verbose=True
            code, definition = find_code_and_def(df,row['NumberedTitle'],row['Industry'],ph,verbose)
            df.at[idx,'Code'] = code
            df.at[idx,'ReferenceDefinition'] = definition
    df.drop(columns=['HasReference'],inplace=True)
    return df

def find_code_and_def(df,title,industry,ph,verbose=False):
    if verbose: print(title+industry)
    if sum((df.NumberedTitle == title) & (df.Industry == industry)) == 0:
        return np.nan, ''
    elif df.loc[(df.NumberedTitle == title) & (df.Industry == industry),'Code'].notna().to_numpy()[0]:
        code = df.loc[(df.NumberedTitle == title) & (df.Industry == industry),'Code'].to_numpy()[0]
        definition = df.loc[(df.NumberedTitle == title) & (df.Industry == industry),'Definition'].to_numpy()[0]
        return code, definition
    elif df.loc[(df.NumberedTitle == title) & (df.Industry == industry),'ReferenceTitle'].notna().to_numpy()[0]:
        reference_title = df.loc[(df.NumberedTitle == title) & (df.Industry == industry),'ReferenceTitle'].to_numpy()[0]
        if ph.has_title(reference_title) and ph.has_industry(reference_title):
            industry = ph.get_industry(reference_title)
            industry = clean_industry(industry)
            reference_title = ph.get_title(reference_title)
        return find_code_and_def(df,reference_title,industry,ph,verbose)
    else:
        return np.nan, ''

##### Inspection Function #####
def next_n(df,n,num):
    start = n*(num-1)
    end = n*num
    return df.loc[start:end,:]

def main():
    dot_1949 = make_1949_data()
    dot_1949.to_csv(os.path.join(DATA_DIR,'raw','1949_DOT_structured.csv'),index=False)

main()
