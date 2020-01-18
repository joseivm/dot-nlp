import pandas as pd
import os
import re
import numpy as np
import jellyfish as jf
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
RANDOM_STATE = 1

##### Parsing Classes #####
class ParserHelper:
    def __init__(self):
        self.code_regex = re.compile("[\d]{3}[\s\.]{1,4}[\d]{3}")
        self.full_title_regex = re.compile("^[A-Z]+[A-Z \-\,]+\([a-zA-Z\s\-\.,;\&]+\)")
        self.lone_industry_regex = re.compile("^\([a-zA-Z\s\-\.,;\&]+\)")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;\&]+\)")
        self.see_regex = re.compile("^see")
        self.see_under_regex = re.compile("(^see under|^see [A-Z]+[A-Z \-\,]+ under [A-Z]+[A-Z \-\,]+\([a-zA-Z\s\-\.,;\&]+\)+)")
        self.code_counter = {}

    def has_see(self, line):
        see_match = self.see_regex.search(line.lower())
        return see_match is not None

    def has_see_under(self, line):
        see_under_match = self.see_under_regex.search(line.lower())
        return see_under_match is not None

    def has_code(self,line):
        code_match = self.code_regex.search(line)
        return code_match is not None

    def has_full_title(self,line):
        title_match = self.full_title_regex.search(line)
        return title_match is not None

    def has_industry(self,line):
        industry_match = self.lone_industry_regex.search(line)
        return industry_match is not None

    def make_code_unique(self,code):
        code_count = self.code_counter.get(code,0)+1
        self.code_counter[code] = code_count
        return code[:3] + '.' + code[-3:] + '.' + str(code_count)

    def get_code(self,title):
        code_match = self.code_regex.search(title)
        code = re.sub('[^\d]','',code_match.group(0))
        code = self.make_code_unique(code)
        return code

    def get_title(self,line):
        title_match = self.full_title_regex.search(line)
        title = title_match.group(0)
        title = re.sub(self.industry_regex,'',title)
        title = title.rstrip()
        return title

    def get_industry(self, title):
        industry_match = self.industry_regex.search(title)
        industry = industry_match.group(0)
        industry = industry.rstrip()
        return(industry)

    def remove_code(self,line):
        line = re.sub(self.code_regex,'',line)
        return line

class AttributesParser65:
    def __init__(self):
        self.ph = ParserHelper()
        self.title_searching = True
        self.current_title = ''
        self.current_definition = ''
        self.current_industry = ''
        self.previous_title = ''
        self.previous_industry = ''
        self.previous_definition = ''
        self.definitions = {}

    def add_current_definition(self):
        if self.ph.has_see(self.current_definition):
            self.title_searching = True
            self.current_definition = ''
        else:
            definition = self.ph.remove_code(self.current_definition)
            full_title = (self.current_title, self.current_industry)
            self.definitions[full_title] = definition
            self.previous_title = self.current_title
            self.previous_industry = self.current_industry
            self.previous_definition = definition
            self.current_definition = ''
        return

    def parse(self,dot):
        for line in dot:
            if self.title_searching:
                if self.ph.has_full_title(line):
                    if 'continued' in line.lower():
                        self.current_title = self.previous_title
                        self.current_industry = self.previous_industry
                        self.current_definition = self.previous_definition
                    else:
                        self.current_title = self.ph.get_title(line)
                        self.current_industry = self.ph.get_industry(line)
                    self.title_searching = False
                else:
                    pass
            else:
                if not line:
                    self.add_current_definition()
                    self.title_searching = True

                elif self.ph.has_industry(line):
                    self.add_current_definition()
                    self.current_industry = self.ph.get_industry(line)

                elif self.ph.has_full_title(line):
                    self.add_current_definition()
                    self.current_title = self.ph.get_title(line)
                    self.current_industry = self.ph.get_industry(line)

                else:
                    self.current_definition += line

        df = pd.DataFrame(list(self.definitions.items()),columns=['OriginalTitle','Definition'])
        df['Title'] = df['OriginalTitle'].apply(lambda x: x[0])
        df['Industry'] = df['OriginalTitle'].apply(lambda x: x[1])

        df = clean_column(df,'Title','CleanedTitle')
        df = clean_column(df,'Industry','CleanedIndustry')
        df['FullTitle'] = df['CleanedTitle'] + ' ' + df['CleanedIndustry']
        df = df.drop_duplicates('FullTitle',keep=False)
        return df

class AttributesParser91:
    def __init__(self):
        self.definitions = []

    def parse(self,dot):
        for line in dot:
            entry = {}
            dot_code = line[8:16]
            definition_type = line[16]
            update_date = line[17:19]
            dpt_code = line[19:25]
            ged_read = line[25]
            ged_math = line[26]
            lang = line[27]
            svp = line[28]
            gen_learn = line[29]
            verbal = line[30]
            numerical = line[31]
            spacial = line[32]
            form_per = line[33]
            clerical_per = line[34]
            motor_coord = line[35]
            finger_dext = line[36]
            manual_dext = line[37]
            eye_hand = line[38]
            color_disc = line[39]
            temp = line[58:67]
            soc_code = line[73:77]
            title = line[111:179]
            industry = line[179:243]
            definition = line[243:-2]
            entry['Title'] = title
            entry['Industry'] = industry
            entry['Code'] = dot_code
            entry['Definition'] = definition
            entry['GED'] = ged_math
            entry['EHFCoord'] = eye_hand
            entry['FingerDexterity'] = finger_dext
            entry['DCP'] = 1 if 'D' in temp else 0
            entry['STS'] = 1 if 'T' in temp else 0
            entry['DLU'] = update_date
            self.definitions.append(entry)

        df = pd.DataFrame(self.definitions)
        df.loc[:,'Definition'] = df['Definition'].str.strip()
        return(pd.DataFrame(self.definitions))

##### Data Loading Functions #####
def load_metadata():
    md = pd.read_excel(PROJECT_DIR +'/data/raw/1965_metadata.xlsx')
    md = clean_column(md,'Title','CleanedTitle')
    md = clean_column(md,'Industry','CleanedIndustry')
    md['FullTitle'] = md['CleanedTitle'] + ' ' + md['CleanedIndustry']
    md = md.drop_duplicates()
    md = md.drop_duplicates('FullTitle',keep=False)
    return(md)

def load_91_dot():
    filepath = os.path.join(PROJECT_DIR,'data','raw','DOT1991.txt')
    with open(filepath,'r',errors='ignore') as f:
        lines = f.readlines()
    return(lines)

def load_65_dot():
    with open('/Users/joseivelarde/dot-nlp/data/raw/1965_Output.txt','r',errors='ignore') as f:
        lines = f.readlines()
    dot = [line.rstrip() for line in lines]
    dot = [remove_extra_spaces(line) for line in dot]
    return(dot)

def clean_column(df,col_to_clean,new_name):
    df.loc[:,new_name] = df[col_to_clean].str.lower()
    df.loc[:,new_name] = df[new_name].str.replace('[^\w\s\d-]','')
    df.loc[:,new_name] = df[new_name].str.replace('\s+-\s+','-')
    df.loc[:,new_name] = df[new_name].str.strip()
    df.loc[:,new_name] = df[new_name].apply(remove_extra_spaces)
    return(df)

def remove_extra_spaces(title):
    title = str(title)
    return(' '.join(title.split()))

##### Data Creation functions #####
def make_1965_data():
    dot_text = load_dot()
    md = load_metadata()
    parser = AttributesParser65()
    df = parser.parse(dot_text)
    md = md.merge(df[['FullTitle','Definition']],on='FullTitle',how='left')
    md = fuzzy_match(md,df)
    md.loc[md.FuzzyScore >= 0.95,'Definition'] = md.loc[md.FuzzyScore >= 0.95,'FuzzyDefinition']
    md = md.loc[md.Definition.notna()]
    md = add_outcomes(md)
    md = md[['Code','Title','Industry','Definition','GED','EHFCoord','FingerDexterity','DCP','STS']]
    return(md)

def make_1991_data():
    dot = load_91_dot()
    parser = AttributesParser91()
    df = parser.parse(dot)
    return(df)

def fuzzy_match(md,df):
    unmatched_md = md.loc[~md['FullTitle'].isin(df['FullTitle']),:]
    unmatched_df = df.loc[~df['FullTitle'].isin(md['FullTitle']),:]
    for title in unmatched_md['FullTitle']:
        match, similarity = find_fuzzy_match(title,unmatched_df['FullTitle'])
        match_definition = df.loc[df.FullTitle == match,'Definition'].to_numpy()[0]
        md.loc[md.FullTitle == title, 'FuzzyTitleMatch'] = match
        md.loc[md.FullTitle == title, 'FuzzyScore'] = similarity
        md.loc[md.FullTitle == title, 'FuzzyDefinition'] = match_definition
    return(md)

def find_fuzzy_match(title,title_list):
    all_similarities = title_list.apply(lambda x: jf.jaro_winkler(title,x))
    max_similarity = all_similarities.max()
    max_index = all_similarities.idxmax()
    match_title = title_list.loc[max_index]
    return(match_title, max_similarity)

def compute_midpoint(code):
    code = re.sub(' ','',code)
    codes = [int(subcode) for subcode in code]
    return np.mean(codes)

def add_outcomes(md):
    md['GED'] = md['GED'].apply(str).apply(compute_midpoint)
    md['EHFCoord'] = md['E'].apply(str).apply(compute_midpoint)
    md['FingerDexterity'] = md['F'].apply(str).apply(compute_midpoint)
    md['DCP'] = md['Temp'].str.contains('4')
    md['STS'] = md['Temp'].str.contains('Y')
    return(md)

##### Data Saving Functions #####
def save_data(df,year):
    train, val, test = train_val_test_split(df)
    write_set(train,year,'train')
    write_set(val,year,'dev')
    write_set(test,year,'test')

def train_val_test_split(df,train_size=0.6,val_size=0.2,test_size=0.2):
    train_idx = int(train_size*len(df))
    val_idx = int((train_size+val_size)*len(df))
    train, val, test = np.split(df.sample(frac=1), [train_idx, val_idx],random_state=RANDOM_STATE)
    return(train, val, test)

def write_set(df,year,set_type):
    outfile = os.path.join(DATA_DIR,'attributes',year,set_type)+'.csv'
    df.to_csv(outfile,index=False)

def main():
    dot_1965 = make_1965_data()
    save_data(dot_1965,'1965')
    dot_1991 = make_1991_data()
    save_data(dot_1991,'1991')
