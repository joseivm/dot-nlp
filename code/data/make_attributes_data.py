import os
import re
import pandas as pd
import numpy as np
import jellyfish as jf
import data_utils as du

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DATA_DIR = os.path.join(PROJECT_DIR,'data')

##### Parsing Classes #####
class ParserHelper:
    def __init__(self):
        self.code_regex = re.compile("[\d]{3}[\s\.]{1,4}[\d]{3}")
        self.starts_with_title_regex = re.compile("^[A-Z]+[A-Z \-\,\.']+\([a-zA-Z\s\-\.,;\&]+\)")
        self.full_title_regex = re.compile("[A-Z]+[A-Z \-\,\.']+\([a-zA-Z\s\-\.,;\&]+\)")
        self.lone_title_regex = re.compile("[A-Z]+[A-Z \-\,\.]+")
        self.lone_industry_regex = re.compile("^\([a-zA-Z\s\-\.,;\&]+\)")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;\&]+\)")
        self.see_regex = re.compile("^see")
        self.see_under_regex = re.compile("(^see under|^see [A-Z]+[A-Z \-\,]+ under [A-Z]+[A-Z \-\,]+\([a-zA-Z\s\-\.,;\&]+\)+)")
        self.code_counter = {}
        self.leading_period_regex = re.compile("^\.")

    def has_see(self, line):
        see_match = self.see_regex.search(line.lower())
        return see_match is not None

    def has_see_under(self, line):
        see_under_match = self.see_under_regex.search(line.lower())
        return see_under_match is not None

    def has_code(self,line):
        code_match = self.code_regex.search(line)
        return code_match is not None

    def starts_with_title(self,line):
        title_match = self.starts_with_title_regex.search(line)
        return title_match is not None

    def has_full_title(self, line):
        title_match = self.full_title_regex.search(line)
        return title_match is not None

    def has_lone_title(self, line):
        title_match = self.lone_title_regex.search(line)
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
        if self.has_lone_title(line):
            title_match = self.lone_title_regex.search(line)
            title = title_match.group(0)
            title = title.rstrip()
            return title
        else:
            return None

    def get_full_title(self,line):
        if self.has_full_title(line):
            title = self.get_title(line)
            industry = self.get_industry(line)
            return(title + ' ' + industry)
        else:
            return None

    def get_industry(self, title):
        industry_match = self.industry_regex.search(title)
        industry = industry_match.group(0)
        industry = industry.rstrip()
        return(industry)

    def remove_code(self,line):
        line = re.sub(self.code_regex,'',line)
        line = re.sub(self.leading_period_regex,'',line)
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
        if self.ph.has_see_under(self.current_definition):
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
                if self.ph.starts_with_title(line):
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

                elif self.ph.starts_with_title(line):
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
        # df['FullTitle'] = list(zip(df.CleanedTitle,df.CleanedIndustry))
        return df

class AttributesParser91:
    def __init__(self):
        self.definitions = []

    def parse(self,dot):
        for line in dot:
            entry = {}
            dot_code = line[7:16]
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
            entry['GED'] = max([float(ged_math),1.5])
            entry['EHFCoord'] = float(eye_hand) if float(eye_hand) != 2.0 else 2.5
            entry['FingerDexterity'] = max([float(finger_dext),1.5]) if float(finger_dext) < 5 else 4.5
            entry['DCP'] = 1 if 'D' in temp else 0
            entry['STS'] = 1 if 'T' in temp else 0
            entry['DLU'] = update_date
            self.definitions.append(entry)

        df = pd.DataFrame(self.definitions)
        df.loc[:,'Definition'] = df['Definition'].str.strip()
        df['DPT'] = df['Code'].str.slice(3,6).apply(harmonize_DPT)
        df['Attr'] = list(zip(df.GED.astype(str),df.EHFCoord.astype(str),
                              df.FingerDexterity.astype(str),df.DCP.astype(str),
                              df.STS.astype(str)))
        df = df[['Title','Code','Definition','DPT','Industry','GED','EHFCoord','FingerDexterity','DCP','STS','Attr']]
        return(df)

##### Data Loading Functions #####
def load_metadata():
    md = pd.read_excel(PROJECT_DIR +'/data/raw/1965_metadata.xlsx')
    md = clean_column(md,'Title','CleanedTitle')
    md = clean_column(md,'Industry','CleanedIndustry')
    md['FullTitle'] = md['CleanedTitle'] + ' ' + md['CleanedIndustry']
    md = md.drop_duplicates()
    # md = md.drop_duplicates('FullTitle')
    md = md.drop_duplicates('FullTitle',keep=False)
    # md['FullTitle'] = list(zip(md.CleanedTitle,md.CleanedIndustry))
    return(md)

def load_91_dot():
    filepath = os.path.join(PROJECT_DIR,'data','raw','DOT1991.txt')
    with open(filepath,'r',errors='ignore') as f:
        lines = f.readlines()
    return(lines)

def load_65_dot():
    filepath = os.path.join(DATA_DIR,'raw','1965_Output.txt')
    with open(filepath,'r',errors='ignore') as f:
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
    dot_text = load_65_dot()
    md = load_metadata()
    parser = AttributesParser65()
    df = parser.parse(dot_text)
    md = md.merge(df[['FullTitle','Definition']],on='FullTitle',how='left')
    md = fuzzy_match(md,df)
    md.loc[md.FuzzyScore >= 0.91,'Definition'] = md.loc[md.FuzzyScore >= 0.91,'FuzzyDefinition']
    md = match_see_defs(md)
    md = md.loc[md.Definition.notna(),:]
    md = md.loc[md.Definition != '',:]
    md = add_outcomes(md)
    md['DPT'] = md['Code'].astype(str).str.slice(-3)
    md.loc[md.DPT.str.contains('\.'),'DPT'] = md.loc[md.DPT.str.contains('\.'),'DPT'].str.slice(1) + '0'
    md['DPT'] = md['DPT'].apply(harmonize_DPT)
    md['Attr'] = list(zip(md.GED.astype(str),md.EHFCoord.astype(str),
                          md.FingerDexterity.astype(str),md.DCP.astype(str),
                          md.STS.astype(str)))
    md = md[['Title','Code','Definition','DPT','Industry','GED','EHFCoord','FingerDexterity','DCP','STS','Attr']]
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

def match_see_defs(md):
    ph = ParserHelper()
    md['SeeDefinition'] = md['Definition'].apply(str).apply(ph.has_see)
    md.loc[md.SeeDefinition,'SeeFullTitle'] = md.loc[md.SeeDefinition, 'Definition'].apply(str).apply(ph.get_full_title)
    md.loc[md.SeeDefinition,'SeeLoneTitle'] = md.loc[md.SeeDefinition, 'Definition'].apply(str).apply(ph.get_title)
    md = clean_column(md,'SeeFullTitle','SeeFullTitle')
    md = clean_column(md,'SeeLoneTitle','SeeLoneTitle')
    md_copy = md.copy()
    for idx, row in md_copy.loc[md_copy.SeeDefinition,:].iterrows():
        if row.SeeFullTitle != 'None':
            definition = md.loc[md.FullTitle == row.SeeFullTitle,'Definition']
        else:
             definition = md.loc[md.CleanedTitle == row.SeeLoneTitle,'Definition']

        if not definition.empty:
            md.loc[idx,'Definition'] = definition.to_numpy()[0]
    return md

def compute_midpoint(code):
    code = re.sub(' ','',code)
    codes = [int(subcode) for subcode in code]
    return np.median(codes)

def add_outcomes(md):
    md['GED'] = md['GED'].apply(str).apply(compute_midpoint)
    md['EHFCoord'] = md['E'].apply(str).apply(compute_midpoint)
    md['FingerDexterity'] = md['F'].apply(str).apply(compute_midpoint)
    md['DCP'] = md['Temp'].astype(str).str.contains('4')
    md['STS'] = md['Temp'].astype(str).str.contains('Y',case=False)
    md = md.replace({True: 1, False: 0})
    return(md)

def harmonize_DPT(code):
    data = int(code[0])
    people = int(code[1])
    things = int(code[2])
    data = 6 if data > 6 else data
    people = 8 if people > 8 else people
    things = 7 if things > 7 else things
    dpt = 'D'+str(data)+str(people)+str(things)
    return dpt

def main():
    dot_1965 = make_1965_data()
    dot_1965.to_csv(os.path.join(DATA_DIR,'Attr','1965','full_data.csv'),index=False)
    du.save_data(dot_1965,'Attr','1965')
    du.save_data(dot_1965,'DPT','1965')
    dot_1991 = make_1991_data()
    dot_1991.to_csv(os.path.join(DATA_DIR,'Attr','1991','full_data.csv'),index=False)
    du.save_data(dot_1991,'Attr','1991')

main()
