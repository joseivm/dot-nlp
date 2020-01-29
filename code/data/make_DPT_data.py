import re
import os
import pandas as pd
import data_utils as du

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

PROJECT_DIR = os.environ.get("PROJECT_DIR")
DOT_1977_PATH = os.path.join(PROJECT_DIR,'data','raw','1977_Output.txt')
DOT_1965_PATH = os.path.join(PROJECT_DIR,'data','raw','1965_Output.txt')
DATA_DIR = os.path.join(PROJECT_DIR,'data')

##### Parser Classes #####
class DPTParser77:
    def __init__(self):
        self.title_regex = re.compile('^[\d]{2}[\-\.\s\d]{0,11}[A-Z]+[A-Z\-,\s]*')
        self.code_regex = re.compile("\d{3}.\d{3}[-\s]{1,2}\d{3}")
        self.name_regex = re.compile("[A-Z]+[\-\s,A-Z]*")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;\&]+\)")
        self.definitions = {}
        self.current_title = ''
        self.current_definition = ''

    def is_title(self,line):
        title_match = self.title_regex.search(line)
        return title_match is not None

    def is_occ_title(self,title):
        code_match = self.code_regex.search(title)
        return code_match is not None

    def get_title(self,line):
        title_match = self.title_regex.search(line)
        return title_match.group(0)

    def get_name(self,title):
        name_match = self.name_regex.search(title)
        return name_match.group(0)

    def get_industry(self, title):
        industry_match = self.industry_regex.search(title)
        return industry_match.group(0)

    def get_occ_code(self,title):
        code_match = self.code_regex.search(title)
        return code_match.group(0)

    def parse(self,dot):
        # Go line by line, if line is a title, start saving definition
        # until you come across another title. Once you come across another
        # title, check if current title is an occ title, if it is, add it to
        # dictionary, otherwise, discard
        for line in dot:
            if self.is_title(line):
                if self.is_occ_title(self.current_title):
                    code = self.get_occ_code(self.current_title)
                    name = self.get_name(self.current_title)
                    self.definitions[code] = (name,self.current_definition)
                self.current_title = self.get_title(line)
                self.current_definition = ''
            else:
                self.current_definition += line

        df = pd.DataFrame(list(self.definitions.items()),columns=['Code','TitleDefinition'])
        df['Title'] = df['TitleDefinition'].apply(lambda x: x[0])
        df['Definition'] = df['TitleDefinition'].apply(lambda x: x[1])
        df['DPT'] = df['Code'].apply(get_DPT)
        return df[['Title','Code','Definition','DPT']]

class DPTParser65:
    def __init__(self):
        self.code_regex = re.compile("[\d]{3}[\s\.]{1,4}[\d]{3}")
        self.code_counter = {}
        self.definitions = {}
        self.current_code = ''
        self.current_title = ''
        self.current_definition = ''
        self.collecting = False

    def has_code(self,line):
        code_match = self.code_regex.search(line)
        return code_match is not None

    def make_code_unique(self,code):
        code_count = self.code_counter.get(code,0)+1
        self.code_counter[code] = code_count
        return code[:3] + '.' + code[-3:] + '.' + str(code_count)

    def get_code(self,title):
        code_match = self.code_regex.search(title)
        code = re.sub('[^\d]','',code_match.group(0))
        code = self.make_code_unique(code)
        return code

    def parse(self,dot):
        # Go line by line, if line is a code, start saving definition
        # until you come across another code or newline. Once you come across a
        # stopping point, save your definition and look for another code.
        for line in dot:
            if not self.collecting:
                if self.has_code(line):
                    self.collecting = True
                    self.current_code = self.get_code(line)
                    self.current_definition = ''
                else:
                    pass
            else:
                if not line:
                    self.collecting = False
                    self.definitions[self.current_code] = ('no title',self.current_definition)
                elif self.has_code(line):
                    self.definitions[self.current_code] = ('no title',self.current_definition)
                    self.current_code = self.get_code(line)
                    self.current_definition = ''
                else:
                    self.current_definition += line

        df = pd.DataFrame(list(self.definitions.items()),columns=['Code','TitleDefinition'])
        df['Title'] = df['TitleDefinition'].apply(lambda x: x[0])
        df['Definition'] = df['TitleDefinition'].apply(lambda x: x[1])
        df['DPT'] = df['Code'].apply(get_DPT)
        return df[['Title','Code','Definition','DPT']]

##### Data Loading Functions #####
def load_77_dot():
    with open(DOT_1977_PATH,'rb') as f:
        dot = f.readlines()
    dot = [line.decode('utf-8',errors='ignore') for line in dot]
    dot = [line.rstrip() for line in dot]
    dot = [line for line in dot if line != '']
    return dot

def load_65_dot():
    with open(DOT_1965_PATH,'r',errors='ignore') as f:
        lines = f.readlines()
    dot = [line.rstrip() for line in lines]
    return dot

##### Data Creation Functions #####
def make_1977_data():
    dot = load_77_dot()
    parser = DPTParser77()
    df = parser.parse(dot)
    return(df)

def make_1965_data():
    dot = load_65_dot()
    parser = DPTParser65()
    df = parser.parse(dot)
    return(df)

def get_DPT(code):
    data = int(code[4])
    people = int(code[5])
    things = int(code[6])
    data = 6 if data > 6 else data
    things = 7 if things > 7 else things
    dpt = str(data)+str(people)+str(things)
    return dpt

def main():
    df_1977 = make_1977_data()
    du.save_data(df_1977,'DPT','1977')

main()
