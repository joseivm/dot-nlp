import re
import csv
import random
import math
import os

DOT_1977_PATH = '/Users/joseivelarde/dot-nlp/data/raw/1977_Output.txt'
DOT_1965_PATH = '/Users/joseivelarde/dot-nlp/data/raw/1965_Output.txt'
OUTFILE = '/Users/joseivelarde/dot-nlp/data/clean/structured_1977.csv'
DATA_DIR = '/Users/joseivelarde/dot-nlp/data/'

class DPTParser77:
    def __init__(self):
        self.title_regex = re.compile('^[\d]{2}[\-\.\s\d]{0,11}[A-Z]+[A-Z\-,\s]*')
        self.code_regex = re.compile("\d{3}.\d{3}[-\s]{1,2}\d{3}")
        self.name_regex = re.compile("[A-Z]+[\-\s,A-Z]*")
        self.industry_regex = re.compile("\([a-zA-Z\s\-\.,;\&]+\)")

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

class DPTParser65:
    def __init__(self):
        self.code_regex = re.compile("[\d]{3}[\s\.]{1,4}[\d]{3}")
        self.code_counter = {}

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

def load_77_dot():
    with open(DOT_1977_PATH,'rb') as f:
        dot = f.readlines()
    dot = [line.decode('utf-8',errors='ignore') for line in dot]
    dot = [line.rstrip() for line in dot]
    dot = [line for line in dot if line != '']
    return dot

def make_77_occ_dictionary():
    dot = load_77_dot()
    tp = TitleParser77()
    definitions = {}
    current_title = ''
    current_definition = ''
    # Go line by line, if line is a title, start saving definition
    # until you come across another title. Once you come across another
    # title, check if current title is an occ title, if it is, add it to
    # dictionary, otherwise, discard
    for line in dot:
        if tp.is_title(line):
            if tp.is_occ_title(current_title):
                code = tp.get_occ_code(current_title)
                name = tp.get_name(current_title)
                definitions[code] = (name,current_definition)
            current_title = tp.get_title(line)
            current_definition = ''
        else:
            current_definition += line
    return definitions

def write_set(definitions,edition,set_type):
    outfile = os.path.join(DATA_DIR,'dpt',edition,set_type)+'.csv'
    with open(outfile,'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in definitions.items():
            writer.writerow([key,value[0],value[1]])

def save_data(definitions,year):
    write_set(definitions['train'],year,'train')
    write_set(definitions['dev'],year,'dev')
    write_set(definitions['test'],year,'test')

def train_dev_test_split(definitions):
    N = len(definitions)
    train_size = int(math.ceil(N*0.6))
    validation_size = int(math.ceil(N*0.2))
    test_size = int(math.ceil(N*0.2))

    assignments = [0]*train_size + [1]*validation_size + [2]*test_size
    random.shuffle(assignments)
    assignments = assignments[:len(definitions)]

    train_definitions = {}
    dev_definitions = {}
    test_definitions = {}

    for assignment, key in zip(assignments,definitions.keys()):
        if assignment == 0:
            train_definitions[key] = definitions[key]
        elif assignment == 1:
            dev_definitions[key] = definitions[key]
        else:
            test_definitions[key] = definitions[key]

    return({'train':train_definitions,'dev':dev_definitions,'test':test_definitions})

def load_65_dot():
    with open(DOT_1965_PATH,'r',errors='ignore') as f:
        lines = f.readlines()
    dot = [line.rstrip() for line in lines]
    return dot

def make_65_occ_dictionary():
    dot = load_65_dot()
    tp = TitleParser65()
    definitions = {}
    collecting = False
    # Go line by line, if line is a code, start saving definition
    # until you come across another code or newline. Once you come across a
    # stopping point, save your definition and look for another code.
    for line in dot:
        if not collecting:
            if tp.has_code(line):
                collecting = True
                current_code = tp.get_code(line)
                current_definition = ''
            else:
                pass
        else:
            if not line:
                collecting = False
                definitions[current_code] = ('no title',current_definition)
            elif tp.has_code(line):
                definitions[current_code] = ('no title',current_definition)
                current_code = tp.get_code(line)
                current_definition = ''
            else:
                current_definition += line
    return definitions

def main():
    definitions_77 = make_77_occ_dictionary()
    save_data(definitions_77,'1977')
    definitions_65 = make_65_occ_dictionary()
    save_data(definitions_65,'1965')
