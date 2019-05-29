import re
import csv
import random
import math

DOT_1977_PATH = '/Users/joseivelarde/Projects/dot-nlp/data/raw/1977_Output.txt'
OUTFILE = '/Users/joseivelarde/Projects/dot-nlp/data/clean/structured_1977.csv'
DATA_DIR = '/Users/joseivelarde/Projects/dot-nlp/data/1977/'

class TitleParser:
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

def load_dot():
    with open(DOT_1977_PATH,'rb') as f:
        dot = f.readlines()
    dot = [line.decode('utf-8',errors='ignore') for line in dot]
    dot = [line.rstrip() for line in dot]
    dot = [line for line in dot if line != '']
    return dot

def make_occ_dictionary(dot):
    tp = TitleParser()
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

def write_set(definitions,set_type):
    outfile = DATA_DIR+set_type+'.csv'
    with open(outfile,'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in definitions.items():
            writer.writerow([key,value[0],value[1]])

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

dot = load_dot()
definitions = make_occ_dictionary(dot)
datasets = train_dev_test_split(definitions)

write_set(datasets['train'],'train')
write_set(datasets['dev'],'dev')
write_set(datasets['test'],'test')
