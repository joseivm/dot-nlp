import pandas as pd
import os
import ast
import numpy as np
# from make_title_mapping import ParserHelper
import re

##### Parsing Classes #####
class ParserHelper:
    def __init__(self):
        self.code_regex = re.compile("[\d]{1}[\s\-]{1,4}[\d]{2}[\.\d]*")
        self.starts_with_full_title_regex = re.compile("^[A-Z]{3,}[A-Z \-\,\.']+[\(a-z\s\-\.,;\& \)]*\([a-zA-Z\s\-\.,;\&]+\)")
        self.starts_with_title_regex = re.compile('''^[A-Z"]{3,}[A-Z \-\,\.']+''')
        self.starts_with_numeral_and_industry_regex = re.compile("^\([IVl]{1,4}\) \([a-z\s\-\.,;\&]+\)")
        self.starts_with_numeral_regex = re.compile("^\([IVl]{1,4}\)")
        self.starts_with_industry_regex = re.compile("^\([a-z\s\-\.,;\&]+\)")
        self.title_regex = re.compile('''[A-Z"]{3,}[A-Z \-\,\.']+''')
        self.numeral_regex = re.compile("\([IVl]{1,4}\)")
        self.industry_regex = re.compile("\([a-z\s\-\.,;\&]+\)")
        self.reference_regex = re.compile('''(^[Ss]ee |^[Aa] [A-Z"]{3,}[A-Z \-\,\.']+ |^[Aa]n |^[Ss]pec[\.,] for )''')
        self.see_under_regex = re.compile("(^see under|^see [A-Z]+[A-Z \-\,]+ under [A-Z]+[A-Z \-\,]+\([a-zA-Z\s\-\.,;\&]+\)+)")
        self.leading_period_regex = re.compile("^\.")
        self.end_regex = re.compile("(\,$|\.$|\. [A-Z]$)")
        self.perform_regex = re.compile('''([Pp]erform(s|ing| )[a-zA-Z ]*duties (of|as described under|described under) [A-Z]{3,}[A-Z \-\,']+)''')
        self.see_regex = re.compile('''(^[Ss]ee [A-Z]{3,}[A-Z \-\,']+)''')
        self.end_num_regex = re.compile(" [I]{1,3}$")
        self.space_bef_hyphen_regex = re.compile("[A-Z]+ \-[A-Z]+")
        # self.title_after_perform_regex = re.compile('''([Pp]erform(s|ing| )[a-zA-Z ]*duties (of|as described under|described under) [A-Z]{3,}[A-Z \-\,\.']+)''')

    def has_reference(self, line):
        reference_match = self.reference_regex.search(line)
        return reference_match is not None

    def get_reference_test(self, line):
        perform_reference_match = self.perform_regex.search(line)
        see_reference_match = self.see_regex.search(line)
        if see_reference_match is not None:
            return see_reference_match.group(0) ## returns this if match first
        if perform_reference_match is not None:
            return perform_reference_match.group(0)
        else:
            return "not found"

    def has_see_or_perform_reference(self, line): ## 1965 definitions are either see XXX or perform(s/ing) (other) duties ..
        see_reference_match = self.reference_regex.search(line)
        perform_reference_match = self.perform_regex.search(line)
        return see_reference_match is not None or perform_reference_match is not None

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

    def has_numeral_and_industry(self,line):
        numeral_match = self.starts_with_numeral_regex.search(line)
        numeral_match = numeral_match is not None
        industry_match = self.has_industry(line)
        # match = self.starts_with_numeral_and_industry_regex.search(line)
        return numeral_match and industry_match

    def has_numeral_no_industry(self,line):
        numeral_match = self.starts_with_numeral_regex.search(line)
        numeral_match = numeral_match is not None
        industry_match = self.has_industry(line)
        return(numeral_match and not industry_match)

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
        industry = industry.strip()
        return(industry)

    def get_numeral(self,title):
        if self.has_numeral(title):
            numeral_match = self.numeral_regex.search(title)
            numeral = numeral_match.group(0)
            numeral = numeral.strip()
            numeral = re.sub("l",'I',numeral)
            numeral = re.sub("[\(\)]",'',numeral)
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
        if self.has_full_title(line):
            title = self.get_title(line)
            industry = self.get_industry(line)
            return(title + ' ' + industry)
        else:
            return None

    def remove_code(self,line):
        line = re.sub(self.code_regex,'',line)
        line = re.sub(self.leading_period_regex,'',line)
        return line

    def remove_trailing_period(self,line):
        line = re.sub(self.end_regex,'',line)
        return(line)

    def remove_trailing_I(self,title):
        line = re.sub(" [I]{1,3}$","",title)
        return(line)

    def remove_spaces_hyphen(self,title):
        title = re.sub("[A-Z]+ \-[A-Z]+",title.replace(" -","-"),title)
        title = re.sub("[A-Z]+\- [A-Z]+",title.replace("- ","-"),title)
        return(title)

def find_code_and_def(ph,df,title):
    total_matched = sum(df.Title == title)
    if total_matched == 0:
        return '', ''
    else:
        ii = 0
        nancode = 0
        while ii < total_matched:
            definition = df.loc[df.Title == title,'Definition'].astype(str).to_numpy()[ii]
            code = df.loc[df.Title == title,'Code'].to_numpy()[ii]
            check_see_regex = ph.see_regex.search(definition)

            if check_see_regex is None:
                return str(code), definition
            else:
                seetitle = ph.title_regex.search(check_see_regex.group(0)).group(0)
                # print(seetitle)
                ii += 1
                if ii == total_matched and seetitle is not None:
                    code, definition = find_code_and_def(ph,df,seetitle)
                    return str(code), definition

def find_titles(ph,listunfound,df,remove_num=False):
    c1 = 0
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    d0 = 0

    for i,title in enumerate(listunfound):
        if remove_num:
            title = re.sub(" [I,V,X]{1,3}$", "",title)
        code, defi = find_code_and_def(ph,df,title)
        # print(title)
        oricode = df.loc[ind[i],'Code']
        if defi != '' and defi != 'nan':
            df.loc[ind[i],'NewDef'] = defi
            # c1 += 1
            d1 += 1
            if code != 'nan' or str(oricode) != 'nan':
            # if str(oricode) == 'nan' and code != 'nan':
                df.loc[ind[i],'Code'] = code # replace nan code with matched code
                c1 += 1

        else:
            title2 = title.replace("-"," ") ## Step 2: replace spaces between hyphens
            code, defi = find_code_and_def(ph,df,title2)
            if defi != '' and defi != 'nan':
                df.loc[ind[i],'NewDef'] = defi
                d2 += 1
                if code != 'nan' or str(oricode) != 'nan':

                # if str(oricode) == 'nan' and code != 'nan':
                    df.loc[ind[i],'Code']
                    c2 += 1
            else:
                title3 = title.replace("- "," ") ## Step 3: replace hyphens with spaces
                code, defi = find_code_and_def(ph,df,title3)
                if defi != '' and defi != 'nan':
                    df.loc[ind[i],'NewDef'] = defi
                    d3 += 1
                    # if str(oricode) == 'nan' and code != 'nan':
                    if code != 'nan' or str(oricode) != 'nan':

                        df.loc[ind[i],'Code']
                        c3 += 1
                else:
                    title4 = title.replace("- ","-") ## Step 4
                    code, defi = find_code_and_def(ph,df,title4)
                    if defi != '' and defi != 'nan':
                        df.loc[ind[i],'NewDef'] = defi
                        d4 += 1
                        if code != 'nan' or str(oricode) != 'nan':

                        # if str(oricode) == 'nan' and code != 'nan':
                            df.loc[ind[i],'Code']
                            c4 += 1
                    else:
                        title5 = title.replace(" ","-") ## Step 5
                        code, defi = find_code_and_def(ph,df,title5)
                        if defi != '' and defi != 'nan':
                            df.loc[ind[i],'NewDef'] = defi
                            d5 += 1
                            if code != 'nan' or str(oricode) != 'nan':
                            # if str(oricode) == 'nan' and code != 'nan':
                                df.loc[ind[i],'Code']
                                c5 += 1
                        else:
                            df.loc[ind[i],'NewDef'] = 0
                            d0 += 1

    print(f"1st: {c1} codes (ori/matched) exists out of {d1}" )
    print(f"2nd: {c2} codes (ori/matched) exists out of {d2}" )
    print(f"3: {c3} codes (ori/matched) exists out of {d3}" )
    print(f"4: {c4} codes (ori/matched) exists out of {d4}" )
    print(f"5: {c5} codes (ori/matched) exists out of {d5}" )
    print(f"unmatched: {d0}")

    return df

def main():
    ph = ParserHelper()
    file = "/home/charlotte/Desktop/econ-proj/supercloud/gdrive2/domain_1965_1939md_full_data_preds_full.csv"
    df = pd.read_csv(file)

    df['HasRef']=df['Original_Definition'].astype(str).apply(ph.has_reference)

    ind = np.where((df.Definition.isnull() & df.HasRef))[0]
    df.loc[ind,'Reftitle']=df.loc[ind,'Original_Definition'].apply(ph.get_title).astype(str).apply(ph.remove_trailing_period)
    listunfound = df.loc[ind,'Reftitle']

    df = find_titles(ph,listunfound,df)
    ind2 = np.where((df.NewDef == 0))[0]
    listunfound2 = df.loc[ind2,'Reftitle']
    print("Now remove trailing numerals")
    df = find_titles(ph,listunfound,df,remove_num=True)

    df.to_csv("/home/charlotte/Desktop/econ-proj/data/raw/newdef_1939.csv",index=False)

    ## make new full data
    # df = pd.read_csv("/home/charlotte/Desktop/econ-proj/data/raw/full_data_49.csv")
    df = pd.read_csv("/home/charlotte/Desktop/econ-proj/data/raw/1939_DOT.csv")
    df2 = pd.read_csv("/home/charlotte/Desktop/econ-proj/data/raw/newdef_1949.csv")

    df2.NewDef = df2.NewDef.fillna('')
    df2.loc[df2.NewDef == '0', 'NewDef'] = ''
    df2[df2.NewDef != '']
    df['ReferenceDefinition']=df['ReferenceDefinition'].fillna("")
    df['NewDefinition'] = df['Definition'] + " " + df['ReferenceDefinition'] +  " " + df2['NewDef']

    newdf = pd.DataFrame()
    newdf['Title'] = df['Title']
    newdf['Code'] = df['Code']
    newdf['Definition'] = df['NewDefinition']
    newdf['indexinDOT'] = np.arange(len(df))
    newdf2 = newdf[newdf.Definition.str.split().str.len()>=10]

    newdf2.to_csv("/home/charlotte/Desktop/econ-proj/data/1939md/full_data.csv", index =False) # overwrite current file
    # df.to_csv("/home/charlotte/Desktop/econ-proj/data/raw/full_data_39_2.csv",index = False)
    df.to_csv("/home/charlotte/Desktop/econ-proj/data/raw/1939_DOT2.csv",index = False)

    # ## make new full data for 1977 from pepe match
    # df = pd.read_csv("/home/charlotte/Desktop/econ-proj/data/raw/full_data_77_2.csv")
    # df['ReferenceDefinition']=df['ReferenceDefinition'].fillna("")
    # df['NewDefinition'] = df['Definition'] + " " + df['ReferenceDefinition']
    #
    # newdf = pd.DataFrame()
    # newdf['Title'] = df['Title']
    # newdf['Code'] = df['Code']
    # newdf['Definition'] = df['NewDefinition']
    # newdf['indexinDOT'] = np.arange(len(df))
    # newdf2 = newdf[newdf.Definition.str.split().str.len()>=10]
    # newdf2.to_csv("/home/charlotte/Desktop/econ-proj/data/1977md/full_data.csv", index =False) # overwrite current file

main()
