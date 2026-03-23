import os
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize, sent_tokenize

def get_median(ls):
    # sort the list
    ls_sorted = ls.sort()
    # find the median
    if len(ls) % 2 != 0:
        # total number of values are odd
        # subtract 1 since indexing starts at 0
        m = int((len(ls)+1)/2 - 1)
        return ls[m]
    else:
        m1 = int(len(ls)/2 - 1)
        m2 = int(len(ls)/2)
        return (ls[m1]+ls[m2])/2

if __name__ == '__main__':

    mimic3_path = r"./mimic-iii-clinical-database-1.4"

    sumInputSec, sumOutputSec = [],[] 
    with open("./section_titles/mimic_sec_SUM.txt", "r", encoding="UTF-8") as f_read:
        sumSec = f_read.readlines()
        sumSec = [_.strip().lower() for _ in sumSec if len(_.strip()) > 0]
        special_index = sumSec.index('===============')
        
        sumOutputSec = list(sumSec[:special_index])
        sumInputSec = list(sumSec[special_index+1:])

    print(sumInputSec)
    print(sumOutputSec)
        
    file_path_list = []
    for file in [f for f in os.listdir(mimic3_path) if f.endswith(".csv.gz")]:
        file_path = os.path.join(mimic3_path, file)
        file_path_list.append(file_path)

    file_path_list.sort()
    
    chunksize = 50000
    NOTEEVENTS = file_path_list[18]

    # ======= ECHO =======
    tgt_column = 'Echo'
    tgt_df = None
    with pd.read_csv(NOTEEVENTS, chunksize=chunksize) as reader:
        for chunk in reader:
            check_subject = chunk[chunk['CATEGORY'] == tgt_column]
            check_subject = check_subject[check_subject['ISERROR'].isna()] # find rows that are not errors
            if check_subject.shape[0] > 0:
                if tgt_df is None:
                    tgt_df = check_subject.copy()
                else:
                    tgt_df = pd.concat([tgt_df, check_subject])
    print('tgt_df', tgt_df.shape)
    
    echoSec = []
    echoInputSec, echoOutputSec = [], []
    avg_sent_input, avg_word_input = 0, 0
    avg_sent, avg_word, count_text = 0, 0, 0

    with open("./section_titles/mimic_sec_ECHO.txt", "r", encoding="UTF-8") as f_read:
        echoSec = f_read.readlines()
        echoSec = [_.strip().lower() for _ in echoSec if len(_.strip()) > 0]
        special_index = echoSec.index('===============')
        
        echoInputSec = list(echoSec[:special_index])
        echoOutputSec = list(echoSec[special_index+1:])

    data_list = []
    
    for idx in tqdm(range(tgt_df.shape[0])):
        raw_text = tgt_df.iloc[idx]['TEXT'].lower()
        inputText, outputText = '', ''
        
        search_res_list = []
        for section in echoSec:
            search_res = re.search(section, raw_text)
            if search_res is not None:
                indice_0, indice_1 = search_res.span()
                search_res_list.append([indice_0, indice_1, section])

        section_count = len(search_res_list)
        if section_count > 0:
            search_res_list = sorted(search_res_list, key=lambda x:x[0])
            flag1, flag2 = False, False
            for sec_res in search_res_list:
                if sec_res[-1] in echoInputSec:
                    flag1 = True
                elif sec_res[-1] in echoOutputSec:
                    flag2 = True
            if flag1 and flag2:
                for res_i in range(section_count):
                    sec_res = search_res_list[res_i]
                    if res_i == section_count - 1:
                        if sec_res[-1] not in echoOutputSec:
                            print(search_res_list)
                        assert sec_res[-1] in echoOutputSec
                        outputText += raw_text[sec_res[1]:].strip()
                    else:
                        sec_res_next = search_res_list[res_i+1]
                        inputText += raw_text[sec_res[0]:sec_res_next[0]].strip() + '\n'

                inputText = inputText.strip()
                matches = re.findall(r'this study was compared to the prior study of[\w\W]+\*\*].', inputText)
                for _ in matches:
                    inputText = inputText.replace(_, '')
                    
                inputText = inputText.strip()
                outputText = outputText.strip()

                # write source and summary into dict
                data_list.append(
                    {'source': inputText,
                    'summary': outputText}
                )

                token_sent_input = sent_tokenize(inputText)
                token_word_input = word_tokenize(inputText)
                avg_sent_input += len(token_sent_input)
                avg_word_input += len(token_word_input)

                token_sent = sent_tokenize(outputText)
                token_word = word_tokenize(outputText)

                avg_sent += len(token_sent)
                avg_word += len(token_word)
                count_text += 1

    to_write_json = {'data': data_list}

    json_name = "ECHO.json"
    with open(os.path.join('./dataset', json_name), 'w', encoding='utf-8') as write_f:
        write_f.write(json.dumps(to_write_json))
                
    print('echo', tgt_df.shape)
    print(count_text / tgt_df.shape[0])
    print(count_text, 'avg_sent_input', avg_sent_input / count_text, 'avg_word_input', avg_word_input / count_text)
    print(count_text, 'avg_sent', avg_sent / count_text, 'avg_word', avg_word / count_text)
    # input: avg_sent_input 30.86580486303478 avg_word_input 315.29775315481686
    # output: avg_sent 4.158079409048938 avg_word 49.98664204370576
    # count: 16245 / 45794
    # ======= ECHO =======
