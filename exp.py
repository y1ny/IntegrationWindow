import os
import gc
import argparse
os.chdir("/path/to/workdir")
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys
import pandas as pd
import csv
import random
import copy
import string
from tqdm import tqdm
import numpy as np
import spacy
from utils import createDir
from utils import save_pkl
from utils import word_tokenize
def get_next_token_distribution(total_input):
    if len(total_input) < bs_size:
        lst = [np.array(total_input)]
    else:
        lst = np.array_split(total_input, len(total_input) // bs_size)
    max_steps = 5
    info = [[] for _ in range(max_steps)]
    
    current_start = 0
    current_end = 0
    for l_idx, input_lst in tqdm(enumerate(lst), total=len(lst)):
        
        input_lst = input_lst.tolist()
        current_end += len(input_lst)
        tokenized = tokenizer(input_lst, padding=True, return_tensors="pt")
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

        past_key_values = None 
        for step in range(max_steps):
            with torch.no_grad():
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
                outputs = model(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values, 
                                use_cache=True)
                logits = outputs.logits  # [batch_size, seq_length, vocab_size]
                past_key_values = outputs.past_key_values
                # 16 layers × 2 (key and value)
                # batch size × head_num × seq_length × d_key

            tmp_input = []
            sub_info = []
            for i in range(len(input_lst),):
                
                last_token_logits = logits[i, -1, :tokenizer.vocab_size]  

                probs = softmax(last_token_logits, dim=-1)  
                top_k_probs, top_k_indices = torch.topk(probs, k=200)
                output_distribution = {}
                for idx, p in zip(top_k_indices, top_k_probs):
                    output_distribution[int(idx.item())] = round(p.item(), 4)
                
                next_token_id = last_token_logits.argmax(dim=-1).unsqueeze(-1).unsqueeze(-1)

                tmp_input.append([next_token_id.item()])
                sub_info.append([tokenizer.decode([next_token_id.item()]), output_distribution])
            
            input_ids = torch.tensor(tmp_input).to('cuda')
            to_append = torch.ones(attention_mask.size(0), 1).to('cuda')
            attention_mask = torch.cat([attention_mask, to_append], dim=1)
            info[step].extend(sub_info)
        current_start += len(input_lst)
    gc.collect()
    return info

def search_length(sent_lst, n_context, n_window):
    for start_idx in range(len(sent_lst)):
        current_length = 0
        current_lst = []
        for idx, sent in enumerate(sent_lst):
            if idx < start_idx:
                continue
            current_length += len(sent.split(' '))
            current_lst.append(sent)
            if current_length == n_context:
                if idx + 1 == len(sent_lst):
                    # return None, None, None
                    continue
                
                if len(sent_lst[idx + 1].split(' ')) > n_window:
                    window = sent_lst[idx + 1].split(' ')[:n_window]
                    predicted_words = sent_lst[idx + 1].split(' ')[n_window:]
                    checked = sent_lst[idx + 1].split(' ')[:n_window + 1]
                    checked = ' '.join(checked)
                    checked = word_tokenize([checked])[0]
                    if set(list('.,!:?')).intersection(set(checked)):
                        continue
                    if len(window) == 1 and window[0] in string.punctuation:
                        continue
                    return current_lst, [' '.join(window), ' '.join(predicted_words)], sent_lst[idx + 1 :]
            elif current_length > n_context:
                break
    return None, None, None

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='set window length in bash')
    parser.add_argument('-w','--n_window',default=5,dest='n_window', type=int, help='window length')

    args = parser.parse_args()
    n_window_lst = args.n_window
    data_path = "article/mixed_data.csv"
    test_type = 'exp1'
    logger.info(f"test type: {test_type}")
    logger.info(f"loading data from: {data_path}")
    csv_file = pd.read_csv(data_path,
                           delimiter='\t',
                           quoting=csv.QUOTE_NONE,
                           quotechar=None,)
    
    key = 'article'
    logger.info(f"sentence key: {key}")
    art_lst = csv_file[key].values.tolist()
    art_lst = list(set(art_lst))
    random_words = []
    for art in art_lst:
        random_words.extend(art.split(" "))

    nlp = spacy.load("en_core_web_md",disable = ['ner', 'tagger', 'parser', 'tok2vec'])
    nlp.add_pipe('sentencizer')

    docs =[]
    for doc in tqdm(nlp.pipe(art_lst, batch_size=16, n_process=4), total=len(art_lst)):
        docs.append(doc)
    
    total_sent_lst = []
    for doc in docs:
        sent_lst = []
        for s in doc.sents:
            sent_lst.append(s.text)
        total_sent_lst.append(sent_lst)
    total_test = []
    logger.info("Constructing data...")
    for n_context in [100]:
        logger.info(f"contructing # context = {n_context}")
        for n_window in tqdm([n_window_lst]):
            test_lst = []
            for sent_lst in total_sent_lst:
                raw_sent_lst = copy.deepcopy(sent_lst)

                while True:
                    context, next_sent, sent_lst = search_length(sent_lst, n_context-n_window, n_window)
                    if not context:
                        break
                    predicted_words = next_sent[1]
                    window = next_sent[0].split(' ')
                    context = ' '.join(context).split(' ')
                    assert len(context) + n_window == 100
                    target_word = predicted_words.split(' ')[0]
                    if target_word.isalpha():
                        lst = []
                        test_range = list(range(n_window - 100, n_window + 100))
                        test_range = list(filter(lambda x: x>0 and x <=20, test_range)) # 设定heatmap长度
                        for idx in test_range:
                            idx = idx - n_window
                            if idx < 0:
                                new_window =  window[-idx:]
                                new_context = context + window[:-idx]
                            elif idx == 0:
                                context_end = list(context[-1])
                                clean_context = []
                                punc_context = []
                                for c_idx, ch in enumerate(context_end[::-1]):
                                    if ch in string.punctuation:
                                        punc_context.append(ch)
                                    else:
                                        clean_context = context_end[:-c_idx]
                                        punc_context = punc_context[::-1]
                                        break
                                new_window = [''.join(punc_context) + '$marker$' + window[-idx]] + window[-idx+1:]
                                new_context = context[:-1] + [''.join(clean_context)]
                            else:
                                new_window = context[-idx:] + window
                                new_context = context[:-idx]
                            lst.append([' '.join(new_context), [' '.join(new_window), predicted_words], raw_sent_lst])
                        test_lst.append(lst)

            test_lst = random.sample(test_lst, 1000)
            total_test.append([test_lst, n_context, n_window])

    gc.collect()
    logger.info("Evaluation...")
    torch.cuda.set_device(0)
    bs_size = 32
    for name in ['gpt','qwen']:
        if name == 'gpt':
            model_name = 'openai-community/gpt2'
        elif name == 'qwen':
            model_name = 'Qwen/Qwen2.5-1.5B'

        logger.info(f"model from: {model_name}" )

        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        
        for info in total_test:
            
            test_lst = info[0]
            n_context = info[1]
            n_window = info[2]
            save_path = f"data/{test_type}-context-{n_context}/{name}-window-{n_window}.pkl"
            if os.path.exists(save_path):
                continue
            total_lst = []
            logger.info(f"# context = {n_context}, # window = {n_window}" )
            for seed in [11,22,33,44,55,66,128,256,233,1024] :
                logger.info(f"shuffle sentences with seed {seed}" )
                random.seed(seed)
                sub_lst = []
                origin_input = []
                shuffle_input = []
                white_input = []
                baseline_input = []
                split_flag = []
                flag = True
                for lst in test_lst:
                    flag = not flag
                    for item in lst:
                        window = item[1][0]
                        if "$marker$" in window:
                            window = window.replace("$marker$", ' ')
                            origin = item[0] + window
                            shuffled = item[0].split(' ')
                            random.shuffle(shuffled)
                            shuffled = ' '.join(shuffled) + window
                            
                            baseline = ' '.join(random.sample(random_words, n_context)) + window
                            white = window
                        else:
                            origin = item[0] + ' ' + window
                            shuffled = item[0].split(' ')
                            random.shuffle(shuffled)
                            shuffled = ' '.join(shuffled) + ' ' + window
                            
                            baseline = ' '.join(random.sample(random_words, n_context)) + ' ' + window
                            white = window
                        shuffle_input.append(shuffled)
                        split_flag.append(flag)
                        white_input.append(white)
                        baseline_input.append(baseline)
                    origin_input.append(origin)
                white_info = get_next_token_distribution(white_input) # 把whiteboard的position ids强制赋值
                origin_info = get_next_token_distribution(origin_input)
                shuffled_info = get_next_token_distribution(shuffle_input)
                baseline_info = get_next_token_distribution(baseline_input)
                current = -1
                current_lst = []
                baseline_lst = []
                white_lst = []
                for idx, flag in enumerate(split_flag):
                    if current == -1:
                        tmp = [t[idx] for t in shuffled_info] 
                        current_lst.append(tmp)

                        tmp = [t[idx] for t in baseline_info] 
                        baseline_lst.append(tmp)

                        tmp = [t[idx] for t in white_info] 
                        white_lst.append(tmp)
                        current = flag
                    elif current != flag:
                        tmp = [t[len(sub_lst)] for t in origin_info]
                        sub_lst.append([tmp, current_lst, baseline_lst, white_lst])
                        
                        tmp = [t[idx] for t in shuffled_info]
                        current_lst = [tmp]
                        tmp = [t[idx] for t in baseline_info]
                        baseline_lst = [tmp]
                        tmp = [t[idx] for t in white_info]
                        white_lst = [tmp]
                        
                        current = flag
                    elif idx == len(split_flag) - 1:
                        tmp = [t[idx] for t in shuffled_info]
                        current_lst.append(tmp)

                        tmp = [t[idx] for t in baseline_info]
                        baseline_lst.append(tmp)

                        tmp = [t[idx] for t in white_info]
                        white_lst.append(tmp)
                        
                        tmp = [t[len(sub_lst)] for t in origin_info]
                        sub_lst.append([tmp, current_lst, baseline_lst, white_lst])
                    elif current == flag:
                        tmp = [t[idx] for t in shuffled_info]
                        current_lst.append(tmp)

                        tmp = [t[idx] for t in baseline_info]
                        baseline_lst.append(tmp)
                        
                        tmp = [t[idx] for t in white_info]
                        white_lst.append(tmp)
                del baseline_info
                del shuffled_info
                gc.collect()
                total_lst.append(sub_lst)
            save_path = f"data/{test_type}-context-{n_context}/{name}-window-{n_window}.pkl"
            createDir('/'.join(save_path.split("/")[:-1]))
            logger.info(f"save to {save_path}")
            save_pkl(save_path, [total_lst, test_lst])
            del total_lst
            gc.collect()
