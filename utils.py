import os
os.chdir("/path/to/workdir")
import pickle
import re


punc_str = "\".,?!;:"
def check_tokenize(s_list):
    new_list = []
    for token in s_list:
        new_token = token
        if len(re.findall(r"\W", token))!=0:
            if len(re.findall(r"\d", token)) == 0:
                for p in punc_str:
                    new_token = new_token.replace(p, " "+p+" ")
        new_list.append(new_token)
    new_sent = " ".join(new_list).strip()
    new_sent = new_sent.replace("  ", ' ')
    return new_sent.split()

def word_tokenize(sent):
    '''
    tokenize for English sentence. especially for " 's, 're, 100% ".
    '''
    token_list = []
    for s in sent:
        s_list = s.split()
        s_list = check_tokenize(s_list)
        token_list.append(s_list)
    clean_sent = []
    for t_list in token_list:
        clean_lst = []
        for token in t_list:
            if len(re.findall(r"\d", token)) == 0:
                clean_lst.append(token)
                continue
            n_d_n = list(re.finditer(r"\d(\.)\d", token))
            n_c_n = list(re.finditer(r"\d(\,)\d", token))
            n_d = list(re.finditer(r"\.", token))
            n_c = list(re.finditer(r"\,", token))
            t_list = list(token)
            n_d_n = [t.regs[1] for t in n_d_n]
            n_c_n = [t.regs[1] for t in n_c_n]
            n_d = [t.span() for t in n_d]
            n_c = [t.span() for t in n_c]
            for d_pos in n_d:
                if d_pos not in n_d_n:
                    t_list[d_pos[0]] = " . "
                    
            for c_pos in n_c:
                if c_pos not in n_c_n:
                    t_list[c_pos[0]] = " , "
            clean_lst.append("".join(t_list))
        clean_lst = " ".join(clean_lst)
        clean_lst = clean_lst.replace("  ", " ")
        clean_sent.append(clean_lst.split())
    return clean_sent

def save_pkl(path,data):
    with open(path,'wb') as f:
        pickle.dump(data,f)
        
def read_pkl(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def save_result(path, data, title, mode='w'):
    with open(path,mode) as f:
        if mode == 'w':
            f.write('\t'.join(title)+"\n")
        for row in data:
            if isinstance(row,list) or isinstance(row, tuple):
                if None in row:
                    continue
                f.write('\t'.join(row)+'\n')
            else:
                if row:
                    f.write(row+"\n")
    
def createDir(filePath):
    if os.path.exists(filePath):
        return
    else:
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)

def clean_str(raw_str):
    raw_str = raw_str.split(' ')
    raw_str = list(filter(lambda x: x and x.strip(), raw_str))
    return ' '.join(raw_str)


if __name__ =="__main__":
    tok = word_tokenize(["'Scope your community when you go out and be on the lookout for apple trees, peach trees, and other edible plants.'"])
    
