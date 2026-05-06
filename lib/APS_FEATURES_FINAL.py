import re 
import pandas as pd 
import tqdm 
from nltk.util import bigrams
from statistics import mean
from nltk import ngrams
import os 

nupos_classes = {
    "acp": "acp word as adverb; adv/conj/pcl/prep",
    "an": "adverb/noun",
    "av": "adverb",
    "cc": "coordinating conjunction",
    "crq": "wh-word",
    "cs": "subordinating conjunction",
    "d": "determiner",
    "fw": "foreign word",  
    "j": "adjective",
    "np": "proper noun",
    "n": "noun",
    "crd": "numeral",
    "pf": "preposition of",
    "pi": "indefinite pronoun",
    "pn": "personal pronoun",
    "po": "possessive pronoun",
    "pp": "preposition",
    "pu": "punctuation",
    "px": "reflexive pronoun",
    "sy": "symbol",
    "uh": "interjection",
    "va": "auxiliary verb",
    "vm": "modal verb",
    "v": "verb",
    "xx": "negative",
    "zz": "undetermined"
}
content = ['an', 'av','fw','j', 'np', 'n', 'crd','va', 'vm', 'v']
function = ['acp', 'cc', 'crq', 'cs', 'd', 'pf', 'pi', 'pn', 'po', 'pp', 'pu', 'px', 'sy', 'uh', 'xx', 'zz']
content = "|".join(content)
function = "|".join(function)

def replace(pattern, str): 
    str = re.sub(pattern," ",str)
    str = re.sub(r"\s+"," ",str).strip()
    return str 

def get_spans(pattern,tokens,pos,lemmata): 
    one = re.findall(pattern,tokens)
    two = re.findall(pattern,pos)
    three = re.findall(pattern,lemmata)
    return one,two,three

more_features_csv = pd.read_csv("../features/HIGH_FREQ_FEATURES.csv").to_dict(orient='records')
MORE_FEATURES = {}
for item in more_features_csv: 
    MORE_FEATURES[item['item']] = item['tag']

AUTHORS_FEATURES = pd.read_csv("../features/AUTHORS_WIKI_FEATURES.csv").to_dict(orient='records')
AUTHORS_FEATURES = {a['TCP_name_heading']:a for a in AUTHORS_FEATURES}
tcpID_aut_csv = pd.read_csv("../CORPUS/tcpID_to_author.csv").to_dict(orient='records')
TCPID_AUTHORS = {}
for item in tcpID_aut_csv: 
    if item['tcpID'] not in TCPID_AUTHORS: TCPID_AUTHORS[item['tcpID']] = []
    TCPID_AUTHORS[item['tcpID']].append(item['author'])

def reinitialize_features(tcpID): 
    features = {}
    features["COUNT_tokens"] = 0 # equivalent to DIV length
    features["COUNT_types"] = set() # convert to int length later
    features["sentence_length"] = [] # take the mean later 
    features['content'] = 0 
    features['function'] = 0 # not counting punctuation 
    features['punctuation'] = 0 # includes boundaries 
    features["female_pronouns"] = 0 
    features["male_pronouns"] = 0 
    features["plural_pronouns"] = 0 
    for f_type in ['char4grams', 'pos2grams', 'np', '<i>']: 
        features[f"COUNT_{f_type}"] = 0 
        features[f"TOP_{f_type}"] = 0 
    for f_type in ["<n>", "<l>", "<h>","<i>"]: 
        features[f"COUNT_{f_type}"] = 0 
    features.update({pos:0 for pos in nupos_classes})
    features.update({pos:0 for pos in MORE_FEATURES})
    authors = TCPID_AUTHORS[tcpID]
    for aut in authors:
        if aut not in AUTHORS_FEATURES: continue 
        aut_info = AUTHORS_FEATURES[aut] 
        for f, count in aut_info.items(): 
            if f == "TCP_name_heading": continue 
            if count != 0: 
                features[f] = 1 
    return features 

def reinitialize_spans(): 
    return {'tokens':"",'pos':"","lemmata":""}

def count_feature(lemma,pos,features): 
    features['COUNT_tokens'] += 1 
    if lemma.lower() == "she": 
        features["female_pronouns"] += 1 
    elif lemma.lower() == "he": 
        features["male_pronouns"] += 1 
    elif lemma.lower() == "they": 
        features["plural_pronouns"] += 1 
    if lemma == pos: # punctuation 
        if lemma not in features: 
            features[lemma] = 0
        features[lemma] += 1 
        features['punctuation'] += 1 
    elif re.search(content,pos):
        features['content'] += 1 
    else: 
        features['function'] += 1  
    for item in nupos_classes: 
        if item in pos: 
            features[item] += 1 
            # can only be in one class 
            break
    return features 

def update_span(span_dict, t,p,l): 
    span_dict['tokens'] += " {}".format(t.lower())
    span_dict['pos'] += " {}".format(p.lower())
    span_dict['lemmata'] += " {}".format(l.lower())
    return span_dict 

def process(fname): 
    texts = pd.read_csv(fname).to_dict(orient='records')
    fname = fname.split("APS_TEXTS_")[-1].split(".")[0]

    FEATURES_DICT = {} # key (tcpID, DIV_idx), value is the features dict 

    def export(): 
        features_df = []  
        # avg num of tokens in DDC_FINAL: 5640      
        # print("\n",mean([FEATURES_DICT[k]['COUNT_tokens'] for k in FEATURES_DICT]),"\n")
        for key, f in FEATURES_DICT.items(): 
            if len(f['sentence_length']) == 0: continue 
            value = {'tcpID':key[0],'curr_div_idx':key[1]}
            value['type_token_ratio'] = len(f['COUNT_types']) / f['COUNT_tokens']
            COUNT_pns = f['female_pronouns'] + f['male_pronouns'] + f['plural_pronouns']
            for item,count in f.items(): 
                # average proportion of sentence to DIV 
                if item in ["sentence_length"]: 
                    value[item] = mean(f[item]) / f['COUNT_tokens']
                elif item == "COUNT_tokens": 
                    value[item] = count 
                # do not add other category counts to feature vector
                elif "COUNT" in item and "<" not in item: continue  
                # proportion of gendered pronouns to all pronouns 
                elif item in ["female_pronouns","male_pronouns","plural_pronouns"]: 
                    if COUNT_pns > 0: 
                        value[item] = count / COUNT_pns
                # proportion of different vocab classes or high freq vocab feature to all tokens 
                elif item in ["content","function","punctuation"] or "vocab" in item:
                    if 'vocab' in item: 
                        item = item.split(" - vocab")[0]
                    value[item] = count / f['COUNT_tokens']
                # Boolean for high-freq span features 
                elif item in MORE_FEATURES.keys() or "author" in item: 
                    # boolean: true if this high-freq feature occurs 
                    value[item] = count 
                # num tokens in high-freq spans of one category to total num tokens in that span category in DIV 
                elif "TOP_" in item: 
                    name = item.split("TOP_")[-1]
                    if f[f'COUNT_{name}'] > 0: 
                        value[item] = count / f[f'COUNT_{name}']
                # POS 
                elif f"{item} - vocab" not in f: 
                    value[item] = count / f['COUNT_tokens']
            features_df.append(value)
        features_df = pd.DataFrame(features_df)
        features_df = features_df.fillna(0)
        features_df.to_csv(f"../feature_vectors/{fname}.csv",index=False)
    
    tcpID_div_to_num_tokens = {}
    for entry in texts:
        key = (entry['tcpID'],entry['curr_div_idx']) 
        if key not in tcpID_div_to_num_tokens: tcpID_div_to_num_tokens[key] = 0 
        tcpID_div_to_num_tokens[key] += len(entry['lemmata'].split())

    progress = tqdm.tqdm(enumerate(texts))
    old_key = (None,None)
    for idx, entry in progress: 
        # if entry['tcpID'] not in ['DDC_FINAL','A25291','A96805']: continue 
        key = (entry['tcpID'],entry['curr_div_idx']) 
        if key not in FEATURES_DICT: # new DIV 
            # tally features for each separate DIV 
            # check if prev div is too short 
            if old_key[0] is None:  
                # first DIV 
                FEATURES_DICT[key] = reinitialize_features(entry['tcpID'])
                old_key = key
            elif FEATURES_DICT[old_key]['COUNT_tokens'] < 2000: 
                # prior DIV is not long enough
                key = old_key
            else: 
                FEATURES_DICT[key] = reinitialize_features(entry['tcpID'])
                old_key = key
        else: 
            # split up overly large DIVs 
            half = tcpID_div_to_num_tokens[key] // 2 
            key = (entry['tcpID'],f"{entry['curr_div_idx']} | {idx}")
            if FEATURES_DICT[old_key]['COUNT_tokens'] >= 5000 and half >= 5000: 
                FEATURES_DICT[key] = reinitialize_features(entry['tcpID'])
                old_key = key 
            else: 
                key = old_key 
        
        # get POS bigrams
        pos2grams = ["_".join(n) for n in ngrams(entry['pos'].split(), 2) if not ("<" in n[0] and "<" in n[1])]
        for gram in pos2grams: 
            if gram in MORE_FEATURES: 
                FEATURES_DICT[key][gram] = 1
                FEATURES_DICT[key][f"TOP_pos2grams"] += 1
            FEATURES_DICT[key][f"COUNT_pos2grams"] += 1  
        
        # get character quadrigrams of the lemmata 
        chars = []
        for word in entry['lemmata'].lower().split(): 
            if "<" in word and ">" in word: 
                chars.append(word)
            elif ".." not in word: 
                for c in word: 
                    chars.append(c)
            chars.append("_")
        chars = chars[:-1]
        char4grams = ["".join(n) for n in ngrams(chars, 4) if len([x for x in n if "<" in x]) < 2]
        for gram in char4grams: 
            if gram in MORE_FEATURES: 
                FEATURES_DICT[key][gram] = 1 
                FEATURES_DICT[key][f"TOP_char4grams"] += 1
            FEATURES_DICT[key][f"COUNT_char4grams"] += 1 
        
        # get other features 
        t_list, pos_list, l_list = entry['tokens'].split(" "),entry['pos'].split(" "),entry['lemmata'].split(" ")
        curr_spans = { # start: [end, False, span]
            '<i>': {'end_tag': '</i>','in_span': False, 'span':reinitialize_spans()},
            '<n>': {'end_tag': '</n>','in_span': False, 'span':reinitialize_spans()},
            '<h>': {'end_tag': '</n>','in_span': False, 'span':reinitialize_spans()},
            '<l>': {'end_tag': '</l>','in_span': False, 'span':reinitialize_spans()}, 
        }
        pos_spans = {
            'np': {'span':reinitialize_spans()}
        }
        curr_s_len = 0
        def normalize_word(word): 
            word = word.lower()
            word = re.sub("-","",word)
            return word 
        for t, p, l in zip(t_list, pos_list, l_list): 
            # LOWER CASE AND STRIP HYPHENS 
            t,p,l = t.lower(), p.lower(), l.lower()
            t = normalize_word(t)
            l = normalize_word(l)
            if ".." not in p: # ignore dividers 
                if t == "</s>": # end of a sentence 
                    FEATURES_DICT[key]['sentence_length'].append(curr_s_len)
                    curr_s_len = 0 # reset 
                elif t != "<s>": # not inclusive of boundaries 
                    curr_s_len += 1 # not including <s> tags 
                    FEATURES_DICT[key] = count_feature(l,p,FEATURES_DICT[key])
                    lemma_key = f"{l} ({p})"
                    FEATURES_DICT[key]['COUNT_types'].update({t})
                    if lemma_key in MORE_FEATURES: 
                        lemma_key = lemma_key + " - vocab"
                        if lemma_key not in FEATURES_DICT[key]: FEATURES_DICT[key][lemma_key] = 0 
                        FEATURES_DICT[key][lemma_key] += 1
            for tag in curr_spans.keys(): 
                if tag == t: # start of a new span 
                    curr_spans[tag]['in_span'] = True 
                elif t == curr_spans[tag]['end_tag']: # end of the span 
                    # add to master dict 
                    c = curr_spans[tag]['span']
                    entry = c['lemmata'].strip() # convert list to tuple to use as key 
                    if entry in MORE_FEATURES and tag == "<i>": # applies to <i> only 
                        FEATURES_DICT[key][entry] = 1 
                        FEATURES_DICT[key][f"TOP_{tag}"] += 1
                    FEATURES_DICT[key][f"COUNT_{tag}"] += 1 
                    curr_spans[tag]['in_span'] = False # reset flag 
                    curr_spans[tag]['span'] = reinitialize_spans() # reset span 
                elif curr_spans[tag]['in_span']: # a lemma in the span 
                    curr_spans[tag]['span'] = update_span(curr_spans[tag]['span'], t,p,l)
            for tag in pos_spans.keys(): 
                if tag in p: # same pos tag 
                    pos_spans[tag]['span'] = update_span(pos_spans[tag]['span'], t,p,l)
                elif tag not in p:  
                    c = pos_spans[tag]['span']
                    if len(c['lemmata']) > 0: 
                        # if there is an active np span  
                        entry = c['lemmata'].strip()
                        # if this np span is a high freq feature  
                        if entry in MORE_FEATURES:
                            FEATURES_DICT[key][entry] = 1 
                            FEATURES_DICT[key][f"TOP_{tag}"] += 1
                        # total number of np spans 
                        FEATURES_DICT[key][f"COUNT_{tag}"] += 1  
                    pos_spans[tag]['span'] = reinitialize_spans() # reset tag      
    export()

if __name__ == "__main__": 
    progress = tqdm.tqdm(sorted(os.listdir("../CORPUS")))
    for fname in progress: 
        if "APS_TEXTS" in fname: 
            progress.set_description(fname)
            fname = "../CORPUS/"+fname 
            process(fname)
            