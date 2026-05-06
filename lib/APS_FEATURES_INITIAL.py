import re 
import pandas as pd 
import tqdm 
from nltk.util import bigrams
from statistics import mean
from nltk import ngrams
import spacy
NER = spacy.load('en_core_web_sm')

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

fname = "../CORPUS/APS_TEXTS_ground.csv"
texts = pd.read_csv(fname).to_dict(orient='records')

def replace(pattern, str): 
    str = re.sub(pattern," ",str)
    str = re.sub(r"\s+"," ",str).strip()
    return str 

def get_spans(pattern,tokens,pos,lemmata): 
    one = re.findall(pattern,tokens)
    two = re.findall(pattern,pos)
    three = re.findall(pattern,lemmata)
    return one,two,three

def reinitiate_features(): 
    features = {}
    features["num_tokens"] = 0 # equivalent to DIV length
    features["num_types"] = set() # convert to int length later
    features["sentence_length"] = [] # take the mean later 
    features['content'] = 0 
    features['function'] = 0 # not counting punctuation 
    features['punctuation'] = 0 # includes boundaries 
    features["female_pronouns"] = 0 
    features["male_pronouns"] = 0 
    features["plural_pronouns"] = 0 
    features.update({pos:0 for pos in nupos_classes})
    return features 
def reinitiate_spans(): 
    return {'tokens':"",'pos':"","lemmata":""}

def count_feature(lemma,pos,features): 
    features['num_tokens'] += 1 
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

FEATURES_DICT = {} # key (tcpID, DIV_idx), value is the features dict 
VOCAB = {'content':{},
         'function':{},
         'punctuation':{}} # to determine high-frequency words 
SPANS = { # spans 
        '<l>': {}, 
        '<i>': {},
        '<n>': {},
        'np': {},
    } 
GRAMS = {
        'char4grams': {},
        'pos2grams': {},
        'entities':{}
    }
entities = set()

def export(): 
    not_pos_features = {"content","function","punctuation","female_pronouns","male_pronouns","plural_pronouns",
                    "sentence_length","type_token_ratio"}
    excluded = ["num_tokens","num_types"]
    not_pos_features.update(entities)
    features_df = []

    for key, f in FEATURES_DICT.items(): 
        if len(f['sentence_length']) == 0: continue 
        value = {'tcpID':key[0],'curr_div_idx':key[1]}
        value['type_token_ratio'] = len(f['num_types']) / f['num_tokens']
        num_pns = f['female_pronouns'] + f['male_pronouns'] + f['plural_pronouns']
        num_entities = sum([freq for ent,freq in f.items() if ent in entities])
        for item,freq in f.items(): 
            if item in ["sentence_length"]: 
                value[item] = mean(f[item])
            elif item in excluded: continue 
            elif item in ["female_pronouns","male_pronouns","plural_pronouns"]: 
                if num_pns > 0: 
                    value[item] = freq / num_pns
                else: 
                    value[item] = 0 
            elif item in entities: 
                if num_entities > 0: 
                    value[item] = freq / num_entities
                else: 
                    value[item] = 0 
            elif item in ["content","function","punctuation"]:
                value[item] = freq / f['num_tokens']
            else: 
                value[item] = freq / f['num_tokens']
        features_df.append(value)
    features_df = pd.DataFrame(features_df)
    features_df = features_df.fillna(0)
    features_df.to_csv("../features/ground_features.csv",index=False)

    spans_df = []
    for tag,items in SPANS.items(): 
        for parts,freq in items.items(): 
            value = {'tag':tag,'frequency':freq, 'tokens':parts[0],'pos': parts[1],'lemmata':parts[2]}
            spans_df.append(value)
    spans_df = pd.DataFrame(spans_df)
    spans_df = spans_df.sort_values(by=['frequency'],ascending=False)
    spans_df.to_csv("../features/ground_spans.csv",index=False)

    grams_df = []
    for tag,items in GRAMS.items(): 
        for gram,freq in items.items(): 
            value = {'tag':tag,'frequency':freq, 'gram':gram}
            grams_df.append(value)
    grams_df = pd.DataFrame(grams_df)
    grams_df = grams_df.sort_values(by=['frequency'],ascending=False)
    grams_df.to_csv("../features/ground_grams.csv",index=False)

    vocab_df = []
    for category,items in VOCAB.items(): 
        for key,freq in items.items(): 
            value = {'category':category,'frequency':freq, 'lemma':f"{key[0]} ({key[1]})",'pos': key[1]}
            vocab_df.append(value)
    vocab_df = pd.DataFrame(vocab_df)
    vocab_df = vocab_df.sort_values(by=['frequency'],ascending=False)
    vocab_df.to_csv("../features/ground_vocab.csv",index=False)

progress = tqdm.tqdm(enumerate(texts))
for idx, entry in progress: 
    # if entry['tcpID'] != "DDC_FINAL": continue 
    key = (entry['tcpID'],entry['curr_div_idx']) 
    if key not in FEATURES_DICT: 
        # tally features for each separate DIV 
        FEATURES_DICT[key] = reinitiate_features()
    # get POS bigrams
    pos2grams = ["_".join(n) for n in ngrams(entry['pos'].split(), 2) if not ("<" in n[0] and "<" in n[1])]
    for gram in pos2grams: 
        if gram not in GRAMS['pos2grams']: GRAMS['pos2grams'][gram] = 0 
        GRAMS['pos2grams'][gram] += 1 
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
        if gram not in GRAMS['char4grams']: GRAMS['char4grams'][gram] = 0 
        GRAMS['char4grams'][gram] += 1 
    # get entities with spacy 
    doc = NER(entry['lemmata'])
    for ent in doc.ents:
        entities.update({ent.label_})
        gram_key = f"entities_{ent.label_}"
        if gram_key not in GRAMS: GRAMS[gram_key] = {}
        if ent.text not in GRAMS[gram_key]: GRAMS[gram_key][ent.text] = 0 
        GRAMS[gram_key][ent.text] += 1 
        if ent.label_ not in FEATURES_DICT[key]: FEATURES_DICT[key][ent.label_] = 0 
        FEATURES_DICT[key][ent.label_] += 1 
    # get other features 
    t_list, pos_list, l_list = entry['tokens'].split(" "),entry['pos'].split(" "),entry['lemmata'].split(" ")
    curr_spans = { # start: [end, False, span]
        '<l>': {'end_tag': '</l>','in_span': False, 'span':reinitiate_spans()}, 
        '<i>': {'end_tag': '</i>','in_span': False, 'span':reinitiate_spans()},
        '<n>': {'end_tag': '</n>','in_span': False, 'span':reinitiate_spans()},
    }
    pos_spans = {
        'np': {'span':reinitiate_spans()}
    }
    curr_s_len = 0
    for t, p, l in zip(t_list, pos_list, l_list): 
        # LOWER CASE AND STRIP HYPHENS 
        t,p,l = t.lower(), p.lower(), l.lower()
        t = re.sub("-","",t)
        l = re.sub("-","",l)
        if ".." not in p: # ignore dividers 
            if t == "</s>": # end of a sentence 
                FEATURES_DICT[key]['sentence_length'].append(curr_s_len)
                curr_s_len = 0 # reset 
            elif t != "<s>": # not inclusive of boundaries 
                curr_s_len += 1 # not including <s> tags 
                FEATURES_DICT[key] = count_feature(l,p,FEATURES_DICT[key])
                lemma_key = (l,p)
                FEATURES_DICT[key]['num_types'].update({t})
                if re.search(function,p): # function word  
                    if lemma_key not in VOCAB['function']: VOCAB['function'][lemma_key] = 0 
                    VOCAB['function'][lemma_key] += 1 
                elif l==p: # punctuation or symbol 
                    if lemma_key not in VOCAB['punctuation']: VOCAB['punctuation'][lemma_key] = 0 
                    VOCAB['punctuation'][lemma_key] += 1 
                else: 
                    if lemma_key not in VOCAB['content']: VOCAB['content'][lemma_key] = 0 
                    VOCAB['content'][lemma_key] += 1 
        for tag in curr_spans.keys(): 
            if tag == t: # start of a new span 
                curr_spans[tag]['in_span'] = True 
            elif t == curr_spans[tag]['end_tag']: # end of the span 
                # add to master dict 
                c = curr_spans[tag]['span']
                c = (c['tokens'].strip(),c['pos'].strip(),c['lemmata'].strip()) # convert list to tuple to use as key 
                if c not in SPANS[tag]: SPANS[tag][c] = 0 
                SPANS[tag][c] += 1 # keep a tally to rank by frequency later 
                curr_spans[tag]['in_span'] = False # reset flag 
                curr_spans[tag]['span'] = reinitiate_spans() # reset span 
            elif curr_spans[tag]['in_span']: # a lemma in the span
                if p in ["<s>","</s>"]: continue  
                curr_spans[tag]['span'] = update_span(curr_spans[tag]['span'], t,p,l)
        for tag in pos_spans.keys(): 
            if tag in p: # same pos tag 
                pos_spans[tag]['span'] = update_span(pos_spans[tag]['span'], t,p,l)
            elif tag not in p:  
                if p in ["<s>","</s>"]: continue 
                c = pos_spans[tag]['span']
                if len(c['tokens']) > 0: 
                    c = (c['tokens'].strip(),c['pos'].strip(),c['lemmata'].strip()) # convert list to tuple to use as key 
                    if c not in SPANS[tag]: SPANS[tag][c] = 0 
                    SPANS[tag][c] += 1 # keep a tally to rank by frequency later 
                pos_spans[tag]['span'] = reinitiate_spans() # reset tag      

    if (idx > 0) and (idx % 10000 == 0): 
        export()
export()
