
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import spacy
from spacy.symbols import ORTH

mask_token = "<mask>"
spacy_en = spacy.load("en")
spacy_en.tokenizer.add_special_case(mask_token, [{ORTH: "<mask>"}])

def get_POS(data):
    sentences = [spacy_en(text) for text in tqdm(data, desc="Loading dataset")]
    POS_map = {}
    for sentence in sentences:
        for word in sentence:
            pos_tag = word.pos_
            if pos_tag not in POS_map:
                POS_map[pos_tag] = []
            if word.text.lower() not in POS_map[pos_tag]:
                POS_map[pos_tag].append(word.text.lower())
    return POS_map


def augment_sentence(sentence, POS_map, mask_prob=0.1, pos_prob=0.2, ngram_prob=0.25, max_ngrams=3):
    augmented_sentence = []
    begin=False
    for word in sentence:
        if word.text in[ 'History', 'Diagnosis']:
            begin=True
            augmented_sentence.append(word.text)
            continue

        if not begin:
            augmented_sentence.append(word.text)
            continue

        u = np.random.uniform()
        if u < mask_prob:
            augmented_sentence.append(mask_token)
        elif u < pos_prob:
            same_pos = POS_map[word.pos_]
            augmented_sentence.append(np.random.choice(same_pos))
        else:
            augmented_sentence.append(word.text)

    if len(augmented_sentence) > 2 and np.random.uniform() < ngram_prob:
        n = min(np.random.choice(range(1, max_ngrams + 1)), len(augmented_sentence) - 1)
        start = np.random.choice(len(augmented_sentence) - n)
        for idx in range(start, start + n):
            augmented_sentence[idx] = mask_token
    return ' '.join(augmented_sentence)


def augment(g, data,POS_map, max_augmented, n_iter=30, mask_prob=0.1, pos_prob=0.1, ngram_prob=0.25, max_ngrams=3):

    sentences = [spacy_en(text) for text in tqdm(data['text'], desc="Loading {} from dataset".format(g))]

    augmented = []
    for i, sentence in enumerate(tqdm(sentences, "augmenting "+g)):
        for _ in range(n_iter):
            new_sample = augment_sentence(sentence, POS_map, mask_prob, pos_prob, ngram_prob, max_ngrams)
            if new_sample not in augmented:
                augmented.append(new_sample)

    print(g,augmented.__len__() , max_augmented)
    if max_augmented>0 and augmented.__len__() >max_augmented:
        augmented=random.sample(augmented,max_augmented)

    return pd.DataFrame(augmented)


