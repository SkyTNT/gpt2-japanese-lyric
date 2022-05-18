import os
import pickle
import random
import time
import unicodedata
import re

import numpy
from tqdm import tqdm
from transformers import T5Tokenizer


def get_token_ids(text, tokenizer):
    text = text.strip()
    text = text.replace('\n', '\\n')
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c.isprintable())
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


if __name__ == '__main__':
    data_dir = "lyric_clean.pkl"

    tokenizer = T5Tokenizer(
        vocab_file="google_sp.model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="\\n",
        mask_token="[MASK]",
        extra_ids=0,
        additional_special_tokens=[],
        do_lower_case=True
    )
    tokenizer.sanitize_special_tokens()

    with open(data_dir, "rb") as f:
        songs = pickle.load(f)

    docs = []
    for song in tqdm(songs):
        ids = get_token_ids(song['name']+"[CLS]"+song['lyric'], tokenizer)
        if tokenizer.unk_token_id not in ids:
            docs.append(ids)
    print(len(docs))
    print(numpy.mean([len(x) for x in docs]))
    with open("lyric_ids_titled.pkl", "wb") as f:
        pickle.dump(docs, f)
