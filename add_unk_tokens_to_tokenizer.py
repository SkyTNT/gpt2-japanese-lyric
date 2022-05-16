import pickle
import re
import unicodedata

import sentencepiece.sentencepiece_model_pb2 as model
import sentencepiece as spm
from tqdm import tqdm

tokens_used = [False] * 32000
unk_tokens = []


def process(text):
    # \u3040-\u30ff为片假名和片假名范围
    if re.search(r"[\u3040-\u30ff]", text) is None:
        return
    text = text.strip()
    text = text.replace('\n', '\\n')
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c.isprintable())
    tokens = sp.EncodeAsPieces(text)
    token_ids = sp.EncodeAsIds(text)
    for t in zip(tokens, token_ids):
        tokens_used[t[1]] = True
        if t[1] == 0:
            unk_tokens.append(t[0])


if __name__ == '__main__':
    with open("google_sp.model", "rb") as f:
        model_data = f.read()
        m = model.ModelProto()
        m.ParseFromString(model_data)
        sp = spm.SentencePieceProcessor()
        sp.LoadFromSerializedProto(model_data)
    with open("lyric_clean.pkl", "rb") as f:
        songs = pickle.load(f)
    for songs in tqdm(songs):
        process(songs['lyric'])
    unk_tokens = list(set(unk_tokens))
    unused_ids = list(reversed([i for i in range(7, 32000) if not tokens_used[i]]))
    print(unk_tokens)
    print(unused_ids)
    num = 0
    for uid in unused_ids:
        if num >= len(unk_tokens):
            break
        print(m.pieces[uid].piece+" ---> "+unk_tokens[num])
        m.pieces[uid].piece = unk_tokens[num]
        num += 1
    unk_tokens = unk_tokens[num:]
    unused_ids = unused_ids[num:]
    print(unused_ids if len(unk_tokens) == 0 else unk_tokens)
    with open("google_sp.model", "wb") as f:
        f.write(m.SerializeToString())
