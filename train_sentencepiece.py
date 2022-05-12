import io
import os
import pickle
import unicodedata
import sentencepiece as spm
import tqdm

if __name__ == '__main__':
    inputs = []

    with open("lyric_raw.pkl", "rb") as f:
        songs = pickle.load(f)

    for song in tqdm.tqdm(songs):
        lyric = unicodedata.normalize("NFKC", song['lyric'])
        lyric = "".join(c for c in lyric if c == '\n' or c.isprintable())
        inputs.append(lyric)

    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(inputs), normalization_rule_name='identity', model_writer=model, vocab_size=32000,
        character_coverage=1.0, user_defined_symbols=['[PAD]', '[CLS]', '[SEP]', '[MASK]'])

    # Serialize the model as file.
    with open('google_sp.model', 'wb') as f:
        f.write(model.getvalue())

    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    print('id of return=', sp.piece_to_id('\n'))
    print(sp.encode('未来に揺れる花 過去にもあった花\n形は違う同じ花', out_type=str))
    print(sp.encode('未来に揺れる花 過去にもあった花\n形は違う同じ花', out_type=int))
