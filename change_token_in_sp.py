import sentencepiece.sentencepiece_model_pb2 as model

if __name__ == '__main__':
    m = model.ModelProto()
    with open("google_sp.model", "rb") as f:
        m.ParseFromString(f.read())
    for p in m.pieces:
        if p.piece == "[SEP]":
            p.piece = "\\n"
            p.type = 1
    with open("google_sp.model", "wb") as f:
        f.write(m.SerializeToString())
        