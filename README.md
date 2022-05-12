# gpt2-japanese-lyric

This project contains training scripts and dataset of [lyric generator](https://lyric.fab.moe/)

## model

The models are here:

[skytnt/gpt2-japanese-lyric-small](https://huggingface.co/skytnt/gpt2-japanese-lyric-small)

[skytnt/gpt2-japanese-lyric-xsmall](https://huggingface.co/skytnt/gpt2-japanese-lyric-xsmall)

## dataset

The lyrics are collected from [uta-net](https://www.uta-net.com/) by [lyric_download](https://github.com/SkyTNT/lyric_downlowd)

- lyric_raw.pkl: 300,000 lyrics without any process
- lyric_clean.pkl: 143,587 lyrics without English characters, and remove parenthesized pronunciations in the lyrics.
- lyric_ids.pkl: token ids of lyric_clean.pkl

## training

My models are fine-tuned models from [rinnakk/japanese-pretrained-models](https://github.com/rinnakk/japanese-pretrained-models).
However, you can start training from scratch.
