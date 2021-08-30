# nuke_pipeline

This is a pipeline for NER in noisy domains

## Requirements

Currently, LUKE is only available on the master branch of Huggingface Transformers:

```bash
pip3 install seqeval git+https://github.com/huggingface/transformers.git
```

Remaining package requirements:


```bash
pip install -r requirements.txt
```

Once SpaCy is install, install english model:

```bash
python3 -m spacy download en_core_web_sm
```


#### Install FLITE

make sure you have `make` installed.

```bash
$ wget tts.speech.cs.cmu.edu/awb/flite-2.0.5-current.tar.bz2
$ tar xjf flite-2.0.5-current.tar.bz2
$ cd flite-2.0.5-current
$ ./configure && make
$ sudo make install
$ cd testsuite
$ make lex_lookup
$ sudo cp lex_lookup /usr/local/bin
```

When installing on MacOS and other systems that use a BSD version of cp, some modification to a Makefile must be made in order to install flite-2.0.5 (between steps 3 and 4). Edit main/Makefile and change both instances of cp -pd to cp -pR. Then resume the steps above at step 4.

#### Get BTC (already downloaded)


```bash
cd datasets
wget https://github.com/GateNLP/broad_twitter_corpus/archive/refs/heads/master.zip
```

#### Download the Lexnorm2015 Dataset (already downloaded)

```bash
cd datasets
mkdir lexnorm
cd lexnorm
wget https://github.com/noisy-text/noisy-text.github.io/raw/master/2015/files/lexnorm2015.tgz
tar -zxvf lexnorm2015.tgz
cp lexnorm2015/* .
rm -rf lexnorm2015 lexnorm2015.tgz
cd ../..
```

#### Get pretrained lexical models

i) Word-Model

already included in Repo

ii) Character-Model

download [here](https://drive.google.com/file/d/1VqWamsjJz3VRJZqoigaWkEAknd3yiT4a/view?usp=sharing) and put it in ./normalization/spelling_modell/

iii) Phonetic-Model

download [here](https://drive.google.com/file/d/1QhjXY1yd2dnTssnkriPBPNOmHgRh9VOy/view?usp=sharing) and put it in ./normalization/phon_model/

## Usage

#### Reproduce Experiments

Experiments can be reproduced using the pretrained models in the repository. However, exact reproduction is not always guaranteed. To run on GPU, add option -gpu 0

i) Run NUKE on BTC

```bash
python3 nuke.py -btc_data ./datasets/broad_twitter_corpus-master/ -logfolder -save_dir ./normalization/hybrid_model -input hybrid -eval -bos -eos -batch_size 32 -share_vocab -data_augm -large_luke True -noise_ratio 0.1 -char_model ./normalization/spelling_modell/model_50_spelling.pt -load_from=./normalization/word_model/model_50_word.pt -lowercase
```

ii) Run LUKE on BTC

```bash
python3 nuke.py -btc_data ./datasets/broad_twitter_corpus-master/ -logfolder -save_dir ./normalization/hybrid_model -input hybrid -eval -bos -eos -batch_size 32 -share_vocab -data_augm -large_luke True -noise_ratio 0.1 -char_model ./normalization/spelling_modell/model_50_spelling.pt -load_from=./normalization/word_model/model_50_word.pt -lowercase -bypass
```

iii) Run Hybrid + Phon Model on LexNorm

```bash
python3 train_normalizer.py -eval -logfolder -save_dir hybrid_model -load_from ./normalization/word_model/model_50_word.pt -char_model ./normalization/spelling_modell/model_50_spelling.pt -phonetic_model ./normalization/phon_model/model_50_phonetic.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab -phonetic_data
```

iv) Run Hybrid on LexNorm

```bash
python3 train_normalizer.py -eval -logfolder -save_dir hybrid_model -load_from ./normalization/word_model/model_50_word.pt -char_model ./normalization/spelling_modell/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab
```

v) Run Hybrid + pretrained Embs on LexNorm

```bash
python3 train_normalizer.py -eval -logfolder -save_dir hybrid_model -load_from ./normalization/word_model/model_50_word_pretrainedEmb.pt -char_model ./normalization/spelling_modell/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab -pretrained_emb True
```



#### Train normalizers

The hybrid model is a combination of two or three Seq2Seq models: a word-level one (**S2S**), a secondary character-level trained on pairs of words (spelling with noise augmented data) and a secondary phonological model trained on IPA-transliterated data

i) Train a word-level model, save results in folder `word_model` 
```bash
python3 train_normalizer.py -logfolder -save_dir word_model -gpu 0 -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
ii) Train a secondary character-level model, save results in folder `spelling_modell`
```bash
python3 train_normalizer.py -logfolder -save_dir spelling_modell -gpu 0 -input spelling -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 500 -dropout 0.5 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.001 -max_grad_norm 5 -rnn_size 500 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```

iii) Train a secondary phonological model, save results in folder `phon_model`
```bash
python3 train_normalizer.py -logfolder -save_dir phon_model -gpu 0 -input phonetic -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 500 -dropout 0.5 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.001 -max_grad_norm 5 -rnn_size 500 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```

#### Run NUKE on data

```bash
python3 nuke.py -btc_data path/to/data -logfolder -save_dir ./normalization/hybrid_model -input hybrid -eval -bos -eos -batch_size 32 -share_vocab -data_augm -large_luke True -noise_ratio 0.1 -char_model ./normalization/spelling_modell/model_50_spelling.pt -load_from=./normalization/word_model/model_50_word.pt -lowercase
```
btc_data should point to local BTC repository.
To run on GPU, add option -gpu 0.

#### Run LUKE on data

```bash
python3 nuke.py -btc_data path/to/data -logfolder -save_dir ./normalization/hybrid_model -input hybrid -eval -bos -eos -batch_size 32 -share_vocab -data_augm -large_luke True -noise_ratio 0.1 -char_model ./normalization/spelling_modell/model_50_spelling.pt -load_from=./normalization/word_model/model_50_word.pt -lowercase -bypass
```

To run on GPU, add option -gpu 0

### Evaluate the normalizer module

```bash
python3 train_normalizer.py -eval -logfolder -save_dir hybrid_model -load_from ./normalization/word_model/model_50_word.pt -char_model ./normalization/spelling_modell/model_50_spelling.pt -phonetic_model ./normalization/phon_model/model_50_phonetic.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab -phonetic_data
```

To run on GPU, add option -gpu 0

### Additional Features

(1) Train word normalizer with pretrained embeddings trained on Twitter:

```bash
python3 train_normalizer.py -logfolder -save_dir word_model -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 200 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50 -pretrained_emb True
```

(2) Download and enrich BTC with syntactic and lexical information

```bash
python3 enrich_btc.py
```

This command downloads and transforms the BTC with the TwitIE tagger and a pretrained FLAIR model. For usage requirements concerning the tagger,
refer to it's README in the twitie-tagger folder.

Depending in your machine, the lexical and syntactical enriching of the BTC MIGHT TAKE A LONG TIME

### Additional Info

- This code adapts and extends code from the paper "Adapting Sequence to Sequence models for Text Normalization in Social Media"
- find the original repo for the normalizers: https://github.com/Isminoula/TextNormSeq2Seq
- the original normalizer code is transported from Python 2 to Python 3
- the original normalizer code is not thoroughly documented
- every added code is thoroughly documented
