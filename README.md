# nuke_pipeline

This is a pipeline for NER in noisy domains

## Requirements

### libraries:

```bash
pip install -r requirements.txt
```
### data:

#### BTC

```bash
cd data
wget https://github.com/GateNLP/broad_twitter_corpus/archive/refs/heads/master.zip
unzip ./broad_twitter_corpus
wget http://downloads.gate.ac.uk/twitie/twitie-tagger.zip
unzip ./twitie-tagger.zip

```

#### LexNorm

## Usage

### Use normalizer on LexNorm

```bash
cd normalization

```



### Run luke on data

```bash
cd ner
python3 luke_conll2003.py path/to/conlldata -large=True
```


## Data

### BTC
### LexNorm2015
