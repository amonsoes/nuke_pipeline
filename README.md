# nuke_pipeline

This is a pipeline for NER in noisy domains

## Requirements

### libraries:

```bash
pip install -r requirements.txt
```

#### BTC

```bash
python3 normalize_btc.py
```

This command downloads and transforms the BTC and the TwitIE tagger. For usage requirements concerning the tagger,
refer to it's README in the twitie-tagger folder.

Depending in your machine, the lexical and syntactical enriching of the BTC MIGHT TAKE A LONG TIME

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
