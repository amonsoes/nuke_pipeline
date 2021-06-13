# Extended Stacked TextNorm Encoder-Decoder

### Download and Transform the Lexnorm2015 Dataset
```bash
mkdir dataset
cd dataset
wget https://github.com/noisy-text/noisy-text.github.io/raw/master/2015/files/lexnorm2015.tgz
tar -zxvf lexnorm2015.tgz
cp lexnorm2015/* .
rm -rf lexnorm2015 lexnorm2015.tgz
cd ..
```

### Epitran add-on "FLITE" Installation

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

### Training a hybrid Seq2Seq model from scratch 
The hybrid model is a combination of two or threee Seq2Seq models: a word-level one (**S2S**), a secondary character-level trained on pairs of words (spelling with noise augmented data) and a secondary phonological model trained on IPA-transliterated data

i) Train a word-level model, save results in folder `word_model` 
```bash
python main.py -logfolder -save_dir word_model -gpu 0 -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50 -pretrained_embs
```
ii) Train a secondary character-level model, save results in folder `spelling_model`
```bash
python main.py -logfolder -save_dir spelling_model -gpu 0 -input spelling -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 500 -dropout 0.5 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.001 -max_grad_norm 5 -rnn_size 500 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```

iii) Train a secondary phonological model, save results in folder `phon_model`
```bash
python main.py -logfolder -save_dir phon_model -gpu 0 -input phonetic -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 500 -dropout 0.5 -emb_size 256 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.001 -max_grad_norm 5 -rnn_size 500 -rnn_type 'LSTM'  -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 30 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```


### Test hybrid Seq2Seq model
Evaluate final model (**HS2S**) combining the trained models described above:
```bash
python main.py -eval -logfolder -save_dir hybrid_model -gpu 0 -load_from word_model/model_50_word.pt -char_model spelling_model/model_50_spelling.pt -phonetic_model phon_model/model_50_word.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab -pretrained_embs
```

IMPORTANT NOTE: If you use a pretrained model with the suffix `_w2v`, make sure to set `-pretrained_embs` in the CLI, Otherwise this will result in an embedding_size error while loading the model.

 
### Pretrained models - Reproducibility
We have done our best to ensure reproducibility of our results, however this is not always [guaranteed](https://pytorch.org/docs/stable/notes/randomness.html).
As an extra reproducibility step, we also release our best performing models. Just unzip the `pretrained_models.zip` found [here](https://uofi.box.com/v/TextNormSeq2SeqModels) and try them by setting the flag `eval` and updating the model folders in your python commands, for example:
- Pre-trained hybrid model (**HS2S**):
```bash
python main.py -eval -logfolder -save_dir hybrid_model -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt -char_model pretrained_models/spelling_model/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -lowercase -bos -eos -batch_size 32 -share_vocab
``` 
- Pre-trained word-level model (**S2S**):
```bash
python main.py -eval -logfolder -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt  -input word -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -dropout 0.5 -emb_size 100 -end_epoch 50 -layers 3 -learning_rate_decay 0.05 -lr 0.01 -max_grad_norm 5 -rnn_size 200 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab -start_decay_after 15 -teacher_forcing_ratio 0.6  -max_train_decode_len 50
```
A separate file `pretrained_models/run.sh` contains all commands for reproducing the aforementioned models.

### Interactive mode
With the `interactive` flag, you can also try the model on any arbitrary text through command line, for example:
```console
foo@bar:~$ python main.py -interactive -gpu 0 -load_from pretrained_models/word_model/model_50_word.pt -char_model pretrained_models/spelling_model/model_50_spelling.pt -input hybrid -data_augm -noise_ratio 0.1 -attention -bias -lowercase -bos -eos -brnn -batch_size 32 -rnn_type 'LSTM' -tie_decoder_embeddings -share_embeddings -share_vocab
Please enter the text to be normalized (q to quit): lol how are u doin
Prediction is:laughing out loud how are you doing
Please enter the text to be normalized (q to quit): q 
foo@bar:~$
```

#### Notes
If you wish to work on CPU, simply remove the flag `-gpu 0` from the following commands.

Each command prints in a file named `output.log` saved in defined by `save_dir`. 
Remove the flag `-logfolder` to output to console. 

# TextNormalization with Seq2Seq 

This code is largly adapted from the repo for the AAAI ICWSM 2019 paper "Adapting Sequence to Sequence models for Text Normalization in Social Media", 