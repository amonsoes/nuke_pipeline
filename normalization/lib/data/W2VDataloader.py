import torchtext
import torch
import csv
import json
import re

from torchtext.data.field import Field

from .constants import *
from collections import defaultdict

class BatchWrapper:
    
    # turn dataset splits into iterators
    
    def __init__(self, dl: torchtext.data.BucketIterator, x_var: str, y_vars: str, ids: int, id_text_dict):
        self.dl, self.x_var, self.y_vars, self.ids = dl, x_var, y_vars, ids
        self.id_text = id_text_dict

    def __iter__(self):
        for batch in self.dl:
            dict_batch = {}
            dict_batch['src_sent_words'] = []
            dict_batch['tgt_sent_words'] = []

            for i in batch.id:
                src_words, tgt_words = self.id_text[str(i.item())]
                dict_batch['src_sent_words'].append(src_words)
                dict_batch['tgt_sent_words'].append(tgt_words)

            src = getattr(batch, self.x_var)
            tgt = getattr(batch, self.y_vars)
            id = getattr(batch, self.ids)
            src_len = torch.tensor([len([i for i in x if i != 1]) for x in src.T])
            tgt_len = torch.tensor([len([i for i in x if i != 1]) for x in tgt.T])
            dict_batch['src'] = (src, src_len)
            dict_batch['tgt'] = (tgt, tgt_len)
            dict_batch['id'] = id
            yield dict_batch

    def __len__(self):
        return len(self.dl)


class Preprocessor:
    def __init__(self):
        self.tokens = []
        self.positions = []

    def lowercase(self):
        self.tokens = [x.lower() for x in self.tokens]
        return

    def isUrl(self, token):
        regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        match = re.match(regex, token)
        if match is None:
            return False
        else:
            return True

    def filter(self):
        filtered = []
        for pos, token in enumerate(self.tokens):
            if self.isUrl(token):
                filtered.append('<url>')
            elif token.startswith('#'):
                filtered.append('<hash>')
            elif token.startswith('@'):
                filtered.append('<mention>')
            else:
                filtered.append(token)
                self.positions.append(pos)
        self.tokens = filtered
        return

    def run(self, tokens, lowercase=False):
        self.tokens = tokens
        self.positions = []
        if(lowercase):
            self.lowercase()
        self.filter()
        return self.tokens, self.positions


class JSONTransformer:
    
    def __init__(self, datapath):
        self.path = datapath
        self.fields = self.retrieve_fields_from_json()
    
    def to_tsv(self, path):
        with open(self.path, 'r') as f:
            with open(path, 'wt') as w:
                dump = json.load(f)
                tsv_writer = csv.writer(w, delimiter='\t')
                tsv_writer.writerow([i for i in self.fields])
                for js in dump:
                    tsv_writer.writerow([js['tid'], js['index'], js['output'], js['input']])

    def test_to_tsv(self, path):
        with open(self.path, 'r') as f:
            with open(path, 'wt') as w:
                dump = json.load(f)
                tsv_writer = csv.writer(w, delimiter='\t')
                tsv_writer.writerow([i for i in self.fields])
                for js in dump:
                    tsv_writer.writerow([js['tid'], js['index'], js['input']])
    
    def to_torchtext_JSON(self, path):
        processor = Preprocessor()
        with open(self.path, 'r') as f:
            with open(path, 'wt') as w:
                dump = json.load(f)
                for js in dump:
                    js['input'] = processor.run(js['input'])[0]
                    js['output'] = processor.run(js['output'])[0]
                    json.dump(js, w)
                    w.write('\n')
    
    
    def retrieve_fields_from_json(self):
        with open (self.path, 'r') as f:
            dump = json.load(f)
            return dump[0].keys()


class W2VDataLoader:
    
    def __init__(self, path, train, dev, shared_vocab, bos_eos, lowercase, batch_size, gpu, valsplit):
        self.device = 'cuda:0' if gpu and torch.cuda.is_available() else 'cpu'
        # TODO: make bos and eos appearance separable
        self.mappings = defaultdict(set)
        self.ID = torchtext.data.Field(sequential=False, use_vocab=False)
        if bos_eos:
            self.SRC = torchtext.data.Field(sequential=True, use_vocab=True, init_token=BOS_WORD, eos_token=EOS_WORD, lower=lowercase)
            self.TGT = torchtext.data.Field(sequential=True, use_vocab=True, init_token=BOS_WORD, eos_token=EOS_WORD)
        else:
            self.SRC = torchtext.data.Field(sequential=True, use_vocab=True)
            self.TGT = torchtext.data.Field(sequential=True, use_vocab=True)  
        fields = { 'output':('tgt', self.TGT), 'input': ('src', self.SRC), 'tid' : ('id', self.ID)}
        train, test = torchtext.data.TabularDataset.splits(path=path,
                                                     train=train,
                                                     validation=dev,
                                                     format='json',
                                                     fields=fields)
        train, valid = train.split(split_ratio=valsplit)
        if shared_vocab:
            self.SRC.build_vocab(train, valid, vectors='glove.twitter.27B.200d')
            self.TGT.vocab = self.SRC.vocab
        else:
            self.SRC.build_vocab(train, valid, vectors='glove.twitter.27B.200d')
            self.TGT.build_vocab(train)
        
        
        self.extract_mapping(train, valid)
        train_iter = torchtext.data.BucketIterator(train, batch_size, sort_within_batch=True,sort=False, train=True, shuffle=True, device=self.device, sort_key=lambda x: len(x.src))
        dev_iter = torchtext.data.BucketIterator(valid, batch_size, sort_within_batch=True, train=True, shuffle=True, device=self.device, sort_key=lambda x: len(x.src))
        test_iter = torchtext.data.BucketIterator(test, batch_size, device=self.device, sort_key=lambda x: len(x.src))
        #get id:text dicts
        train_id_text = {example.id:(example.src, example.tgt) for example in train_iter.dataset}
        dev_id_text = {example.id:(example.src, example.tgt) for example in dev_iter.dataset}
        test_id_text = {example.id:(example.src, example.tgt) for example in test_iter.dataset}
        
        id_text = {**train_id_text, **dev_id_text, **test_id_text}
        self.train_iter = BatchWrapper(train_iter, "src", "tgt", 'id', id_text)
        self.dev_iter = BatchWrapper(dev_iter, "src", "tgt", 'id', id_text)
        self.test_iter = BatchWrapper(test_iter, "src", "tgt", 'id', id_text)
        print('\nJSONDataLoader initialized\n')
    
    def extract_mapping(self, train, dev):
        for dataset in [train, dev]:
            for example in dataset:
                for src, tgt in zip(example.src, example.tgt):
                    self.mappings[src].add(tgt)
    
    def return_iterators(self):
        return self.train_iter, self.dev_iter, self.test_iter


if __name__ == '__main__':
    
    #trm = JSONTransformer('/Users/amonsoares/TextNormSeq2Seq/dataset/test_truth.json')
    #trm.to_torchtext_JSON('./dev.json')
    
    path = '/Users/amonsoares/TextNormSeq2Seq/dataset/json_data/'
    train = 'train_data.json'
    dev = 'test_truth.json'
    
    data = W2VDataLoader(path=path,
                         train=train,
                         dev=dev,
                         shared_vocab=False,
                         bos_eos=True,
                         lowercase=False,
                         batch_size=32,
                         gpu=False,
                         valsplit=0.9)


    