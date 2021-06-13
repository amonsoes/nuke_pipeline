from ner.luke_conll2003 import LukeLoader

class Nuke:
    
    def __init__(self, luke, hybrid_norm, path, split_sym):
       self.luke = luke
       self.hybrid_norm = hybrid_norm
       self.pair_generator = self.load_examples(path, split_sym)
    
    def load_examples(self, path, split_sym):
        with open(path, 'r') as f:
            tokens, labels, idx = [], [], []
            enum = 0
            for line in f:
                if not line:
                    yield dict(tokens=tokens,
                               labels=labels,
                               string=''.join(tokens),
                               idx=idx)
                else:
                    splitted_line = line.split(split_sym)
                    word = splitted_line[0]
                    tokens.append(word)
                    labels.append(splitted_line[-1])
                    idx.append(enum)
                    enum += len(word)
    
                    
    def inference(self, example):
        predictions = 
        