from ner.luke_conll2003 import LukeLoader
from normalization.HybridSeq2Seq import HybridSeq2Seq
from normalization.parameters import change_args, parser
from metrics.metric import NukeMetric


class Nuke:
    
    def __init__(self, opt):
        self.hybrid_norm = HybridSeq2Seq(opt)
        self.luke = LukeLoader(opt)
        self.example_generator = self.load_examples(opt.btc_data, opt.btc_split_sym)
    
    def load_examples(self, path, split_sym):
        with open(path, 'r') as f:
            tokens, labels, idx = [], [], []
            enum = 0
            for line in f:
                if line == '\n':
                    yield dict(tokens=tokens,
                               labels=labels,
                               string=' '.join(tokens),
                               idx=idx)
                    tokens, labels, idx = [], [], []
                    enum = 0
                else:
                    splitted_line = line.strip().split(split_sym)
                    word = splitted_line[0]
                    tokens.append(word)
                    labels.append(splitted_line[-1])
                    idx.append(enum)
                    enum += (len(word) + 1)
                  
    def inference(self, example):
        example['tokens'] = self.hybrid_norm(example['tokens'])[2][1]
        example['ner_prediction'] = self.luke.inference(example)
        return example

    def bypass_inference(self, example):
        example['ner_prediction'] = self.luke.inference(example)
        return example
    
    def process_examples(self, bypass=False):
        if bypass:
            for example in self.example_generator:
                yield self.bypass_inference(example)
        else:
            for example in self.example_generator:
                yield self.inference(example) 

class NukeEvaluator:
    
    def __init__(self, nuke, metric):
        self.nuke = nuke
        self.metric = metric
    
    def get_nuke_scores(self):
        for example in self.nuke.process_examples():
            self.metric(example)
            self.metric.build_scores()
        return self.metric.get_scores()
    
    def get_luke_scores(self):
        num = 0
        for example in self.nuke.process_examples(bypass=True):
            num += 1
            self.metric(example)
            if num > 5:
                break
        self.metric.build_scores()
        return self.metric.get_scores()
    
    def to_file(self, path='./results.txt'):
        with open(path, 'w') as w:
            w.write('RESULTS\n\n')
            w.write(f'MACRO-F1 : {self.metric.macro_f1}\n')
            w.write(f'ACCURACY : {self.metric.accuracy}\n')
            for k in self.metric.seen_labels:
                w.write(f'\nclass {k}:\n\n')
                w.write(f'F1: {self.metric.classes[k].f1}\n')
                w.write(f'PRECISION: {self.metric.classes[k].precision}\n')
                w.write(f'RECALL: {self.metric.classes[k].recall}\n')
    
        

if __name__ == '__main__':
    
    # add NUKE and LUKE parameters to imported normalization parameters
    parser.add_argument('-btc_data', type=str, help='path to btc data')
    parser.add_argument('-large_luke', type=lambda x: x in ['true', 'True', '1', 'yes'], default=False,help='decide if you want to transfer large model')
    parser.add_argument('-btc_split_sym', type=str, default='\t', help='operator to split btc data')
    opt = parser.parse_args()
    opt = change_args(opt)
    
    nuke = Nuke(opt)
    metric = NukeMetric(['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG'])
    
    evaluation = NukeEvaluator(nuke, metric)
    scores = evaluation.get_luke_scores()
    evaluation.to_file()
