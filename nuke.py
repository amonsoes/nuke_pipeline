from ner.luke_conll2003 import LukeLoader
from normalization.HybridSeq2Seq import HybridSeq2Seq
from normalization.parameters import change_args, parser
from metrics.metric import NukeMetric
import os


class Nuke:
    
    '''loads instance of NUKE with a hybrid normalizer and NER-model LUKE
    Init accepts opt from arguments to pass CLI args to the constructors. 
    See normalization/parameters.py for opt
    '''
    
    def __init__(self, opt):
        opt.is_nuke = True
        self.hybrid_norm = HybridSeq2Seq(opt)
        self.luke = LukeLoader(opt)
                  
    def inference(self, example):
        example['tokens'] = self.hybrid_norm(example['tokens'])[2][0]
        example['ner_prediction'] = self.luke.inference(example)
        return example

    def bypass_inference(self, example):
        example['ner_prediction'] = self.luke.inference(example)
        return example


class NukeEvaluator:
    
    ''' evaluation class for NUKE, accepts a path to the data, 
    an instance of class NUKE and a metric
    '''
    
    def __init__(self, opt, path, nuke, metric):
        self.opt = opt
        self.nuke = nuke
        self.metric = metric
        self.example_generator = self.load_examples(path, opt.btc_split_sym)
    
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

    
    def process_examples(self, bypass=False):
        if bypass:
            for example in self.example_generator:
                yield self.nuke.bypass_inference(example)
        else:
            for example in self.example_generator:
                yield self.nuke.inference(example)
    
    def get_nuke_scores(self):
        for example in self.process_examples():
            try:
                self.metric(example)
            except:
                continue
        self.metric.build_scores()
        return self.metric.get_scores()
    
    def get_luke_scores(self):
        for example in self.process_examples(bypass=True):
            self.metric(example)
        self.metric.build_scores()
        return self.metric.get_scores()
    
    def to_file(self, path='./results.txt'):
        with open(path, 'w') as w:
            w.write('RESULTS\n\n')
            w.write(f'MACRO-F1 : {self.metric.macro_f1}\n')
            w.write(f'ACCURACY : {self.metric.accuracy}\n')
            for k in self.metric.classes:
                w.write(f'\nclass {k}:\n\n')
                w.write(f'F1: {self.metric.classes[k].f1}\n')
                w.write(f'PRECISION: {self.metric.classes[k].precision}\n')
                w.write(f'RECALL: {self.metric.classes[k].recall}\n')
    
def process_btc(opt):
    nuke = Nuke(opt)
    opt.is_inference = True
    for _,_, files in os.walk('./datasets/broad_twitter_corpus-master'):
        for file in files:
            if file.endswith('.conll'):
                path = './datasets/broad_twitter_corpus-master/' + file
                print(f'processing {path}...\n')
                metric = NukeMetric()
                evaluator = NukeEvaluator(opt, path, nuke, metric)
                if opt.bypass:
                    scores = evaluator.get_luke_scores()
                    evaluator.to_file(path+'_results_bypassed.txt')
                else:
                    scores = evaluator.get_nuke_scores()
                    evaluator.to_file(path+'_results.txt')
    return scores
                 
if __name__ == '__main__':
    
    parser.add_argument('-btc_data', type=str, help='path to btc data')
    parser.add_argument('-large_luke', type=lambda x: x in ['true', 'True', '1', 'yes'], default=False,help='decide if you want to transfer large model')
    parser.add_argument('-btc_split_sym', type=str, default='\t', help='operator to split btc data')
    opt = parser.parse_args()
    opt = change_args(opt)
    
    process_btc(opt)
