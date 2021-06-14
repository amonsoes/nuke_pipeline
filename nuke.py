from ner.luke_conll2003 import LukeLoader
from normalization.HybridSeq2Seq import HybridSeq2Seq
from normalization.parameters import change_args, parser

class Nuke:
    
    def __init__(self, opt):
        #self.hybrid_norm = HybridSeq2Seq(opt)
        self.luke = LukeLoader(opt)
        self.pair_generator = self.load_examples(opt.btc_data, opt.btc_split_sym)
    
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
        
        words_preds_labels = self.luke.inference(example)

if __name__ == '__main__':
    
    # add NUKE and LUKE parameters to imported normalization parameters
    parser.add_argument('-btc_data', type=str, help='path to btc data')
    parser.add_argument('-large_luke', type=lambda x: x in ['true', 'True', '1', 'yes'], default=False,help='decide if you want to transfer large model')
    parser.add_argument('-btc_split_sym', type=str, default='\t', help='operator to split btc data')
    opt = parser.parse_args()
    opt = change_args(opt)
    
    nuke = Nuke(opt)
    nuke.luke.inference_raw('Millions of family-run #farms hold the key to global #hunger reveals #UN report . http://t.co/9JKaxcMKJ0')
    for example in nuke.pair_generator:
        nuke.luke.inference(example)