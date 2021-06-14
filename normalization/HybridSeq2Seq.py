from normalization.parameters import parser, change_args
from normalization.lib.data.Tweet import Tweet
from normalization.lib.data.DataLoader import create_data
from normalization.lib.data.W2VDataloader import W2VDataLoader
from normalization.lib.data.PhonTransliterator import PhonTransliterator
import logging
import os
import copy
import normalization.lib as lib

class HybridSeq2Seq:
    
    def __init__(self, opt):
        self.opt = opt
        self.char_model, self.char_optim = self.load_char_model(opt)
        self.word_model, self.word_optim = self.load_word_model(opt)
        if opt.phonetic_model:
            self.phon_model, self.phon_optim = self.load_phon_model(opt)
        else:
            self.phon_model = None
        self.evaluator =lib.train.Evaluator(self.word_model, opt, self.char_model, self.phon_model)
    
    def __call__(self, tokenlist):
        tweets = [Tweet(tokenlist, tokenlist, '1', '1') for i in range(2)]
        test_data, test_vocab, mappings = create_data(tweets, opt=self.opt, vocab=self.vocab, mappings=self.mappings)
        prediction = self.evaluator.eval(test_data)
        return prediction
        
        
    def load_word_model(self, opt):
        print('\nloading word model...\n')
        if not opt.load_complete_model:
            if opt.pretrained_emb:
                dloader = W2VDataLoader(path=opt.datapath,
                                        train=opt.traindata,
                                        dev=opt.testdata,
                                        bos_eos=opt.bos,
                                        lowercase=opt.lowercase,
                                        shared_vocab=opt.share_vocab,
                                        batch_size=opt.batch_size,
                                        gpu=opt.gpu,
                                        valsplit=opt.valsplit)
                train_data, valid_data, test_data = dloader.return_iterators()
                vocab = {'src': dloader.SRC.vocab, 'tgt': dloader.TGT.vocab}
                self.mappings = dloader.mappings
            else:
                train_data, valid_data, test_data, self.vocab, self.mappings = lib.data.create_datasets(opt)
            model, optim = lib.model.create_model((self.vocab['src'], self.vocab['tgt']), opt)
            print('Loading test data from "%s"' % opt.testdata)
            print('Loading training data from "%s"' % opt.traindata)
            print(' * Vocabulary size. source = %d; target = %d' % (len(self.vocab['src']), len(self.vocab['tgt'])))
            print(' * Maximum batch size. %d' % opt.batch_size)
        else:
            model = lib.model.create_model_from_complete(opt, 'word')
            optim = None
        print(model)
        return model, optim
    
    def load_char_model(self, opt):
        print('\nloading char model...\n')
        opt = copy.deepcopy(opt)
        opt.input = 'spelling'
        if not opt.load_complete_model:
            train_data, valid_data, test_data, vocab, mappings = lib.data.create_datasets(opt)
            char_model, char_optim = lib.model.create_model((vocab['src'], vocab['tgt']), opt, is_char_model = True)
            print(char_model.opt)
            print('Loading test data for character model from "%s"' % opt.testdata)
            print('Loading training data for character model from "%s"' % opt.traindata)
            print(' * Character model vocabulary size. source = %d; target = %d' % (len(vocab['src']), len(vocab['tgt'])))
            print(' * Character model maximum batch size. %d' % opt.batch_size)
        else:
            char_model = lib.create_model_from_complete(opt, is_char_model=True)
            char_optim = None
        print(char_model)
        return char_model, char_optim
    
    def load_phon_model(self, opt):
        if not opt.phonetic_model:
            return None
        print('\nloading phonetic model...\n')
        opt = copy.deepcopy(opt)
        opt.input = 'phonetic'
        if not opt.load_complete_model:
            opt.traindata = 'phonetic_data/' + opt.traindata
            opt.testdata = 'phonetic_data/' + opt.testdata
            print('\n creating data for phonetic model. If no phonetic data is provided, this can take a while...\n')
            train_data, valid_data, test_data, vocab, mappings = lib.data.create_datasets(opt)
            phon_model, phon_optim = lib.model.create_model((vocab['src'], vocab['tgt']), opt, is_phon_model = True)
            print(phon_model.opt)
            print('Loading test data for phonetic model from "%s"' % opt.testdata)
            print('Loading training data for phonetic model from "%s"' % opt.traindata)
            print(' * Phonetic model vocabulary size. source = %d; target = %d' % (len(vocab['src']), len(vocab['tgt'])))
            print(' * Phonetic model maximum batch size. %d' % opt.batch_size)
        else:
            phon_model = lib.create_model_from_complete(opt, is_phon_model=True)
            phon_optim = None
        phon_model.transliterator = PhonTransliterator()
        print(phon_model)
        return phon_model, phon_optim
        
if __name__ == '__main__':
    opt = parser.parse_args()
    opt = change_args(opt)
    normalizer = HybridSeq2Seq(opt)
    prediction = normalizer(['Ight', 'imma', 'head', 'out'])
    print('donezies')