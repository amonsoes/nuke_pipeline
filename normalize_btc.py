import os
import flair

from flair.models import SequenceTagger

class BTCSentence:
    
    def __init__(self):
        self.labels = []
    
    def generate_string(self):
        string = " ".join([word for word,label in self.labels])
        return string
    

class BTCData:
    
    def __init__(self, dir):
        self.dir = dir if dir.endswith('/') else dir+'/'
        self.file_generator = self.get_files('.conll')
        self.chunk_parser = SequenceTagger.load("flair/chunk-english")
        self.tokenizer = lambda x: x.split()
    
    def process_dir(self):
        if not os.path.exists('./intermediate/'):
            os.mkdir('./intermediate/')
        savedir = self.dir+'intermediate/'
        for file in self.file_generator:
            orig_path = self.dir + file
            target_path_sents = savedir + file + '.txt'
            target_path_labels = savedir + file + '_labels.txt'
            self.sents_to_file(orig_path, target_path_sents, target_path_labels)
    
    def sent_generator(self, path):
        with open(path, 'r') as f:
            sentence = BTCSentence()
            for line in f:
                if line == '\n' or line.strip() == '':
                    sentence.generate_string()
                    yield sentence
                    sentence = BTCSentence()
                else:
                    word, label = line.split('\t')
                    sentence.labels.append((word, label.strip()))
    
    def get_files(self, suffix):
        for _, _, files in os.walk(self.dir):
            for file in files:
                if file.endswith(suffix):
                    yield file
    
    def get_file(self, suffix, instance):
        for _, _, files in os.walk(self.dir):
            for file in files:
                if file.endswith(suffix) and file.startswith(instance):
                    return file
    
    def sents_to_file(self, orig_path, path, labelpath):
        with open(path, 'w') as w:
            with open(labelpath, 'w') as y:
                for sent in self.sent_generator(orig_path):
                    w.write(sent.generate_string())
                    w.write('\n')
                    y.write('\t'.join([label for _, label in sent.labels]))
                    y.write('\n')
    
    def word_pos_to_split(self, word_pos):
        rev = word_pos[::-1]
        rev_ind = rev.find('_')
        true_split = (len(word_pos)-1) - rev_ind
        word, pos = word_pos[:true_split], word_pos[true_split+1:]
        return (word, pos)
        
        
    
    def tagged_to_conll(self, path, labelpath, targetpath):
        enum = 0
        with open(path, 'r') as f:
            with open(labelpath, 'r') as j:
                with open(targetpath, 'w') as w:
                    for sent, labels in zip(f,j):
                        try:
                            enum += 1
                            word_pos_tuples = [self.word_pos_to_split(word_pos) for word_pos in sent.split()]
                            sentence = []
                            for word, _ in word_pos_tuples:
                                sentence.append(word)
                            sentence = " ".join(sentence)
                            #sentence = " ".join([word for word, _ in word_pos_tuples])
                            flair_sent = flair.data.Sentence(sentence, use_tokenizer=self.tokenizer)
                            self.chunk_parser.predict(flair_sent)
                            word_chunk_tuples = self.chunks_to_tups(flair_sent)
                            labels = labels.split('\t')
                            for (word, pos, chunk), label in zip(self.merge_tuple_lists(word_pos_tuples, word_chunk_tuples), labels):
                                w.write(f'{word} {pos} {chunk} {label}\n')
                            w.write('\n')
                        except:
                            print(enum)
                            continue
    
    def conll_pos_dataset(self):
        os.chdir('./intermediate')
        for sent_file in self.get_files('txt_tagged.txt'):
            instance = sent_file[0]
            label_file = self.get_file('_labels.txt', instance)
            sent_path = self.dir + sent_file
            label_path = self.dir + label_file
            target_path = self.dir + sent_file[:-14] + 'pos_ext.conll'
            self.tagged_to_conll(sent_path, label_path, target_path)
    
    def chunks_to_tups(self, flair_sent):
        sent = []
        for entity in flair_sent.get_spans('np'):
            type = entity.get_labels()[0].value
            for e, token in enumerate(entity.tokens):
                if e == 0:
                    sent.append((token.text, 'B-' + type))
                else:
                    sent.append((token.text, 'I-' + type))
        return sent
    
    def merge_tuple_lists(self, word_pos, word_chunk):
        words = [word for word,_ in word_chunk]
        pos = [pos for _, pos in word_pos]
        chunks = [chunk for _,chunk in word_chunk]
        return zip(words, pos, chunks)
    
    def append_pos(self):
        save_dir='./broad_twitter_corpus_master/intermediate/'
        for file in self.get_files('.conll.txt'):
            loadpath = self.dir + file
            targetpath = self.dir + file + '_tagged.txt'
            os.system(f'java -jar twitie_tag.jar models/gate-EN-twitter.model {loadpath} > {targetpath}')
            
        
        

if __name__ == '__main__':
    os.system('cd data')
    
    print('\nINFO: downloading BTC and twitie-tagger to initialize processing... \n')
    os.system('wget https://github.com/GateNLP/broad_twitter_corpus/archive/refs/heads/master.zip')
    os.system('unzip ./master.zip')
    os.system('rm master.zip')
    os.system('wget http://downloads.gate.ac.uk/twitie/twitie-tagger.zip')
    os.system('unzip ./twitie-tagger.zip')
    os.system('rm twitie-tagger.zip')
    print('\ndone\n')
    
    print('\nINFO: using tagger to create intermediate tagged files and labels... \n')
    data = BTCData('./broad_twitter_corpus_master/')
    data.process_dir()
    try:
        BTCData.append_pos()
    except:
        raise SystemError('Using java command to use twitie tagger failed. Check your JAVA distribution and refer to the README of the twitie tagger')
    
    

    
    

    #path = '/Users/amonsoares/broad_twitter_corpus/intermediate/a.conll.txt_tagged.txt'
    #labelpath = '/Users/amonsoares/broad_twitter_corpus/intermediate/a.conll_labels.txt'
    #target = '/Users/amonsoares/broad_twitter_corpus/intermediate/a_ext.conll'
    
    data.conll_pos_dataset()
    print('done')
    
    
    
        
        