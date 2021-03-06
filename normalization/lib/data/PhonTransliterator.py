import epitran
import json

class PhonTransliterator:
    
    """posesses an instance of epitran to transliterate string to API
    """
    
    def __init__(self, preproc=True, postproc=True):
        self.epi = epitran.Epitran('eng-Latn', preproc=preproc, postproc=postproc)
    
    def __call__(self, string):
        return self.epi.transliterate(string)
    
    def phonetic_dataset(self, inpath, outpath):
        """transforms an input dataset with character sequences to dataset with API sequences 
        """
        with open(inpath, 'r') as f:
            with open(outpath, 'w') as w:
                dump = json.load(f)
                target_dump = []
                for js in dump:
                   js['input'] = [self(x) for x in js['input']]
                   target_dump.append(js)
                json.dump(target_dump , w)
                   

if __name__ == '__main__':
    pass