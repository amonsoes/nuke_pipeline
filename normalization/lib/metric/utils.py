import torch.nn.functional as F
from torch.autograd import Variable
import normalization.lib as lib
import functools
import torch
import logging

logger = logging.getLogger("model")


def clean_sentence(sent, remove_unk=False, remove_eos=True, remove_bos=True):
    if lib.data.constants.EOS_WORD in sent:
        sent = sent[:sent.index(lib.data.constants.EOS_WORD) + 1]
    if remove_unk:
        sent = filter(lambda x: x != lib.data.constants.UNK_WORD, sent)
    if remove_eos:
        if len(sent) > 0 and sent[-1] == lib.data.constants.EOS_WORD:
            sent = sent[:-1]
    if remove_bos:
        if len(sent) > 0 and sent[0] == lib.data.constants.BOS_WORD:
            sent = sent[1:]
    return sent


def handle_tags(input_words, pred_words):
    assert len(input_words) == len(pred_words)
    ret = []
    for input_tokens, pred_tokens in zip(input_words, pred_words):
        if lib.data.constants.URL in pred_tokens or lib.data.constants.MENTION in pred_tokens:
            sent_length = min(len(input_tokens),len(pred_tokens))
            for i in range(sent_length):
                if(pred_tokens[i] == lib.data.constants.URL or pred_tokens[i] == lib.data.constants.MENTION):
                    pred_tokens[i] = input_tokens[i]
        ret.append(pred_tokens)
    return ret

"""
def handle_tags(input_words, pred_words):
    assert len(input_words) == len(pred_words)
    ret = []
    for input_tokens, pred_tokens in zip(input_words, pred_words):
        if lib.data.constants.URL in pred_tokens or lib.data.constants.HASH in pred_tokens or lib.data.constants.MENTION in pred_tokens:
            sent_length = min(len(input_tokens),len(pred_tokens))
            for i in range(sent_length):
                if(pred_tokens[i] == lib.data.constants.URL or pred_tokens[i] == lib.data.constants.HASH or pred_tokens[i] == lib.data.constants.MENTION):
                    pred_tokens[i] = input_tokens[i]
        ret.append(pred_tokens)
    return ret
    """


def handle_numbers(input_words, pred_words):
    assert len(input_words) == len(pred_words)
    ret = []
    for input_tokens, pred_tokens in zip(input_words, pred_words):
        sent_length = min(len(input_tokens),len(pred_tokens))
        for i in range(sent_length):
            if(any(char.isdigit() for char in pred_tokens[i])):
                pred_tokens[i] = input_tokens[i]
        ret.append(pred_tokens)
    return ret

def handle_unk(input, input_words, pred_words, unk_model, unkowns_file=None, is_nuke=False, confidence_tres=95):
    if(unk_model):
        assert len(input) == len(pred_words)
        ret = []
        if is_nuke:
            input = [input[0]]
            input_words = [input_words[0]]
            pred_words = [pred_words[0]]
        for input_tokens, input_words_tokens, pred_tokens in zip(input, input_words, pred_words):
            if lib.data.constants.UNK_WORD in input_tokens:
                sent_length = min(len(input_tokens),len(pred_tokens))
                for i in range(sent_length):
                    if(input_tokens[i]==lib.data.constants.UNK_WORD):
                        print('processing unk: ', input_words_tokens[i])
                        unk_src = unk_model.encoder.vocab.to_indices(input_words_tokens[i],
                                        eosWord=unk_model.opt.eos,bosWord=unk_model.opt.bos).view(1, -1)
                        #Repeat as many times as the batch size, awful but works
                        unk_src = torch.cat([unk_src]*unk_model.opt.batch_size)
                        unk_src = Variable(unk_src)
                        if input_words_tokens[i] == '' or input_words_tokens[i] == ' ':
                            continue
                        src_lens = Variable(torch.LongTensor([len(p) for p in unk_src]))
                        if unk_model.opt.cuda: unk_src = unk_src.cuda()
                        if unk_model.opt.cuda: src_lens = src_lens.cuda()
                        unk_src = unk_src.t()
                        batch = {}
                        batch['src'] = unk_src, src_lens
                        batch['tgt'] = unk_src, src_lens
                        probs, translation = unk_model.translate(batch)
                        confidence = probs.transpose()[0].max()
                        translation = translation.t().tolist()
                        trsl2wrds = lib.metric.to_words(translation, unk_model.encoder.vocab)
                        if unkowns_file:
                            unkowns_file.writerow([input_words_tokens[i], ''.join(trsl2wrds[0]), confidence])
                        pred_tokens[i] = ''.join(trsl2wrds[0]) if confidence > confidence_tres and input_words_tokens[i].isalpha()  else input_words_tokens[i] 
                        if input_words_tokens[i]!=pred_tokens[i]:
                            logger.info('secondary model confidence:{}, unk_word:{}, prediction:{}'.format(confidence, input_words_tokens[i], pred_tokens[i]))
            ret.append(pred_tokens)
    else:
        ret = copy_unks(input, input_words, pred_words)
    return ret

def handle_unk_with_phon(input, input_words, pred_words, unk_model, phon_model, unkowns_file=None, is_nuke=False, confidence_tres=95):
    if(unk_model):
        assert len(input) == len(pred_words)
        ret = []
        if is_nuke:
            input = [input[0]]
            input_words = [input_words[0]]
            pred_words = [pred_words[0]]
        for input_tokens, input_words_tokens, pred_tokens in zip(input, input_words, pred_words):
            if lib.data.constants.UNK_WORD in input_tokens:
                sent_length = min(len(input_tokens),len(pred_tokens))
                for i in range(sent_length):
                    if(input_tokens[i]==lib.data.constants.UNK_WORD):
                        print('processing unk: ', input_words_tokens[i])
                        #Repeat as many times as the batch size, awful but works
                        
                        # character unk processing
                        unk_src = unk_model.encoder.vocab.to_indices(input_words_tokens[i], eosWord=unk_model.opt.eos, bosWord=unk_model.opt.bos).view(1, -1)
                        unk_src = torch.cat([unk_src]*unk_model.opt.batch_size)
                        unk_src = Variable(unk_src)
                        src_lens = Variable(torch.LongTensor([len(p) for p in unk_src]))
                        if unk_model.opt.cuda: unk_src = unk_src.cuda()
                        if unk_model.opt.cuda: src_lens = src_lens.cuda()
                        if input_words_tokens[i] == '' or input_words_tokens[i] == ' ':
                            continue
                        unk_src = unk_src.t()
                        unk_batch = {}
                        unk_batch['src'] = unk_src, src_lens
                        unk_batch['tgt'] = unk_src, src_lens
                        unk_probs, unk_translation = unk_model.translate(unk_batch)
                        unk_confidence = unk_probs.transpose()[0].max()
                        if unk_confidence < 80.0:
                            phon_probs, phon_translation = phoneme_model_pred(phon_model, input_words_tokens, i)
                            phon_confidence = phon_probs.transpose()[0].max()
                        else:
                            phon_confidence = 0.0
                            phon_translation = ''
                        choice ,transl, confidence = max((1, unk_translation, unk_confidence),(0, phon_translation, phon_confidence), key=lambda x: x[2])
                        translation = transl.t().tolist()
                        vocab = unk_model.encoder.vocab if choice == 1 else phon_model.encoder.vocab
                        trsl2wrds = lib.metric.to_words(translation, vocab)
                        if unkowns_file: 
                            unkowns_file.writerow([input_words_tokens[i], ''.join(trsl2wrds[0]), confidence])
                        pred_tokens[i] = ''.join(trsl2wrds[0]) if confidence > confidence_tres and input_words_tokens[i].isalpha()  else input_words_tokens[i] 
                        if input_words_tokens[i]!=pred_tokens[i]: 
                            logger.info('secondary model confidence:{}, unk_word:{}, prediction:{}'.format(confidence, input_words_tokens[i], pred_tokens[i]))
            ret.append(pred_tokens)
    else:
        ret = copy_unks(input, input_words, pred_words)
    return ret

def phoneme_model_pred(phon_model, input_words_tokens, i):
    phon_src = phon_model.encoder.vocab.to_indices(phon_model.transliterator(input_words_tokens[i]), eosWord=phon_model.opt.eos, bosWord=phon_model.opt.bos).view(1, -1)
    src_lens = Variable(torch.LongTensor([len(p) for p in phon_src]))
    phon_src = torch.cat([phon_src]*phon_model.opt.batch_size)
    phon_src = Variable(phon_src)
    phon_src_lens = Variable(torch.LongTensor([len(p) for p in phon_src]))
    if phon_model.opt.cuda: phon_src = phon_src.cuda()
    if phon_model.opt.cuda: phon_src_lens = src_lens.cuda()
    phon_src = phon_src.t()
    phon_batch = {}
    phon_batch['src'] = phon_src, phon_src_lens
    phon_batch['tgt'] = phon_src, phon_src_lens
    phon_probs, phon_translation = phon_model.translate(phon_batch)
    return phon_probs, phon_translation


def copy_unks(input, input_words, pred_words):
    assert len(input) == len(pred_words)
    ret = []
    for input_tokens, input_words_tokens, pred_tokens in zip(input, input_words, pred_words):
        if lib.data.constants.UNK_WORD in input_tokens or lib.data.constants.UNK_WORD in pred_tokens:
            sent_length = min(len(input_tokens),len(pred_tokens))
            for i in range(sent_length):
                if(input_tokens[i] == lib.data.constants.UNK_WORD or pred_tokens[i] == lib.data.constants.UNK_WORD):
                    pred_tokens[i] = input_words_tokens[i]
        ret.append(pred_tokens)
    return ret


def clean_self_toks(inputs, preds, token):
    ret_preds = []
    for input_tokens, pred_tokens in zip(inputs, preds):
        if token in pred_tokens:
            length = min(len(input_tokens), len(pred_tokens))
            for i in range(length):
                if pred_tokens[i] == token:
                    pred_tokens[i] = input_tokens[i]
        ret_preds.append(pred_tokens)
    return ret_preds


def to_words(sents, dict):
    ret = []
    for sent in sents:
        sent = [dict.itos(id) for id in sent]
        sent = clean_sentence(sent, remove_unk=False)
        ret.append(sent)
    return ret

def to_words_pretrained(sents, vocab):
    ret = []
    for sent in sents:
        sent = [vocab.itos[id] for id in sent]
        sent = clean_sentence(sent, remove_unk=False)
        ret.append(sent)
    return ret


def char_to_words(sents):
    ret = []
    for sent in sents:
        sent = ''.join(sent).split('#')
        ret.append(sent)
    return ret


def compute_single(pair, metric_fn=None):
    input, pred, gold  = pair
    if len(pred) == 0:
        score = 0.
    else:
        score = metric_fn(input, pred, gold)['f1']
    return score


def compute_batch(inputs, preds, golds, metric_fn):
    compute_single_with_metric = functools.partial(compute_single, metric_fn=metric_fn)
    scores = map(compute_single_with_metric, zip(inputs, preds, golds))
    return list(scores)


def compute_numcorrects(dec_logits, targets, pad_masks=None):
    log_dist = F.log_softmax(dec_logits, dim=-1)
    pred_flat = log_dist.max(-1)[1]
    num_corrects = int(pred_flat.eq(targets).masked_select(pad_masks).float().data.sum()) if pad_masks is not None\
        else int(pred_flat.eq(targets).float().data.sum())
    return num_corrects
