import unicodedata
import numpy as np
import seqeval.metrics
import spacy
import torch
import argparse


from tqdm import tqdm, trange
from transformers import LukeTokenizer, LukeForEntitySpanClassification

class LukeLoader:
    
    def __init__(self, data, pretrained, splitsymbol=' ', batch_size=2):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'training on {self.device}')
        self.model = LukeForEntitySpanClassification.from_pretrained(pretrained)
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = LukeTokenizer.from_pretrained(pretrained)
        self.test_documents = self.load_documents(data, splitsymbol)
        self.test_examples = self.load_examples(self.test_documents)
        self.test_on_data(batch_size, )
        self.spacy_nlp = spacy.load("en_core_web_sm")


    def load_documents(self, dataset_file, splitsymbol):
        documents, words, labels, sentence_boundaries = [], [], [], []
        with open(dataset_file) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:
                        documents.append(dict(
                            words=words,
                            labels=labels,
                            sentence_boundaries=sentence_boundaries
                        ))
                        words = []
                        labels = []
                        sentence_boundaries = []
                    continue

                if not line:
                    if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    items = line.split(splitsymbol)
                    # only use word and IOB-label
                    words.append(items[0])
                    labels.append(items[-1])

        if words:
            documents.append(dict(
                words=words,
                labels=labels,
                sentence_boundaries=sentence_boundaries
            ))
            
        return documents


    def load_examples(self, documents):
        examples = []
        max_token_length = 510
        max_mention_length = 30

        for document in tqdm(documents):
            words = document["words"]
            subword_lengths = [len(self.tokenizer.tokenize(w)) for w in words]
            total_subword_length = sum(subword_lengths)
            sentence_boundaries = document["sentence_boundaries"]

            for i in range(len(sentence_boundaries) - 1):
                sentence_start, sentence_end = sentence_boundaries[i:i+2]
                if total_subword_length <= max_token_length:
                    # if the total sequence length of the document is shorter than the
                    # maximum token length, we simply use all words to build the sequence
                    context_start = 0
                    context_end = len(words)
                else:
                    # if the total sequence length is longer than the maximum length, we add
                    # the surrounding words of the target sentence　to the sequence until it
                    # reaches the maximum length
                    context_start = sentence_start
                    context_end = sentence_end
                    cur_length = sum(subword_lengths[context_start:context_end])
                    while True:
                        if context_start > 0:
                            if cur_length + subword_lengths[context_start - 1] <= max_token_length:
                                cur_length += subword_lengths[context_start - 1]
                                context_start -= 1
                            else:
                                break
                        if context_end < len(words):
                            if cur_length + subword_lengths[context_end] <= max_token_length:
                                cur_length += subword_lengths[context_end]
                                context_end += 1
                            else:
                                break

                text = ""
                for word in words[context_start:sentence_start]:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    text += word
                    text += " "

                sentence_words = words[sentence_start:sentence_end]
                sentence_subword_lengths = subword_lengths[sentence_start:sentence_end]

                word_start_char_positions = []
                word_end_char_positions = []
                for word in sentence_words:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    word_start_char_positions.append(len(text))
                    text += word
                    word_end_char_positions.append(len(text))
                    text += " "

                for word in words[sentence_end:context_end]:
                    if word[0] == "'" or (len(word) == 1 and self.is_punctuation(word)):
                        text = text.rstrip()
                    text += word
                    text += " "
                text = text.rstrip()

                entity_spans = []
                original_word_spans = []
                for word_start in range(len(sentence_words)):
                    for word_end in range(word_start, len(sentence_words)):
                        if sum(sentence_subword_lengths[word_start:word_end]) <= max_mention_length:
                            entity_spans.append(
                                (word_start_char_positions[word_start], word_end_char_positions[word_end])
                            )
                            original_word_spans.append(
                                (word_start, word_end + 1)
                            )

                examples.append(dict(
                    text=text,
                    words=sentence_words,
                    entity_spans=entity_spans,
                    original_word_spans=original_word_spans,
                ))

        return examples

    @staticmethod
    def is_punctuation(char):
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def test_on_data(self, batch_size):
        all_logits = []

        for batch_start_idx in trange(0, len(self.test_examples), batch_size):
            batch_examples = self.test_examples[batch_start_idx:batch_start_idx + batch_size]
            texts = [example["text"] for example in batch_examples]
            entity_spans = [example["entity_spans"] for example in batch_examples]

            inputs = self.tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            all_logits.extend(outputs.logits.tolist())
            

        final_labels = [label for document in self.test_documents for label in document["labels"]]

        final_predictions = []
        for example_index, example in enumerate(self.test_examples):
            logits = all_logits[example_index]
            max_logits = np.max(logits, axis=1)
            max_indices = np.argmax(logits, axis=1)
            original_spans = example["original_word_spans"]
            predictions = []
            for logit, index, span in zip(max_logits, max_indices, original_spans):
                if index != 0:  # the span is not NIL
                    predictions.append((logit, span, self.model.config.id2label[index]))

            # construct an IOB2 label sequence
            predicted_sequence = ["O"] * len(example["words"])
            for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
                if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                    predicted_sequence[span[0]] = "B-" + label
                    if span[1] - span[0] > 1:
                        predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

            final_predictions += predicted_sequence

        print(seqeval.metrics.classification_report([final_labels], [final_predictions], digits=4))


    def inference(self, text):
        doc = self.spacy_nlp(text)

        entity_spans = []
        original_word_spans = []
        for token_start in doc:
            for token_end in doc[token_start.i:]:
                entity_spans.append((token_start.idx, token_end.idx + len(token_end)))
                original_word_spans.append((token_start.i, token_end.i + 1))

        inputs = self.tokenizer(text, entity_spans=entity_spans, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        max_logits, max_indices = logits[0].max(dim=1)

        predictions = []
        for logit, index, span in zip(max_logits, max_indices, original_word_spans):
            if index != 0:  # the span is not NIL
                predictions.append((logit, span, self.model.config.id2label[int(index)]))

        # construct an IOB2 label sequence
        predicted_sequence = ["O"] * len(doc)
        for _, span, label in sorted(predictions, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        for token, label in zip(doc, predicted_sequence):
            print(token, label)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-large', type=lambda x: x in ['true', 'True', '1', 'yes'], default=False,help='decide if you want to transfer large model')
    parser.add_argument('-split_sym', type=str, default=' ', help='set symbol to split data')
    args = parser.parse_args()
    
    pretrained = "studio-ousia/luke-large-finetuned-conll-2003" if args.large else "studio-ousia/luke-base"
    data = './g.conll.pos_ext.conll'
    LukeLoader(data=data, pretrained=pretrained)