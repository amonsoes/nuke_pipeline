import numpy as np

class ClassMetric:
    
    def __init__(self, name):
        self.eps = np.finfo(float).eps
        self.label = name
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
    
    def process_pair(self, label, pred):
        correct = 0
        if self.label == label:
            if self.label == pred:
                self.true_positive += 1
                correct += 1
            else:
                self.false_negative += 1
        else:
            if label == pred:
                self.true_negative += 1
            elif self.label == pred:
                self.false_positive += 1
        return correct

    def calc_precision(self):
        self.precision = self.true_positive / (self.true_positive + self.false_positive + self.eps)
    
    def calc_recall(self):
        self.recall = self.true_positive / (self.true_positive + self.false_negative+self.eps)

    def calc_f1(self):
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall+self.eps)

class NukeMetric:
    
    def __init__(self, classes):
        self.correct = 0
        self.total = 0
        self.classes = {name:ClassMetric(name) for name in classes}
        self.seen_labels = set()
    
    def __call__(self, example):
        for pred, label in zip(example['ner_prediction'], example['labels']):
            self.seen_labels.add(label)
            self.total += 1
            for k in self.classes:
                self.correct += self.classes[k].process_pair(label, pred)
                
    def calc_accuracy(self):
        self.accuracy = self.correct / self.total
    
    def build_scores(self):
        for k in self.seen_labels:
            self.classes[k].calc_precision()
            self.classes[k].calc_recall()
            self.classes[k].calc_f1()
        self.macro_f1 = sum([self.classes[k].f1 for k in self.seen_labels]) / len(self.seen_labels)
        self.calc_accuracy()
    
    def get_scores(self):
        return {'accuracy': self.accuracy,
                'macro-f1': self.macro_f1,
                'classes' : self.classes}