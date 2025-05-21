import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def calculate_precision(Gt, P):
    return precision_score(Gt, P)

def calculate_recall(Gt, P):
    return recall_score(Gt, P)

def calculate_f1(Gt, P):
    return f1_score(Gt, P)

def calculate_auc(Gt, P):
    return roc_auc_score(Gt, P)

def calculate_confusion_matrix(Gt, P):
    return confusion_matrix(Gt, P)

def evaluate(Gt, P):
  return {
    "accuracy": (accuracy_score(Gt, P) * 100),
    "precision": (precision_score(Gt, P) * 100),
    "recall": (recall_score(Gt, P) * 100),
    "f1": (f1_score(Gt, P) * 100),
    "auc": (roc_auc_score(Gt, P) * 100),
    "conf_matrix": (confusion_matrix(Gt, P))
  }
 





