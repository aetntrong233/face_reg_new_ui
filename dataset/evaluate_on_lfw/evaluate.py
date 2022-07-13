import math
import numpy as np


def probability(embeddings1, embeddings2):
    probs = []
    for i in range(len(embeddings1)):
        embedding1 = embeddings1[i]
        embedding2 = embeddings2[i]
        prob = np.dot(embedding1, embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
        # dist = np.arccos(prob) / math.pi
        probs.append(prob)
    return probs

def calculate_accuracy(embeddings1, embeddings2, threshold, actual_issame):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    probs = probability(embeddings1, embeddings2)
    accuracy, precision, recall = calculate_accuracy(threshold, probs, actual_issame)
    predict_issame = np.less(threshold, probs)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    recall = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    precision = 0 if (tp+fp==0) else float(tp) / float(tp+fp)

    # tpr = 0 if (tp+fn==0) else float(tp) / float(tp + fn)
    # fpr = 0 if (fp+tn==0) else float(fp) / float(fp + tn)
    # fnr = 0 if (tp+tn==0) else float(fn) / float(tp + tn)
    # tnr = 0 if (fp+tn==0) else float(tn) / float(fp + tn)

    accuracy = float(tp+tn)/probs.size
    return accuracy, precision, recall