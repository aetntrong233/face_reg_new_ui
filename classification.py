import numpy as np
import sklearn
import pickle


def predict_cosine(audit_feature, feature_db):
    probability_list = []
    for feature in feature_db:
        if audit_feature.size == feature.size:
            probability = np.dot(audit_feature, feature)/(np.linalg.norm(audit_feature)*np.linalg.norm(feature))
        else:
            probability = 0.0
        probability_list.append(probability)  
    # lấy ảnh có tỷ lệ giống cao nhất và so sánh với ngưỡng (THRESHOLD)
    max_prob = np.max(probability_list)
    max_index = probability_list.index(max_prob)
    return max_index, max_prob


# def train_svm(data, data_masked, id_list):
#     clf1 = sklearn.svm.SVC()
#     clf1.fit(data, id_list)
#     clf2 = sklearn.svm.SVC()
#     clf2.fit(data_masked, id_list)
#     pickle.dump(clf1, open('storage/svm_classifier.sav', 'wb'))
#     pickle.dump(clf2, open('storage/mask_svm_classifier.sav', 'wb'))
#     return clf1, clf2


# def predict_svm(clf, audit_feature):
#     p = np.array(clf.decision_function([audit_feature])) # decision is a voting function
#     prob = np.exp(p)/np.sum(np.exp(p),axis=1, keepdims=True) # softmax after the voting
#     max_prob = np.max(prob)
#     preds = clf.predict([audit_feature])
#     max_index = np.argmax(preds)
#     return max_index, max_prob
