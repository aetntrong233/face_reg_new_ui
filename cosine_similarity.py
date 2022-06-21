import numpy as np


def cosine_similarity(vec1, vec2):
    if vec1.size != vec2.size:
        return 0.0
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))


def dot(vec1, vec2):
    ret = 0.0
    for i in range(vec1.size):
        ret += vec1[i]*vec2[i]
    return ret


def norm(vec):
    ret = 0
    for p in vec:
        ret += p*p
    return np.sqrt(ret)