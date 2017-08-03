import tensorflow as tf
import numpy as np
import pickle as pickle
import random

#      0: Nothing
#      1: One pair
#      2: Two pairs
#      3: Three of a kind
#      4: Straight
#      5: Flush
#      6: Full house
#      7: Four of a kind
#      8: Straight flush
#      9: Royal flush

train_file = 'poker_hands_data/poker-hand-training-true.data'
test_file = 'poker_hands_data/poker-hand-testing.data'

def process_data(file):
    hm_lines = 1000000

    featureset = []

    c = 0
    with open(file,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            c += 1
            if(c%100000 == 0):
                print("Step:",c)

            split = l.split(",")
            split[-1] = split[-1].strip()

            features = np.zeros(52)
            classification = np.zeros(10)

            i = 0
            while (i < 10):
                n = (int(split[i])-1)*13 + int(split[i+1])
                i += 2
                features[n-1] = 1

            classification[int(split[-1])] = 1

            features = list(features)
            featureset.append([features,classification])

    return featureset

def process_features(l):
    split = l.split(",")
    split[-1] = split[-1].strip()

    features = np.zeros(52)

    i = 0
    while (i < 10):
        n = (int(split[i])-1)*13 + int(split[i+1])
        i += 2
        features[n-1] = 1

    features = list(features)

    return features

def create_feature_sets_and_labels(train,test,test_size = 0.1):
    features = []
    features += process_data(train)
    features += process_data(test)
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y
