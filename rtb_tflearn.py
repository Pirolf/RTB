from sklearn.metrics import *
import numpy as np
import pandas as pd
import tflearn

from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.ensemble import BalanceCascade
from sklearn.tree import DecisionTreeClassifier


KFOLD_SEED=42

input_layer = tflearn.input_data(shape=[None, 88])
dense1 = tflearn.fully_connected(input_layer, 88, activation='relu',
                                 weights_init=tflearn.initializations.xavier(),
                                 regularizer='L2', weight_decay=0.01)
dense2 = tflearn.fully_connected(dense1, 64, activation='tanh', weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.01)
bn1 = tflearn.batch_normalization(dense2)
dropout1 = tflearn.dropout(bn1, 0.9)
dense3 = tflearn.fully_connected(dropout1, 32, activation='tanh', weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.01)
dense4 = tflearn.fully_connected(dense3, 16, activation='relu', weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.01)
bn2 = tflearn.batch_normalization(dense4)
dropout2 = tflearn.dropout(bn2, 0.9)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay
sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.95, decay_step=20)
net = tflearn.regression(softmax, optimizer=sgd,
                 loss='roc_auc_score')
input_path = '~/data/biddings.csv'
data = pd.read_csv(input_path)
print(data.shape)

train = data[:800000]
test = data[800000:]

sample = train.sample(frac=1)
features = sample.drop('convert', axis=1).values
labels_1d = sample.convert.ravel()
labels = to_categorical(labels_1d)


test_features = test.drop('convert', axis=1).values
test_labels_1d = test.convert.ravel()
test_labels = to_categorical(test.convert.ravel())

print("finished loading data")


# Training
model = tflearn.DNN(net, tensorboard_verbose=2)

'''
Resampling
'''
print(features.shape, labels.shape)


def shuffle(features, labels):
    p = np.random.permutation(len(features))
    return features[p], labels[p]


def deep_ensemble_merged(smote=None):
    dt = DecisionTreeClassifier(max_features=0.2, random_state=KFOLD_SEED)
    ensembler = BalanceCascade(estimator=dt, n_max_subset=10, random_state=KFOLD_SEED)

    print("fitting sample")
    X_res, y_res = ensembler.fit_sample(features, labels_1d)
    print(X_res.shape, y_res.shape)
    
    print("training")

    # Merge sample batches
    Xs = None
    ys = None
    for i, X_train in enumerate(X_res):
        if Xs is None:
            Xs = np.array(X_res[i])
            ys = np.array(y_res[i])
            print(Xs.shape, ys.shape)
        else:
            Xs = np.concatenate((Xs, np.array(X_res[i])))
            ys = np.concatenate((ys, np.array(y_res[i])))
    
    print(Xs.shape, ys.shape)
    shuffle(Xs, ys)
    
    # Generate more synthetic samples
    if smote is not None:
        Xs, ys = smote.fit_sample(Xs, ys)
    
    shuffle(Xs, ys)
    ys = to_categorical(ys, 2)

    return Xs, ys


rus = RandomUnderSampler(ratio={0: 1531*30, 1: 1531})
smote = SMOTE(n_jobs=-1, random_state=42,
              k_neighbors=3, m_neighbors=5)
rus2 = RandomUnderSampler(ratio={0: 1531*100, 1: 1531*50})
 
#ros = RandomOverSampler(ratio={0: 1531*10, 1: 1531*5})
# smoteenn = SMOTEENN(smote=SMOTE(n_jobs=-1))

print("Resampling")

'''
0.589
resampled_features, resampled_labels = rus.fit_sample(features, labels[:, 1])
resampled_features, resampled_labels = smote.fit_sample(
        resampled_features, resampled_labels)

#resampled_features, resampled_labels = rus2.fit_sample(
#        resampled_features, resampled_labels)

shuffled_features, shuffled_labels = shuffle(
    resampled_features, resampled_labels)

shuffled_labels = to_categorical(shuffled_labels)
'''

print("Resampling done")

shuffled_features, shuffled_labels = deep_ensemble_merged()
model.fit(shuffled_features, shuffled_labels, n_epoch=10,
#         validation_set=0.2,
        batch_size=32, show_metric=True)

'''
Evaluation
'''


def rtb_confusion_matrix(test_labels, y_preds):
    m = confusion_matrix(test_labels[:,1], y_preds.argmax(axis=-1))

    print("Confusion Matrix:")
    print("True Negative = %d" % m[0][0])
    print("False Negative = %d" % m[1][0])
    print("True Positive = %d" % m[1][1])
    print("False Positive = %d" % m[0][1])


def rtb_f1_score(test_labels, y_preds):
    f = f1_score(test_labels[:, 1], y_preds.argmax(axis=-1))
    print("f1 score = %0.3f" % f)


y_preds = model.predict(test_features)
print(y_preds.shape)

train_preds = model.predict(shuffled_features)

print("--------test---------")
rtb_confusion_matrix(test_labels, y_preds)
rtb_f1_score(test_labels, y_preds)
print(roc_auc_score(test_labels[:,1], y_preds.argmax(axis=-1)))

print("--------train---------")
rtb_confusion_matrix(shuffled_labels, train_preds)
rtb_f1_score(shuffled_labels, train_preds)
print(roc_auc_score(shuffled_labels[:,1], train_preds.argmax(axis=-1)))
