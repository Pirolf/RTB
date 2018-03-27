from sklearn.metrics import *
import numpy as np
import pandas as pd
import tflearn

from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler


def shuffle(features, labels):
    p = np.random.permutation(len(features))
    return features[p], labels[p]


input_layer = tflearn.input_data(shape=[None, 88])
dense1 = tflearn.fully_connected(input_layer, 64,
                                 weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.001)

dropout1 = tflearn.dropout(dense1, 0.9)
dense2 = tflearn.fully_connected(dropout1, 32, weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.001)

dropout2 = tflearn.dropout(dense2, 0.9)
softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.98)
net = tflearn.regression(softmax, optimizer=sgd,
                 loss='roc_auc_score')
input_path = '~/data/biddings.csv'
data = pd.read_csv(input_path)
print(data.shape)

train = data[:800000]
test = data[800000:]

sample = train.sample(frac=1)
features = sample.drop('convert', axis=1).values
labels = to_categorical(sample.convert.ravel())


test_features = test.drop('convert', axis=1).values
test_labels = to_categorical(test.convert.ravel())

print("finished loading data")


# Training
model = tflearn.DNN(net, tensorboard_verbose=2)

'''
Resampling
'''
print(features.shape, labels.shape)
rus = RandomUnderSampler(ratio={0: 1531*20, 1: 1531})
# ros = RandomOverSampler(ratio={0: 1531*20, 1: 1531*5})
smote = SMOTE(n_jobs=-1, random_state=42)
# smoteenn = SMOTEENN(smote=SMOTE(n_jobs=-1))

print("Resampling")
resampled_features, resampled_labels = rus.fit_sample(features, labels[:, 1])
resampled_features, resampled_labels = smote.fit_sample(
        resampled_features, resampled_labels)

shuffled_features, shuffled_labels = shuffle(
    resampled_features, resampled_labels)

shuffled_labels = to_categorical(shuffled_labels)


print("Resampling done")

model.fit(shuffled_features, shuffled_labels, n_epoch=10,
        validation_set=0.2,
        batch_size=16, show_metric=True)

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


def rtb_precision_recall(test_labels, y_preds):
    precision, recall, fbeta_score, support = precision_recall_fscore_support(
        test_labels[:, 1], y_preds.argmax(axis=-1))
    print("Precision = %0.3f, Recall = %0.3f" % (np.mean(precision), np.mean(recall)))
    return precision, recall


y_preds = model.predict(test_features)
print(y_preds.shape)


rtb_confusion_matrix(test_labels, y_preds)
rtb_f1_score(test_labels, y_preds)
rtb_precision_recall(test_labels, y_preds)
print(roc_auc_score(test_labels, y_preds))

