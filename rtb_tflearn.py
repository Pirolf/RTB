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
dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',
                                 weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.01)

dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 32, weights_init=tflearn.initializations.normal(),
                                 regularizer='L2', weight_decay=0.01)

dropout2 = tflearn.dropout(dense2, 0.8)
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
rus = RandomUnderSampler(ratio={0: 1531*30, 1: 1531})
smote = SMOTE(n_jobs=-1, random_state=42,
	      k_neighbors=3, m_neighbors=5)
rus2 = RandomUnderSampler(ratio={0: 1531*100, 1: 1531*50})
 
#ros = RandomOverSampler(ratio={0: 1531*10, 1: 1531*5})
# smoteenn = SMOTEENN(smote=SMOTE(n_jobs=-1))

print("Resampling")

resampled_features, resampled_labels = rus.fit_sample(features, labels[:, 1])
resampled_features, resampled_labels = smote.fit_sample(
        resampled_features, resampled_labels)
#resampled_features, resampled_labels = rus2.fit_sample(
#        resampled_features, resampled_labels)

shuffled_features, shuffled_labels = shuffle(
    resampled_features, resampled_labels)

shuffled_labels = to_categorical(shuffled_labels)


print("Resampling done")

model.fit(shuffled_features, shuffled_labels, n_epoch=3,
        validation_set=0.2,
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
