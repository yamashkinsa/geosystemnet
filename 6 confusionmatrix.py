import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import keras
from sklearn import metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

"""
6 Confusion Matrix Builder
The script provides calculating of confusion matrix and metrics for a comparative assessment of the accuracy of trained models
"""

[x8, x12, x14, xHR, y] = pickle.load(open('!back/!generated_files/hierarchyData.data', "rb"))

test_site = 0.99
x8_train, x8_test, x12_train, x12_test, x14_train, x14_test, xHR_train, xHR_test, y_train, y_test = train_test_split(x8, x12, x14, xHR, y, test_size=test_site, random_state=42)

model_title = 'InceptionV3__august'

model = keras.models.load_model('models_trained/model_'+model_title+'.h5')

#y_pred = to_categorical( np.argmax(model.predict([x8_test, x12_test, x14_test, xHR_test]), axis=1) , 10) # x8_test, x12_test, x14_test, xHR_test
y_pred = to_categorical( np.argmax(model.predict([xHR_test]), axis=1) , 10) # x8_test, x12_test, x14_test, xHR_test

matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
matrix_sum = np.sum(matrix, axis=1)
matrix_percentage = matrix.astype(float) / matrix_sum.astype(float) * 100

labels = ['Annual Crop','Forest','Herbaceous','Highway','Industrial','Pasture','Permanent Crop','Residential','River','Water']
df_cm = pd.DataFrame(matrix, index = [i for i in labels], columns = [i for i in labels])
fig = plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="Greens", fmt="d", linewidths=.5, linecolor='black')

plt.show()
fig.savefig('cm_'+model_title+'.png')

accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, average=None)
precision = metrics.precision_score(y_test, y_pred, average=None)
recall = metrics.recall_score(y_test, y_pred, average=None)
hamming_loss = metrics.hamming_loss(y_test, y_pred)
fbeta_score = metrics.fbeta_score(y_test, y_pred, beta=0.5, average=None)
log_loss = metrics.log_loss(y_test, y_pred)
matthews_corrcoef = metrics.matthews_corrcoef (y_test.argmax(axis=1), y_pred.argmax(axis=1))
cohen_kappa_score = metrics.cohen_kappa_score (y_test.argmax(axis=1), y_pred.argmax(axis=1))
classification_report = metrics.classification_report(y_test, y_pred)

print (classification_report)
print('Accuracy: '+str(accuracy))
print ('Hamming loss: '+str(hamming_loss))
print ('Log Loss: ' + str(log_loss))
print ('Matthews Corrcoef: ' + str(matthews_corrcoef))
print ('Cohen Kappa Score: ' + str(cohen_kappa_score))

print ('Precision')
print (precision)
print ('Recall')
print (recall)
print ('F1')
print (f1)
print ('Fbeta Score')
print (fbeta_score)
