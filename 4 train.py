import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# gpu or cpu
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import pickle
from keras import optimizers
import time
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import models_keras

"""
4 Deep Models Training
The script performs training deep learning models with prepared data.
processed_data_dir - directory for generated data
"""

processed_data_dir = 'generated_files'
batch_size = 32
suffix = '_august' 
num_classes = 10
epochs = 50 #15
number_exp = 2 #5
loss_function = 'categorical_crossentropy' 
models = [
    ['simple_cnn', 0],
    ['geosystemnet', 1],
]
test_size = 0.9

[x8, x12, x14, xHR, y] = pickle.load(open(processed_data_dir+'/hierarchyData.data', "rb"))
x8_train, x8_test, x12_train, x12_test, x14_train, x14_test, xHR_train, xHR_test, y_train, y_test = train_test_split(x8, x12, x14, xHR, y, test_size=test_size)
shape_data = x8.shape
img_rows, img_cols, depth = shape_data[1], shape_data[2], shape_data[3]
input_shape = (img_rows, img_cols, depth)

for modelClass in models:
    model_experimental_data = []
    for i in range(number_exp):

        mdl = models_keras.get_model(modelClass[0], img_rows, img_cols, depth, num_classes)
        model = mdl['model']
        model_title = mdl['title'] + '_' + suffix
        print (model_title)

        model.summary()
        keras.utils.plot_model(model, 'models_trained/model_' + model_title + '.png', show_shapes=True)
        model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        if (modelClass[1]):
            xTrain = [x8_train, x12_train, x14_train, xHR_train]
            xTest = [x8_test, x12_test, x14_test, xHR_test]
        else:
            xTrain = xHR_train
            xTest = xHR_test

        start_time = time.time()

        history = model.fit(xTrain,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(xTest, y_test),
                            verbose=1 #callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
                            )

        elapsed_time = (time.time() - start_time)
        score = model.evaluate(xTest, y_test, verbose=0)
        model_experimental_data.append({
            'model': model_title,
            'elapsed_time': elapsed_time,
            'i': i,
            'loss': score[0],
            'accuracy': score[1],
            'history_accuracy': history.history['accuracy'],
            'history_val_accuracy': history.history['val_accuracy'],
            'history_loss': history.history['loss'],
            'history_val_loss': history.history['val_loss'],
        })

        print (i)
        print (score)
        print (elapsed_time)

        plt.plot(history.history['accuracy'])
        plt.grid(b=True, which='major', color='grey', linestyle='-')
        plt.grid(b=True,  which='minor', color='silver', linestyle='-', alpha=0.5)
        plt.show()

    model.save('models_trained/model_'+model_title+'.h5')
    keras.utils.plot_model(model, 'models_trained/model_'+model_title+'.png', show_shapes=True)
    pickle.dump(model_experimental_data, open('models_trained/model_'+model_title+'.data', 'wb'))
