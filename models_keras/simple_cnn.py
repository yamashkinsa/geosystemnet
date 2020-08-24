from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, Average, Maximum, Minimum, Concatenate
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D



def get_model(img_rows, img_cols, depth, num_classes):

    model_title = 'simple_cnn'
    # 1 - multihead work test
    inputHR = Input(shape=(img_rows, img_cols, depth))

    if (num_classes > 2):
        activation = 'softmax'
    else:
        activation = 'sigmoid'

    resp = Conv2D(64, (3, 3))(inputHR)
    resp = Activation('relu')(resp)
    resp = MaxPooling2D(pool_size=(2, 2))(resp)
    resp = Conv2D(32, (3, 3))(resp)
    resp = Activation('relu')(resp)
    resp = MaxPooling2D(pool_size=(2, 2))(resp)
    resp = Flatten()(resp)  # this converts our 3D feature maps to 1D feature vectors
    resp = Dense(64)(resp)
    resp = Activation('relu')(resp)
    resp = Dropout(0.5)(resp)
    resp = Dense(num_classes)(resp)
    resp = Activation(activation)(resp)

    model = Model(inputs=[inputHR],
                  outputs=resp)  # To define a model, just specify its input and output layers

    return {'title': model_title, 'model': model}
