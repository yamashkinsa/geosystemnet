from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, Average, Maximum, Minimum, \
    Concatenate
from keras.layers import Conv2D, SeparableConv2D, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization

def base_unit(x):
    x_1 = SeparableConv2D(64, (3, 3))(x)
    x_1b = BatchNormalization()(x_1)
    x_2 = Activation('relu')(x_1b)
    x_3 = MaxPooling2D(pool_size=(2, 2))(x_2)

    x_4 = SeparableConv2D(128, (3, 3))(x_3)
    x_4b = BatchNormalization()(x_4)
    x_5 = Activation('relu')(x_4b)
    x_6 = MaxPooling2D(pool_size=(2, 2))(x_5)

    return [x_3, x_6]

def merge_unit(x8, x12, x14, xHR):
    merge_1 = Concatenate()([x8[0], x12[0], x14[0], xHR[0]])
    merge_1 = SeparableConv2D(128, (3, 3))(merge_1) # было 32
    merge_1b = BatchNormalization()(merge_1)
    merge_1 = Activation('relu')(merge_1b)
    merge_1 = MaxPooling2D(pool_size=(2, 2))(merge_1)

    merge_2 = Concatenate()([x8[1], x12[1], x14[1], xHR[1], merge_1])
    merge_2 = SeparableConv2D(256, (3, 3))(merge_2) # было 64
    merge_2b = BatchNormalization()(merge_2)
    merge_2 = Activation('relu')(merge_2b)
    merge_2 = MaxPooling2D(pool_size=(2, 2))(merge_2)

    return merge_2

def mlp(x, num_classes):

    if (num_classes > 2):
        activation = 'softmax'
    else:
        activation = 'sigmoid'

    x = Dense(128)(x)  # 512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(64)(x)  # 256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(num_classes)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x

def get_model(img_rows, img_cols, depth, num_classes):
    model_title = 'geosystemnet'

    input_x8 = Input(shape=(img_rows, img_cols, depth))
    input_x12 = Input(shape=(img_rows, img_cols, depth))
    input_x14 = Input(shape=(img_rows, img_cols, depth))
    input_xHR = Input(shape=(img_rows, img_cols, depth))

    # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->

    x8  = base_unit(input_x8)
    x12 = base_unit(input_x12)
    x14 = base_unit(input_x14)
    xHR = base_unit(input_xHR)

    xMerge = merge_unit(x8, x12, x14, xHR)
    x = Flatten()(xMerge)
    x = Dropout(0.4)(x)
    x = mlp(x, num_classes)

    model = Model(inputs=[input_x8, input_x12, input_x14, input_xHR], outputs=x)  # To define a model, just specify its input and output layers

    return {'title': model_title, 'model': model}
