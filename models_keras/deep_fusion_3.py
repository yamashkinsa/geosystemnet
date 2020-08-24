from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, Average, Maximum, Minimum, \
    Concatenate
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization

def base_unit(x):
    x_1 = Conv2D(32, (5, 5))(x)
    x_2 = Activation('relu')(x_1)
    x_3 = MaxPooling2D(pool_size=(2, 2))(x_2)
    x_4 = Conv2D(64, (4, 4))(x_3)
    x_5 = Activation('relu')(x_4)
    x_6 = MaxPooling2D(pool_size=(2, 2))(x_5)
    x_7 = Conv2D(64, (3, 3))(x_6)
    x_8 = Activation('relu')(x_7)
    x_9 = MaxPooling2D(pool_size=(2, 2))(x_8)

    return [x_3, x_6, x_9]

def merge_unit(x8, x12, x14, xHR):
    merge_1 = Concatenate()([x8[0], x12[0], x14[0], xHR[0]])
    merge_1 = Conv2D(32, (4, 4))(merge_1)
    merge_1 = Activation('relu')(merge_1)
    merge_1 = MaxPooling2D(pool_size=(2, 2))(merge_1)
    merge_1 = Dropout(0.5)(merge_1)

    merge_2 = Concatenate()([x8[1], x12[1], x14[1], xHR[1], merge_1])
    merge_2 = Conv2D(64, (3, 3))(merge_2)
    merge_2 = Activation('relu')(merge_2)
    merge_2 = MaxPooling2D(pool_size=(2, 2))(merge_2)
    merge_2 = Dropout(0.5)(merge_2)

    merge_3 = Concatenate()([x8[2], x12[2], x14[2], xHR[2], merge_2])
    merge_3 = Conv2D(64, (3, 3))(merge_3)
    merge_3 = Activation('relu')(merge_3)
    merge_3 = MaxPooling2D(pool_size=(2, 2))(merge_3)
    merge_3 = Dropout(0.5)(merge_3)

    return merge_3

def get_model(img_rows, img_cols, depth, num_classes):
    model_title = 'deep_fusion_3'

    input_x8 = Input(shape=(img_rows, img_cols, depth))
    input_x12 = Input(shape=(img_rows, img_cols, depth))
    input_x14 = Input(shape=(img_rows, img_cols, depth))
    input_xHR = Input(shape=(img_rows, img_cols, depth))

    x8  = base_unit(input_x8)
    x12 = base_unit(input_x12)
    x14 = base_unit(input_x14)
    xHR = base_unit(input_xHR)

    xMerge = merge_unit(x8, x12, x14, xHR)

    x = Flatten()(xMerge)

    x = Dense(1024)(x)  # 1024
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)  # 128
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=[input_x8, input_x12, input_x14, input_xHR],
                  outputs=x)  # To define a model, just specify its input and output layers

    return {'title': model_title, 'model': model}