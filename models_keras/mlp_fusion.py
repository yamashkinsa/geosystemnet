from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, Average, Maximum, Minimum, \
    Concatenate
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization


def get_model(img_rows, img_cols, depth, num_classes):
    model_title = 'deep_fusion'

    input_x8 = Input(shape=(img_rows, img_cols, depth))
    input_x12 = Input(shape=(img_rows, img_cols, depth))
    input_x14 = Input(shape=(img_rows, img_cols, depth))
    input_xHR = Input(shape=(img_rows, img_cols, depth))

    x8_1 = Conv2D(64, (3, 3))(input_x8)
    #x8_1 = Conv2D(64, (3, 3))(x8_1)
    x8_2 = Activation('relu')(x8_1)
    x8_3 = MaxPooling2D(pool_size=(2, 2))(x8_2)
    #x8_3d = Dropout(0.5)(x8_3)
    x8_4 = Conv2D(64, (3, 3))(x8_3)
    #x8_4 = Conv2D(64, (3, 3))(x8_4)
    x8_5 = Activation('relu')(x8_4)
    x8_6 = MaxPooling2D(pool_size=(2, 2))(x8_5)
    #x8_6d = Dropout(0.5)(x8_6)
    x8_7 = Conv2D(32, (3, 3))(x8_6)
    #x8_7 = Conv2D(32, (3, 3))(x8_7)
    x8_8 = Activation('relu')(x8_7)
    x8_9 = MaxPooling2D(pool_size=(2, 2))(x8_8)
    #x8_9d = Dropout(0.5)(x8_9)

    x12_1 = Conv2D(64, (3, 3))(input_x12)
    #x12_1 = Conv2D(64, (3, 3))(x12_1)
    x12_2 = Activation('relu')(x12_1)
    x12_3 = MaxPooling2D(pool_size=(2, 2))(x12_2)
    #x12_3d = Dropout(0.5)(x12_3)
    x12_4 = Conv2D(64, (3, 3))(x12_3)
    #x12_4 = Conv2D(64, (3, 3))(x12_4)
    x12_5 = Activation('relu')(x12_4)
    x12_6 = MaxPooling2D(pool_size=(2, 2))(x12_5)
    #x12_6d = Dropout(0.5)(x12_6)
    x12_7 = Conv2D(32, (3, 3))(x12_6)
    #x12_7 = Conv2D(32, (3, 3))(x12_7)
    x12_8 = Activation('relu')(x12_7)
    x12_9 = MaxPooling2D(pool_size=(2, 2))(x12_8)
    #x12_9d = Dropout(0.5)(x12_9)

    x14_1 = Conv2D(64, (3, 3))(input_x14)
    #x14_1 = Conv2D(64, (3, 3))(x14_1)
    x14_2 = Activation('relu')(x14_1)
    x14_3 = MaxPooling2D(pool_size=(2, 2))(x14_2)
    #x14_3d = Dropout(0.5)(x14_3)
    x14_4 = Conv2D(64, (3, 3))(x14_3)
    #x14_4 = Conv2D(64, (3, 3))(x14_4)
    x14_5 = Activation('relu')(x14_4)
    x14_6 = MaxPooling2D(pool_size=(2, 2))(x14_5)
    #x14_6d = Dropout(0.5)(x14_6)
    x14_7 = Conv2D(32, (3, 3))(x14_6)
    #x14_7 = Conv2D(32, (3, 3))(x14_7)
    x14_8 = Activation('relu')(x14_7)
    x14_9 = MaxPooling2D(pool_size=(2, 2))(x14_8)
    #x14_9d = Dropout(0.5)(x14_9)

    xHR_1 = Conv2D(64, (3, 3))(input_xHR)
    #xHR_1 = Conv2D(64, (3, 3))(xHR_1)
    xHR_2 = Activation('relu')(xHR_1)
    xHR_3 = MaxPooling2D(pool_size=(2, 2))(xHR_2)
    #xHR_3d = Dropout(0.5)(xHR_3)
    xHR_4 = Conv2D(64, (3, 3))(xHR_3)
    #xHR_4 = Conv2D(64, (3, 3))(xHR_4)
    xHR_5 = Activation('relu')(xHR_4)
    xHR_6 = MaxPooling2D(pool_size=(2, 2))(xHR_5)
    #xHR_6d = Dropout(0.5)(xHR_6)
    xHR_7 = Conv2D(32, (3, 3))(xHR_6)
    #xHR_7 = Conv2D(32, (3, 3))(xHR_7)
    xHR_8 = Activation('relu')(xHR_7)
    xHR_9 = MaxPooling2D(pool_size=(2, 2))(xHR_8)
    #xHR_9d = Dropout(0.5)(xHR_9)

    merge_1 = Concatenate()([x8_3, x12_3, x14_3, xHR_3])
    merge_1 = Conv2D(128, (3, 3))(merge_1)
    #merge_1 = Conv2D(128, (3, 3))(merge_1)
    # merge_1 = BatchNormalization()(merge_1)
    merge_1 = Activation('relu')(merge_1)
    merge_1 = MaxPooling2D(pool_size=(2, 2))(merge_1)
    merge_1 = Dropout(0.5)(merge_1)

    merge_2 = Concatenate()([x8_6, x12_6, x14_6, xHR_6, merge_1])
    merge_2 = Conv2D(192, (3, 3))(merge_2)
    #merge_2 = Conv2D(192, (3, 3))(merge_2)
    #merge_2 = BatchNormalization()(merge_2)
    merge_2 = Activation('relu')(merge_2)
    merge_2 = MaxPooling2D(pool_size=(2, 2))(merge_2)
    merge_2 = Dropout(0.5)(merge_2)

    merge_3 = Concatenate()([x8_9, x12_9, x14_9, xHR_9, merge_2])
    merge_3 = Conv2D(160, (3, 3))(merge_3)
    #merge_3 = Conv2D(160, (3, 3))(merge_3)
    #merge_3 = BatchNormalization()(merge_3)
    merge_3 = Activation('relu')(merge_3)
    merge_3 = MaxPooling2D(pool_size=(2, 2))(merge_3)
    merge_3 = Dropout(0.5)(merge_3)

    x = Flatten()(merge_3)

    x = Dense(512)(x)  # 1024
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