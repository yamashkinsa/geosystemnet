from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Add, Multiply, Average, Maximum, Minimum, Concatenate
from keras.layers import Conv2D, Conv3D, MaxPooling3D, MaxPooling2D



def get_model(img_rows, img_cols, depth, num_classes):

    model_title = 'mlp_fusion_2'

    input_head = []
    input_head.append(Input(shape=(img_rows, img_cols, depth)))
    input_head.append(Input(shape=(img_rows, img_cols, depth)))
    input_head.append(Input(shape=(img_rows, img_cols, depth)))
    input_head.append(Input(shape=(img_rows, img_cols, depth)))

    outs = []
    for inp in input_head:
        x = Conv2D(64, (3, 3))(inp)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)  # this converts our 3D feature maps to 1D feature vectors
        outs.append(x)  # Output softmax layer

    x_add = Concatenate()([outs[0], outs[1], outs[2]])

    mlp_1 = 1024 # 1024
    mlp_2 = 128 # 128

    x_add = Dense(mlp_1)(x_add) # 1024
    x_add = Activation('relu')(x_add)
    x_add = Dropout(0.5)(x_add)
    x_add = Dense(mlp_2)(x_add)   #128
    x_add = Activation('relu')(x_add)
    x_add = Dropout(0.5)(x_add)

    x_hr = Dense(mlp_1)(outs[3]) #1024
    x_hr = Activation('relu')(outs[3])
    x_hr = Dropout(0.5)(outs[3])
    x_hr = Dense(mlp_2)(outs[3])  #128
    x_hr = Activation('relu')(outs[3])
    x_hr = Dropout(0.5)(outs[3])

    x = Concatenate()([x_add, x_hr])
    x = Dense(mlp_1)(x) #1024
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(mlp_2)(x) #128
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=[input_head[0], input_head[1], input_head[2], input_head[3]], outputs=x)  # To define a model, just specify its input and output layers

    return {'title': model_title, 'model': model}