import importlib
from models_keras import mlp_fusion
from models_keras import mlp_fusion_2
from models_keras import resnet50
from models_keras import simple_cnn
from models_keras import deep_fusion
from models_keras import deep_fusion_2
from models_keras import deep_fusion_3
from models_keras import geosystemnet
from models_keras import googlenet
from models_keras import densenet
from models_keras import resnet101
from models_keras import vgg16
from models_keras import inception_v3

def get_model(model, img_rows, img_cols, depth, num_classes):
    if (model == 'mlp_fusion'):
        return mlp_fusion.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'mlp_fusion_2'):
        return mlp_fusion_2.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'resnet50'):
        return resnet50.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'simple_cnn'):
        return simple_cnn.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'deep_fusion'):
        return deep_fusion.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'deep_fusion_2'):
        return deep_fusion_2.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'deep_fusion_3'):
        return deep_fusion_3.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'geosystemnet'):
        return geosystemnet.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'googlenet'):
        return googlenet.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'densenet'):
        return densenet.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'resnet101'):
        return resnet101.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'vgg16'):
        return vgg16.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'inception_v3'):
        return inception_v3.get_model(img_rows, img_cols, depth, num_classes)