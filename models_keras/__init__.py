import importlib
from models_keras import mlp_fusion
from models_keras import mlp_fusion_2
from models_keras import deep_fusion_3
from models_keras import simple_cnn
from models_keras import geosystemnet


def get_model(model, img_rows, img_cols, depth, num_classes):
    if (model == 'mlp_fusion'):
        return mlp_fusion.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'mlp_fusion_2'):
        return mlp_fusion_2.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'simple_cnn'):
        return simple_cnn.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'deep_fusion_3'):
        return deep_fusion_3.get_model(img_rows, img_cols, depth, num_classes)
    if (model == 'geosystemnet'):
        return geosystemnet.get_model(img_rows, img_cols, depth, num_classes)
