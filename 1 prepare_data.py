import os
from osgeo import gdal, osr
import numpy as np
import utm
import re
import pickle
from sklearn.model_selection import train_test_split

"""
2.1 Data preparation
The script performs preliminary processing of ERS data from the Sentinel-2 satellite: normalization and export to a file for further use.
sentinel_source_dir - geotiff image storage path 
processed_data_dir - directory for generated data
"""

sentinel_source_dir = 'sentinel_source/sentinel-tif'
processed_data_dir = '/generated_files'

train_x = []
train_y = []
train_meta = []

for root, dirs, files in os.walk(sentinel_source_dir):
    classNumber = len(dirs)
    class_i = 1
    for class_title in dirs:
        class_max = 0
        class_vector = [0] * classNumber
        class_vector[class_i-1] = 1
        images_path = sentinel_source_dir + '/' + class_title

        for class_root, class_dirs, class_files in os.walk(images_path):
            for image_file in class_files:
                file_path = class_root+'/'+image_file
                ds = gdal.Open(file_path)
                metadata = ds.GetMetadata()
                matrix = np.asarray(ds.ReadAsArray())
                np.putmask(matrix, matrix >= (2 ** 12 - 1), (2 ** 12 - 1))
                shape = matrix.shape
                prj = ds.GetProjection()
                x = ds.GetGeoTransform()[0]
                y = ds.GetGeoTransform()[3]
                pixel_width  = abs(ds.GetGeoTransform()[1])
                pixel_height = abs(ds.GetGeoTransform()[5])
                srs = osr.SpatialReference(wkt=prj)
                zone = re.search('(\d+)([A-Z])', srs.GetAttrValue('projcs', 0))
                [lat, lon] = (utm.to_latlon(*(x, y, int(zone.group(1)), zone.group(2))))
                train_x.append(matrix)
                train_y.append(class_vector)
                train_meta.append([lat, lon])
        class_i = class_i + 1

pickle.dump([train_x, train_y, train_meta], open(processed_data_dir+'/data_not_normalized.data', 'wb'))

x = np.array(train_x).astype(float)
y = np.array(train_y)

mean = np.mean(x, axis=(0, 1, 2))
std = np.std(x, axis=(0, 1, 2))
x -= mean
x /= std
x = np.rollaxis(x, 1, 4)
x = x[:,:,:,[1,2,3]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
pickle.dump([x_train, x_test, y_train, y_test], open(processed_data_dir+'/data_normalized.data', 'wb'), protocol=4) # data_ready
