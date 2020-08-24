import numpy as np
import pickle
from PIL import Image, ImageDraw

def normalize_it(images_path):
    image = Image.open(images_path)
    image = image.resize((64,64), Image.LANCZOS)
    matrix = np.array(image).astype(float)
    matrix_n = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix_n

"""
3 Data augmention with MapBox API. Normalization
The script performs normalization of extended data imported from MapBox API.
processed_data_dir - directory for generated data
z - zoom level of extended data
"""


processed_data_dir = 'generated_files'
z = 14

[xData, yData, train_meta] = pickle.load(open(processed_data_dir+'/data_normalized.data', "rb"))

numInCat = 0
index_in_cat = 0
last_cat = 0
i = 0

xData  = ([])

for yArr in yData:
    cat_id = np.argmax(yArr) + 1
    if (last_cat != cat_id):
        last_cat = cat_id
        index_in_cat = 1

    images_path = 'additional/' + str(cat_id) + '/'+ str(z) + '/' + str(index_in_cat) + '.jpg'
    xData.append(normalize_it(images_path))
    index_in_cat = index_in_cat + 1
    i = i + 1

hierarchyData = ([
    np.asarray(xData, dtype=np.float32),
    np.asarray(yData, dtype=np.float32)
])

pickle.dump(hierarchyData, open(processed_data_dir+'/hierarchyData_lack.data', 'wb'))
