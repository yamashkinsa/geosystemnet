import numpy as np
import pickle
from PIL import Image, ImageDraw

def normalize_it(images_path):
    image = Image.open(images_path)
    image = image.resize((64,64), Image.LANCZOS)
    matrix = np.array(image).astype(float)
    matrix_n = (matrix - np.mean(matrix)) / np.std(matrix)
    return matrix_n

processed_data_dir = '!back/!generated_files'

[xData, yData, train_meta] = pickle.load(open(processed_data_dir+'/data_normalized.data', "rb"))

numInCat = 0
index_in_cat = 0
last_cat = 0
i = 0

xDataHR  = ([])
xData8   = ([])
xData12  = ([])
xData14  = ([])

for yArr in yData:
    cat_id = np.argmax(yArr) + 1
    if (last_cat != cat_id):
        last_cat = cat_id
        index_in_cat = 1

    ##########
    images_path_8 = 'additional/' + str(cat_id) + '/8/' + str(index_in_cat) + '.jpg'
    images_path_12 = 'additional/' + str(cat_id) + '/12/' + str(index_in_cat) + '.jpg'
    images_path_14 = 'additional/' + str(cat_id) + '/14/' + str(index_in_cat) + '.jpg'
    images_path_hr = 'additional/' + str(cat_id) + '/hr/' + str(index_in_cat) + '.jpg'

    xData8.append(normalize_it(images_path_8))
    xData12.append(normalize_it(images_path_12))
    xData14.append(normalize_it(images_path_14))
    xDataHR.append(normalize_it(images_path_hr))
    index_in_cat = index_in_cat + 1
    i = i + 1

hierarchyData = ([
    np.asarray(xData8, dtype=np.float32),
    np.asarray(xData12, dtype=np.float32),
    np.asarray(xData14, dtype=np.float32),
    np.asarray(xDataHR, dtype=np.float32),
    np.asarray(yData, dtype=np.float32)
])

pickle.dump(hierarchyData, open('!back/!generated_files/hierarchyData_lack.data', 'wb'))