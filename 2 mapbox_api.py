import pickle
import math
import numpy as np
import requests
import pathlib

def list_contains_string(lst, string):
   return any(string in v for v in lst)

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def download_image(src, trg_dir, trg_file):
    img_data = requests.get(src).content
    path = pathlib.Path(trg_dir)
    path.mkdir(parents=True, exist_ok=True)
    handler = open(trg_dir+'/'+trg_file, 'wb')
    handler.write(img_data)
    return True

"""
2 Data augmention with MapBox API
The script performs normalized Sentinel-2 data augmention with MapBox API and export results to a file for further use.
processed_data_dir - directory for generated data
mapbox_token - your MapBox token
z - zoom level of extended data
"""

processed_data_dir = 'generated_files'
mapbox_token = 'pk.MAPBOX_TOKEN_HERE'
z = 14

[xData, yData, train_meta] = pickle.load(open(processed_data_dir+'/data_normalized.data', "rb"))



i = 0
numInCat = 0
index_in_cat = 0
last_cat = 0
for coords in train_meta:
    [x, y] = (deg2num(coords[0], coords[1], z))
    cat_id = np.argmax(yData[i]) + 1
    if (last_cat != cat_id):
        last_cat = cat_id
        index_in_cat = 1
    trg_dir  = 'additional/'+str(cat_id)+'/'+str(z)+'/'
    trg_file = str(index_in_cat)+'.jpg'
    src = 'https://api.mapbox.com/v4/mapbox.satellite/'+str(z)+'/'+str(x)+'/'+str(y)+'.jpg?access_token='+mapbox_token
    download_image(src, trg_dir, trg_file)
    print(trg_dir+trg_file)
    index_in_cat = index_in_cat + 1
    i = i + 1
