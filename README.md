# GeoSystemNet

### 1 GeoSystemNet Code
The GeoSystemnet deep model code is located in **geosystemnet/models_keras/geosystemnet.py**

### 2 Helper scripts
This section provides a list of auxiliary scripts that allow you to prepare Sentinel-2 data for analysis, train deep neural network models, evaluate their accuracy by building diagrams, calculating error matrices and metrics based on them.

#### 2.1 Data preparation (1 prepare_data.py)
The script performs preliminary processing of ERS data from the Sentinel-2 satellite: normalization and export to a file for further use.

Input data:
* sentinel_source_dir - geotiff image storage path 
* processed_data_dir - directory for generated data

Output data:
* List of normalized data for test and training, saved in file

#### 2.2 Data augmention with MapBox API (2 mapbox_api.py)
The script performs normalized Sentinel-2 data augmention with MapBox API and export results to a file for further use.

Input data:
* mapbox_token - your MapBox token
* z - zoom level of extended data

Output data:
* Extended dataset

#### 2.3 Data augmention with MapBox API. Normalization (3 prepare_data_mapbox.py)
The script performs normalization of extended data imported from MapBox API.

Input data:
* z - zoom level of extended data

Output data:
* Normalized Extended dataset

#### 2.4 Deep Models Training (4 train.py)
The script performs training deep learning models with prepared data.

Input data:
* Trained models

Output data:
* Trained Models

#### 2.5 Graphics Builder (5 graph.py)
The script provides graphing for a comparative assessment of the accuracy of trained models

Input data:
* Trained models

Output data:
* Graphing for a comparative assessment of the accuracy of trained models

#### 2.6 Confusion Matrix Builder (6 confusionmatrix.py)
The script provides calculating of confusion matrix and metrics for a comparative assessment of the accuracy of trained models

Input data:
* Trained models

Output data:
* Confusion matrix and metrix
