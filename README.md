# GeoSystemNet




#### 1 Data preparation
The script performs preliminary processing of ERS data from the Sentinel-2 satellite: normalization and export to a file for further use.

Input data:
* sentinel_source_dir - geotiff image storage path 
* processed_data_dir - directory for generated data

Output data:
* List of normalized data for test and training, saved in file

#### 2 Data augmention with MapBox API
The script performs normalized Sentinel-2 data augmention with MapBox API and export results to a file for further use.
mapbox_token - your MapBox token
z - zoom level of extended data

Output data:
* Extended dataset

#### 3 Data augmention with MapBox API. Normalization
The script performs normalization of extended data imported from MapBox API.
z - zoom level of extended data

Output data:
* Normalized Extended dataset

#### 4 Deep Models Training
The script performs training deep learning models with prepared data.

Output data:
* Trained Models

#### 5 Graphics Builder
The script provides graphing for a comparative assessment of the accuracy of trained models

Output data:
* Graphing for a comparative assessment of the accuracy of trained models

#### 6 Confusion Matrix Builder
The script provides calculating of confusion matrix and metrics for a comparative assessment of the accuracy of trained models

Output data:
* Confusion matrix and metrix
