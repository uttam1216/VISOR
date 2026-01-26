# VISOR
VIsual Seizure Onset detection peRsonalized for epilepsy patients <br>

Disclaimer: This software is not intended for placement on the market but is intended exclusively for research applications.

<b>Notes: </b> <br>
1) Due to a Non-Disclosure Agreement (NDA) signed, no part of the TUH EEG Seizure Data used in this work has been shared in any form, and thus it is not possible to share the original notebook with the EDA and Model results of this work. However, the full data can be downloaded from - https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v2.0.0/   after signing a NDA with the data provider. <br>
2) Given the data, by following the below instructions, the codes provided can be run to exactly reach the results depicted in the submitted paper. <br>

<b> Repository Usage: </b> <br>
Installation: All Python packages can be installed by running the following command in your terminal- <br>
<b> <i> pip install -r requirements.txt </i> </b> <br>

<b> Data Preprocessing: </b> <br> Once you have the TUH EEG Seizure corpus at a location -<input_folder_path> and you are willing to keep all the processed data files at a location <output_folder_path>, then the data can be preprocessed by running the following command in your terminal- <br>
<b> <i> python3 data_preprocessing.py input_folder_path output_folder_path </i> </b> <br>

<b> Feature Extraction: </b> <br> Once you have the preprocessed data, the respective features can be extracted from it by running the following command in your terminal: <br>
<b> <i> python3 feature_extraction.py train_test_folder_path output_folder_path time_interval_window_length eeg_graph_nodes </i> </b> <br>
where train_test_folder_path is the path where you have the preprocessed data with train test files; output_folder_path is the path where you want to keep the files with extracted features; time_interval_window_length is an optional time interval window (6sec, 8sec, 10sec, 12sec or any time interval window (>=6 sec) of your choice for which you would like to run this model, default is 6 sec; and eeg_graph_nodes is the list of all nodes for which you would like to run our model, e.g. ['T3', 'T5', 'T4', 'T6'] or ['C3', 'CZ', 'C4'] or ['',''...''] which exists in our EEG Graph based on standard international 10-20 system for electrodes placement on human scalp for EEG recording. <br> 

There are many functions in this <b> feature_extraction.py </b> and also in <b> utilities.py </b> script which takes in a pandas dataframe and returns the dataframe after enriching it with desired features. These can be better called and visualized from a jupyter notebook. While our original data processing and feature extraction work was done on jupyter notebook, due to data security reasons exact notebook cannot be shared but with all the functions described, a notebook can be easily made depicting exact flow of data for model input pipeline.  <br> 

<b> Detection Model: </b> <br> Once you have the features extracted, the model can be run by running the following command in the terminal: <br>
<b> <i> python3 model.py train_dataset_path test_dataset_path num_epochs </i> </b> <br>
such that train_dataset_path is the path where training data is kept, test_dataset_path is the location of the test dataset, and num_epochs is an optional parameter giving the number of epochs for which you want to train the model. To change other parameters of the model, code can be changed in model.py for further research and experiments. <br>

<b> code_files </b> contains Python codes for data_preprocessing, feature extraction, model and an utilities file with many common functions.  <br>
<b> data_files </b> contains a Readme.txt file with instructions. <br>

