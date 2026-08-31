# VISOR
VIsual Seizure Onset detection peRsonalized for epilepsy patients <br>

The code in this repository is an implementation of our published approach: <br>
Uttam Kumar, Ran Yu, Michael Wenzel, and Elena Demidova. “VISOR:VIsual Seizure Onset Detection PeRsonalized for Epilepsy Patients”. In: Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining, PAKDD 2025. LNAI. Springer, 2025, pp. 482–494. https://doi.org/10.1007/978-981-96-8173-0_38 <br>
If you use the source code in your research, please cite our paper.

<b>Notes: </b> <br>
1) This repository does not contain any data. We used TUH Seizure (TUSZ) dataset v2.0.0 for our research. To access this data, please directly contact the TUH Seizure (TUSZ) dataset author whose reference is: "Shah, V., et al.: The temple university hospital seizure detection corpus. Front. Neuroinform. 12, 83 (2018)" <br>
2) Given the data, by following the below instructions, the codes provided can be run to exactly reach the results depicted in the submitted paper. <br>

<b> Repository Usage: </b> <br>
Installation: All Python packages can be installed by running the following command in your terminal- <br>
<b> <i> pip install -r requirements.txt </i> </b> <br>

<b> Loading TUH  Seizure Annotations: </b> <br>
The following python command will load all seizure annotations needed for data processing in a postgresql database (install if not existing) for easy and faster data processing later:
<b> <i> python3 load_tuh_metadata_sqlalchemy.py tuh_eeg_sz_filepath  </i> </b> <br>
where an example format of tuh_eeg_sz_filepath (as present in my disk after download from the TUH dataset authors) is: "/media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*"

<b> Data Preprocessing: </b> <br> Once you have the TUH EEG Seizure corpus along with the seizure annotated times present in the freshly created postgresql db achieved via previous script run, by running the following command, eeg segments for seizure-onsets and non-seizure gets created at a location eeg_filepath <br>
<b> <i> python3 generate_fnsz_eeg_segments.py eeg_filepath </i> </b> <br>
where eeg_filepath is of format src+'eeg_segments', src='/media/data/ukumar/iBehave/data_files/feb25/' or wherever you want to make a parent folder for all processes; additionally the length of eeg segments required (in seconds) and the stride for moving window (in seconds) can be given along after the input_eeg_filepath argument. By default these values are set to have 8 sec and 1 sec respectively. <br>
After this we normalize the generated eeg_segments on per-channel per-patient basis using the following command giving a location for saving the normalized segments. <br> 
<b> <i> python3 normalize_eeg_segments.py eeg_filepath </i> </b> <br>
where eeg_filepath is of format src+'eeg_segments'. This saves the normalized eeg segments in same parent folder src, where the earlier eeg_segments existed with the name of folder as normalized_eeg_segments and is now ready for creating train test split and/or all types of feature extraction <br>


<b> Feature Extraction: </b> <br> Once you have the normalized EEG segments at a location input_folder_path that mainly refers to location of <normalized_eeg_segments> and you are willing to keep all the processed data files at a location <output_folder_path>, respective features can be extracted from it by running the following command in your terminal: <br>
<b> <i> python3 feature_extraction.py train_test_folder_path output_folder_path time_interval_window_length eeg_graph_nodes </i> </b> <br>
where train_test_folder_path is the path where you have the preprocessed data with train test files; output_folder_path is the path where you want to keep the files with extracted features; time_interval_window_length is an optional time interval window (6sec, 8sec, 10sec, 12sec or any time interval window (>=6 sec) of your choice for which you would like to run this model, default is 6 sec; and eeg_graph_nodes is the list of all nodes for which you would like to run our model, e.g. ['T3', 'T5', 'T4', 'T6'] or ['C3', 'CZ', 'C4'] or ['',''...''] which exists in our EEG Graph based on standard international 10-20 system for electrodes placement on human scalp for EEG recording. <br> 

There are many functions in this <b> feature_extraction.py </b> and also in <b> utilities.py </b> script which takes in a pandas dataframe and returns the dataframe after enriching it with desired features. These can be better called and visualized from a jupyter notebook. While our original data processing and feature extraction work was done on jupyter notebook, due to data security reasons exact notebook cannot be shared but with all the functions described, a notebook can be easily made depicting exact flow of data for model input pipeline.  <br> 

<b> Detection Model: </b> <br> Once you have the features extracted, the model can be run by running the following command in the terminal: <br>
<b> <i> python3 model.py train_dataset_path test_dataset_path num_epochs </i> </b> <br>
such that train_dataset_path is the path where training data is kept, test_dataset_path is the location of the test dataset, and num_epochs is an optional parameter giving the number of epochs for which you want to train the model. To change other parameters of the model, code can be changed in model.py for further research and experiments. <br> 

<b> code_files </b> contains Python codes for data_preprocessing, feature extraction, model and an utilities file with many common functions.  <br>
<b> data_files </b> contains a Readme.txt file with instructions. <br>

<b>Disclaimer:</b>  This software is published only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market.

<b>Affiliations: </b> The source code is being published in affiliation with the University of Bonn (https://www.uni-bonn.de/de). 

<b>Acknowledgements: </b> This work was partially funded by the Ministry of Culture and Science of the State of North Rhine-Westphalia, Germany (“iBehave”)  and the Lamarr Institute for Machine Learning and Artificial Intelligence (https://lamarr-institute.org/).

