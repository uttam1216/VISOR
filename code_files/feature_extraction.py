import json
from scipy.signal import welch
from scipy.stats import kurtosis,skew,pearsonr
from scipy.signal import butter, lfilter, sosfilt
import mne
import math
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from utilities import *

#function to return channels when an electrode is passed
def fetch_ref_electrode(tcp_ref,electrode):
    if 'tcp_ar' in str(tcp_ref):
        ref_electrode = 'EEG '+electrode+'-REF'
    else:
        ref_electrode = 'EEG '+electrode+'-LE'
    return ref_electrode

#functions for finding spectral power from eeg saved in jsons in a given output folder and then get their corresponding heatmap images saved in a given output folder
def process_dict(del_tht_pow):
    processed_dict = {}
    
    for key, value in del_tht_pow.items():
        # Remove the first 4 and last 4 characters from the key as it is a channel in form EEG FP1-REF
        new_key = key[4:-4]
        
        # Multiply the value and round to 3 decimal places
        new_value = round(value * 100000000000, 3)
        
        # Add to the new dictionary
        processed_dict[new_key] = new_value
    
    return processed_dict
    
    
#function to compute spectral power for Delta (0.5-4 Hz) and Theta (4-8 Hz) bands for specific EEG channels.
def compute_spectral_power(raw_edf_file, window_start_time, channels, out_json_file_path, window_duration=8, sfreq=256):
    raw = mne.io.read_raw_edf(raw_edf_file, preload=True)
    sfreq = raw.info["sfreq"]   # get the sampling freq of the EEG signal
    channel_indices = [raw.ch_names.index(ch) for ch in channels]  #get the indices for the specified channels
    data, times = raw[channel_indices, :]  # 'data' is shape (n_channels, n_samples) 
    
    # Define the window start and end sample indices
    start_sample = int(window_start_time * sfreq)
    end_sample = start_sample + int(window_duration * sfreq)
    
    #extract the data for the t-second time interval for the selected channels
    window_data = data[:, start_sample:end_sample]
    
    delta_band = (0.5, 4)   # Delta frequency range in Hz
    theta_band = (4, 8)     # Theta frequency range in Hz
    
    delta_power = {}    # to store power values for each specified channel
    theta_power = {}    # to store power values for each specified channel
    
    # Compute spectral power for each channel
    for i, ch_name in enumerate(channels):
        freqs, psd = welch(window_data[i], sfreq, nperseg=sfreq * 2)  # 2-second segments

        delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
        delta_power[ch_name] = np.trapz(psd[delta_mask], freqs[delta_mask])

        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        theta_power[ch_name] = np.trapz(psd[theta_mask], freqs[theta_mask])

    delta_power = process_dict(delta_power)
    theta_power = process_dict(theta_power)

    with open(out_json_file_path, 'w') as json_file:
        json.dump(delta_power, json_file, indent=4)
    with open(out_json_file_path.replace('delta','theta'), 'w') as json_file:
        json.dump(theta_power, json_file, indent=4)
    return delta_power, theta_power
    
    
#function to build 5*5 matrix in form of EEG Graph 
def build_matrix(E): 
    A = [[0 for _ in range(5)] for _ in range(5)]  # Initialize a 5x5 matrix with zeros
    constant_positions = [(0, 0), (0, 2), (0, 4), (4, 0), (4, 2), (4, 4)]  # assigning constant values to specific positions
    for i, j in constant_positions:
        A[i][j] = 0
    
    # Assign values from the dictionary E to specific positions
    E_positions = {
        (0, 1): 'FP1', (0, 3): 'FP2',
        (1, 0): 'F7',  (1, 1): 'F3',  (1, 2): 'FZ',  (1, 3): 'F4',  (1, 4): 'F8',
        (2, 0): 'T3',  (2, 1): 'C3',  (2, 2): 'CZ',  (2, 3): 'C4',  (2, 4): 'T4',
        (3, 0): 'T5',  (3, 1): 'P3',  (3, 2): 'PZ',  (3, 3): 'P4',  (3, 4): 'T6',
        (4, 1): 'O1',  (4, 3): 'O2'
    }
    for (i, j), label in E_positions.items():
        A[i][j] = E[label]
    return A
    
#function to plot heatmap img of res. 192 * 192 
def plot_heatmap(matrix, out_file_path):
    fig = plt.figure(dpi=100)  # Create figure with DPI
    #plt.title('Heatmap of Matrix A')
    
    # Set vmin and vmax to restrict the color range between 0.38 to 0.99
    plt.imshow(matrix, cmap='coolwarm', vmin=0.53, vmax=0.99, interpolation='nearest')
    
    # Set the color for 0 values to white
    plt.set_cmap('coolwarm')
    cmap = plt.get_cmap()
    cmap.set_under(color='white')
    
    #plt.colorbar(label='Values')
    plt.axis('off')  # Turn off both x and y axes
    
    # Set the size of the figure in inches
    fig.set_size_inches(1.92, 1.92)
    plt.show()
    fig.savefig(out_file_path, dpi=100)
    plt.close(fig)
    
def save_theta_and_delta_entropy_images(df, out_folder, theta_or_delta):
    for row in df.itertuples(index=False): 
        filename = row.uid
        row_dict = dict(zip(df.columns, row))
        if theta_or_delta == 'theta':
           matrix = build_matrix_theta_rev(row_dict)
        else:
           matrix = build_matrix_delta_rev(row_dict)
           out_folder = out_folder.replace('theta','delta')
        out_file_path = out_folder + filename + '.png'
        plot_heatmap_rev(matrix, out_file_path)
        
def calculate_stats(folder_path):
    # Initialize a dictionary to store lists of values for each key
    data = {}

    # Iterate through each JSON file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as json_file:
                dict_data = json.load(json_file)
                # For each key-value pair, append the value to the corresponding list in 'data'
                for key, value in dict_data.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(value)

    # Initialize a dictionary to store the min, max, mean, and median for each key
    stats = {}

    # Calculate the required statistics for each key
    for key, values in data.items():
        values_array = np.array(values)
        stats[f"{key}_min"] = np.min(values_array)
        stats[f"{key}_max"] = np.max(values_array)
        stats[f"{key}_mean"] = np.mean(values_array)
        stats[f"{key}_median"] = np.median(values_array)

    return stats
    
# func to pass a df and get sp framed from its respective columns 
def get_sp_for_df(df,out_file_path, edf_file_path_df_idx, win_start_time_df_idx, uid_df_idx):
    lst_temporal, lst_central, lst_frontal, lst_parietal, lst_occipital = ['T3', 'T5', 'T4', 'T6'], ['C3', 'CZ', 'C4'], ['F3', 'F7', 'FP1', 'FZ', 'FP2', 'F4', 'F8'], ['P3', 'PZ', 'P4'], ['O1', 'O2']
    lst_all_elec = lst_temporal + lst_central + lst_frontal + lst_parietal + lst_occipital
    lst_channels_to_save = [fetch_ref_electrode('tcp_ar',electrode) for electrode in lst_all_elec]
    for row in df.itertuples(index=False):
        # file path, win_start_time, lst_channels_to_save, output_file_path needs to be passed
        compute_spectral_power(row[edf_file_path_df_idx], row[win_start_time_df_idx], lst_channels_to_save, out_file_path.replace('.json',row[uid_df_idx]+'.json'))  #general
        
   
#function to load a patient specific df
def load_pat_spf_df(df,patient):
    df['patient'] = df.apply(lambda row: (row['uid'].split('__')[0]),axis=1)
    df = df.loc[df['patient']==patient]
    df = df.drop("patient", axis=1)
    return df
    
# for patient profile's personal feature statistics function to get min max for the train df
# function to process the train dfs of sz and ns class with addition of a label to last column of the dfs
def get_processed_train_dfs(seizType_prop_onset_test_df, seizType_prop_onset_train_df, term_ns_prop_onset_test_df, term_ns_prop_onset_train_df,st_col_idx=5,ed_col_idx=-1):
    lst_col = list(seizType_prop_onset_train_df.columns[st_col_idx:ed_col_idx])
    # merge the train dfs for 
    seizType_prop_onset_train_df = seizType_prop_onset_train_df[lst_col]
    seizType_prop_onset_train_df['label'] = 1 
    seizType_prop_onset_test_df = seizType_prop_onset_test_df[lst_col]
    seizType_prop_onset_test_df['label'] = 1
    term_ns_prop_onset_train_df = term_ns_prop_onset_train_df[lst_col]
    term_ns_prop_onset_train_df['label'] = 0
    term_ns_prop_onset_test_df = term_ns_prop_onset_test_df[lst_col]
    term_ns_prop_onset_test_df['label'] = 0
    #print(len(seizType_prop_onset_train_df.columns),seizType_prop_onset_train_df.columns)
    #print(len(term_ns_prop_onset_train_df.columns),term_ns_prop_onset_train_df.columns)
    train_test_dwt_df = pd.concat([seizType_prop_onset_train_df, seizType_prop_onset_test_df, term_ns_prop_onset_train_df, term_ns_prop_onset_test_df], axis=0)
    #print('total no. of records in the merged train_dfs of sz and ns is: ',len(train_test_dwt_df))
    ## shuffle the rows and drop index--no shuffling as of 11th Nov 
    #train_test_dwt_df = train_test_dwt_df.sample(frac=1).reset_index(drop=True)
    return train_test_dwt_df
    

def split_features_labels(ttm_df, len_train_sz, len_test_sz, len_train_ns, len_test_ns):
    rec_sz_train_df = ttm_df.iloc[:len_train_sz]
    rec_sz_test_df  = ttm_df.iloc[len_train_sz:len_train_sz + len_test_sz]
    rec_ns_train_df = ttm_df.iloc[len_train_sz + len_test_sz:len_train_sz + len_test_sz + len_train_ns]
    rec_ns_test_df  = ttm_df.iloc[len_train_sz + len_test_sz + len_train_ns:]
    
    # separate features (all columns except the last) and labels (last column) for each dataframe
    X_sz_train = rec_sz_train_df.iloc[:, :-1].values
    y_sz_train = rec_sz_train_df.iloc[:, -1].values
    
    X_sz_test = rec_sz_test_df.iloc[:, :-1].values
    y_sz_test = rec_sz_test_df.iloc[:, -1].values
    
    X_ns_train = rec_ns_train_df.iloc[:, :-1].values
    y_ns_train = rec_ns_train_df.iloc[:, -1].values
    
    X_ns_test = rec_ns_test_df.iloc[:, :-1].values
    y_ns_test = rec_ns_test_df.iloc[:, -1].values

    # Concatenate seizure and non-seizure data for train and test sets
    X_train = np.concatenate([X_sz_train, X_ns_train])
    y_train = np.concatenate([y_sz_train, y_ns_train])
    
    X_test = np.concatenate([X_sz_test, X_ns_test])
    y_test = np.concatenate([y_sz_test, y_ns_test])
    
    return X_train, X_test, y_train, y_test
    

def get_train_test_merged_df(k_top_feat = 250, patient=None):
    seizType_prop_onset_test_df, seizType_prop_onset_train_df, term_ns_prop_onset_test_df, term_ns_prop_onset_train_df = load_vistraPP_6w_2s_dfs()  
    if not patient is None:
       seizType_prop_onset_test_df = load_pat_spf_df(seizType_prop_onset_test_df,patient)
       seizType_prop_onset_train_df = load_pat_spf_df(seizType_prop_onset_train_df,patient)
       term_ns_prop_onset_test_df = load_pat_spf_df(term_ns_prop_onset_test_df,patient) 
       term_ns_prop_onset_train_df = load_pat_spf_df(term_ns_prop_onset_train_df,patient)
    # here apply filter to get patient specific dfs, if it needs to be done separately for each patient
    len_train_sz  = len(seizType_prop_onset_train_df)
    len_test_sz   = len(seizType_prop_onset_test_df)
    len_train_ns  = len(term_ns_prop_onset_train_df)
    len_test_ns   = len(term_ns_prop_onset_test_df)
    train_test_dwt_df = get_processed_train_dfs(seizType_prop_onset_test_df, seizType_prop_onset_train_df, term_ns_prop_onset_test_df, term_ns_prop_onset_train_df)
    return train_test_dwt_df, len_train_sz, len_test_sz, len_train_ns, len_test_ns, k_top_feat 
    
    
# function for VCF top k patient feature selection using XGBoost
def pat_spf_feat_sel():
    train_test_dwt_df, len_train_sz, len_test_sz, len_train_ns, len_test_ns, k = get_train_test_merged_df()
    #k is the number of top features to be retained in feature selection for future use
    X_pca, pca = pca_model(train_test_dwt_df)
    X_train, X_test, y_train, y_test = split_features_labels(train_test_dwt_df, len_train_sz, len_test_sz, len_train_ns, len_test_ns)
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit the scaler on the training data and transform both train and test sets
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Initialize and train the XGBoost model
    model = XGBClassifier()
    model.fit(X_train_normalized, y_train)
    # Make predictions and evaluate
    y_pred = model.predict(X_test_normalized)
    # Explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train_normalized)

    # Plot SHAP summary for the PCA-transformed data
    shap.summary_plot(shap_values, X_train_normalized)

    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    # Get indices of the top k features based on SHAP values
    top_k_shap_indices = np.argsort(mean_shap_values)[-k:][::-1]
    # Get the list of original feature column names from the original dataframe
    feature_columns = train_test_dwt_df.columns[:1283]  # Assuming the first 1283 columns are features
    # Map the indices to column names
    top_k_shap_feature_names = feature_columns[top_k_shap_indices]
    lst_top_k_shap_feature_names = list(top_k_shap_feature_names)
    return lst_top_k_shap_feature_names
    

# returns a list of topmost contributing features aligned with the EEG graph
def fetch_fixed_matrix(lst_top_k_feature_names):
    lst_elec = ['FP1','FP2','F7','F3','FZ','F4','F8','T3','C3','c3','CZ','plv','C4','c4','T4','T5','P3','PZ','P4','T6','O1','O2'] 
    lst_types = ['en','st','mn','pd','sh','kt','sk','_d','_t','_a','_b'] 
    lst_arranged_features = []
    for elec in lst_elec:
        for typ in lst_types:
            for feat in lst_top_k_feature_names:
                if elec in feat:
                   if typ in feat:
                      lst_arranged_features.append(feat)
    
    return lst_arranged_features
    
    
# Function to build the matrix A
def build_matrix(E,E_positions,r=10,c=25):    # To be made 10*25
    # E should be a dict with column_name and value of the df, r is rows count and c s columns count
    # E_positions is a dict with positions(i,j) and labels(column names or keys of teh dict E)
    A = [[0 for _ in range(c+1)] for _ in range(r)]  # Initialize a 5x5 matrix with zeros
    
    for (i, j), label in E_positions.items():
        A[i][j] = E[label]
    
    return A 
    
# function to create eletrode positions in form of a dict
def create_E_positions(lst_arranged_features,rows=10,cols=25):  ## To be made 10*25
    E_positions = {}
    
    # Ensure the number of features matches the grid size
    if len(lst_arranged_features) != rows * cols:
        raise ValueError(f"The list must contain exactly {rows * cols} features.")

    # Populate the E_positions dictionary
    feature_index = 0
    for row in range(rows):
        for col in range(1, cols + 1):
            E_positions[(row, col)] = lst_arranged_features[feature_index]
            feature_index += 1

    return E_positions
    
    
# func to find min max dict
def compute_min_max_dict(stats_df):
    min_max_dict = {}

    for index, row in stats_df.iterrows():
        # Calculate max as mean + 2 * std
        max_value = row['mean'] + 2 * row['std']
        
        # Create dictionary entry for each feature
        min_max_dict[index] = {'min': row['min'], 'max': max_value}
    
    return min_max_dict
    

# this func creates heatmaps against a grid of dwt feature values for patients one by one
def process_heatmaps_250(patient):
    seizType_prop_onset_test_df, seizType_prop_onset_train_df, term_ns_prop_onset_test_df, term_ns_prop_onset_train_df = load_vistraPP_6w_2s_dfs()
    pat_sz_train_df = load_pat_spf_df(seizType_prop_onset_train_df,patient)
    pat_sz_test_df = load_pat_spf_df(seizType_prop_onset_test_df,patient) 
    pat_ns_train_df = load_pat_spf_df(term_ns_prop_onset_train_df,patient) 
    pat_ns_test_df = load_pat_spf_df(term_ns_prop_onset_test_df,patient) 
    
    dataframes = {
    "sz_train_df": pat_sz_train_df,
    "sz_test_df": pat_sz_test_df,
    "ns_train_df": pat_ns_train_df,
    "ns_test_df": pat_ns_test_df
    }
    train_test_dwt_df = get_processed_train_dfs_for_min_max(pat_sz_train_df, pat_ns_train_df)
    # Exclude the last column (label) and calculate the statistics for the 779 feature columns
    stats_df = train_test_dwt_df.iloc[:, :-1].agg(['min', 'max', 'mean', 'median', 'std']).T
    # Assuming stats_df is already calculated as in the previous step
    pat_min_max_dict = compute_min_max_dict(stats_df)
    # Loop over all the dataframes and process each

    #to populate pat_E_positions i.e. eg aaaaarsm_E_positions
    pat_lst_top_k_feature_names = pat_topmost_features_dict[patient]
    # create unique grid structure dict for a specific patient
    pat_E_positions = create_E_positions(pat_lst_top_k_feature_names,10,25)
    try:
       for name, df in dataframes.items():
           lst_dwt_col=list(df.columns[5:-1])
           if 'sz' in name:
               out_file_path = "/media/data/fol/visor/data_files/seiz_prop/dwt_corrDiff_imfs_plv_hmaps_6w_2s/train_test_merged/sz/"
           elif 'ns' in name:
               out_file_path = "/media/data/fol/visor/data_files/seiz_prop/dwt_corrDiff_imfs_plv_hmaps_6w_2s/train_test_merged/ns/"
           # if the following needs to be run only for sz or ns then needs to be shifted right
           for row in df.itertuples(index=False): 
               out_file = out_file_path + row.uid + '.png'
               temp_df = df.loc[df['uid']==row.uid]
               sel_df = temp_df[lst_dwt_col]
               df_dict = sel_df.iloc[0].to_dict()
               #print('df_dict: ',df_dict['aaaaaelb'])
               matrix=build_matrix(df_dict,pat_E_positions,10,25)
               plot_heatmap(matrix, out_file, pat_min_max_dict, pat_E_positions,'plasma',5.7,2.6)
               #break
           #break    
    except Exception as ex:
       print(ex)


# function to find 2 hop connections of the EEG Graph
def find_two_hop_connections(lst_1hop_nn_elec_pairs):
    # Create an empty list to store the two-hop connections
    all_two_hop_connections = []

    # Create a dictionary to hold all connections by electrode for easier lookup
    electrode_map = {}
    
    # Populate the electrode_map with p1-p2 pairs
    for pair in lst_1hop_nn_elec_pairs:
        p1, p2 = pair.split('-')
        if p1 not in electrode_map:
            electrode_map[p1] = []
        if p2 not in electrode_map:
            electrode_map[p2] = []
        
        # Add the connections to both electrodes
        electrode_map[p1].append(p2)
        electrode_map[p2].append(p1)

    # Iterate over each pair in the list
    for pair in lst_1hop_nn_elec_pairs:
        p1, p2 = pair.split('-')
        
        # Find all elements where p2 is connected
        for other_connection in electrode_map[p2]:
            if other_connection != p1:
                # Combine p1 with all other connections of p2
                two_hop_pair = f"{p1}-{other_connection}"
                # Append the two-hop connection to the list if not already present
                if two_hop_pair not in all_two_hop_connections and f"{other_connection}-{p1}" not in all_two_hop_connections:
                    all_two_hop_connections.append(two_hop_pair)
    
    return all_two_hop_connections
    

        
# func to get correlation between two nodes of EEG Graph
def get_corr_bw_elec(edf_file, lst_channels_to_save, elec1, elec2, win_st, win_dur):
    raw = load_notch_filtered_eeg(edf_file.replace('.csv','.edf').replace('.csv_bi','.edf'))
    fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(win_st),
                                                    math.floor(win_st + win_dur))
    #print('fil_sig_for_elec:',fil_sig_for_elec)
    fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
    idx_e1 = lst_channels_to_save.index('EEG ' + elec1 + '-REF')
    #print('idx_e1:',idx_e1)
    idx_e2 = lst_channels_to_save.index('EEG ' + elec2 + '-REF')
    #finding pearson correlation coefficient for the given window length time interval
    temp_corr, p_value = pearsonr(fil_win_dur_inc_onset[idx_e1].ravel(),fil_win_dur_inc_onset[idx_e2].ravel())
    return temp_corr
    
    
# func to find non-seizure mean of corr 
def process_2hop_ns_mean_corr(df,term_ns_prop_onset_train_df):
    lst_mean_ns_corr = []
    for row in df.itertuples(index=False):
        row_mean_ns_corr, ctr = 0, 0
        pat = row.patient
        elec1 = row.nn.split('-')[0]
        elec2 = row.nn.split('-')[-1]
        pat_ns_df = load_pat_spf_df(term_ns_prop_onset_train_df,pat) 
        pat_ns_df = pat_ns_df[['pstrst','non_seiz_onset_win_start_tm','file_path']]
        # find mean of ns for pat_ns_df between electrodes row.nn.split('-')[0] and row.nn.split('-')[-1]
        for idx, rw in pat_ns_df.iterrows():
            ctr+=1
            edf_file = rw.file_path
            win_st = rw.non_seiz_onset_win_start_tm
            row_mean_ns_corr += get_corr_bw_elec(edf_file, lst_channels_to_save, elec1, elec2, win_st, 8)
            print(row_mean_ns_corr)
        lst_mean_ns_corr.append(row_mean_ns_corr/ctr)
        print(lst_mean_ns_corr)
    df['ns_corr_mean'] = lst_mean_ns_corr


# func to compute absolute diff of 2hop and 1hop correlation b/w node pairs of eeg graph
# adds the diff of corr of current window - non-seiz mean for same channel(nn comb.)
def add_sig_corr_diff_feat(df, ns_df, lst_channels_to_save, tem_cen_nn, win_dur, seiz_onset):
    lst_excp_pstrst, ctr = [], 0
    df = df.drop(['label'], axis=1)
    df = df.reindex(df.columns.tolist() + tem_cen_nn, axis=1)
    for index, row in df.iterrows():
        pat_id = row.pstrst.split('$')[0]
        tem_cen_nn_vals = []
        try:
            if 1 == 1:  # not os.path.isfile(output_file_path):
                # i.e. only if file does not already exist, do following
                raw = load_notch_filtered_eeg(row.file_path.replace('.csv','.edf').replace('.csv_bi','.edf'))
                #if seiz_onset == 1:
                   #fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.seiz_onset_win_start_tm),
                   #                                             math.floor(row.seiz_onset_win_start_tm + win_dur))
                fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.start_time),
                                                                math.floor(row.start_time + win_dur))
                #else:
                #   #fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.non_seiz_onset_win_start_tm),
                #   #                                             math.floor(row.non_seiz_onset_win_start_tm + win_dur))
                #   fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.start_time),
                #                                                math.floor(row.start_time + win_dur))
                fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
                for elec_comb in tem_cen_nn:
                    elec_1 = elec_comb.split('-')[0]
                    elec_2 = elec_comb.split('-')[-1]
                    idx_e1 = lst_channels_to_save.index('EEG ' + elec_1 + '-REF')
                    idx_e2 = lst_channels_to_save.index('EEG ' + elec_2 + '-REF')
                    #finding pearson correlation coefficient for the given window length time interval
                    temp_corr, p_value = pearsonr(fil_win_dur_inc_onset[idx_e1].ravel(),
                                                  fil_win_dur_inc_onset[idx_e2].ravel())
                    ns_corr_mean = ns_df.query("patient == @pat_id and (nn == @elec_comb or nn_rev == @elec_comb)")['ns_corr_mean'].values[0]
                    tem_cen_nn_vals.append(abs(round((temp_corr-ns_corr_mean), 2)))
                for i in range(len(tem_cen_nn)):
                    df.at[index, tem_cen_nn[i]] = tem_cen_nn_vals[i]
                df['label'] = seiz_onset
        except Exception as ex:
            print(ex)
            ctr += 1
            lst_excp_pstrst.append(row.pstrst)
            pass
    return df, ctr, lst_excp_pstrst
    
    
#func to add correlation diff columns to a df
# adds the diff of corr of current window - non-seiz mean for same channel(nn comb.)
# the following function is same as add_sig_corr_diff_feat with just the label removed
def add_sig_corr_diff_feat_wo_label(df, ns_df, lst_channels_to_save, tem_cen_nn, win_dur, seiz_onset):
    lst_excp_pstrst, ctr = [], 0
    #df = df.drop(['label'], axis=1)
    df = df.reindex(df.columns.tolist() + tem_cen_nn, axis=1)
    for index, row in df.iterrows():
        pat_id = row.pstrst.split('$')[0]
        tem_cen_nn_vals = []
        try:
            if 1 == 1:  # not os.path.isfile(output_file_path):
                # i.e. only if file does not already exist, do following
                raw = load_notch_filtered_eeg(row.file_path.replace('.csv','.edf'))
                if seiz_onset == 1:
                   fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.seiz_onset_win_start_tm),
                                                                math.floor(row.seiz_onset_win_start_tm + win_dur))
                else:
                   fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, math.floor(row.non_seiz_onset_win_start_tm),
                                                                math.floor(row.non_seiz_onset_win_start_tm + win_dur))
                fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
                for elec_comb in tem_cen_nn:
                    elec_1 = elec_comb.split('-')[0]
                    elec_2 = elec_comb.split('-')[-1]
                    idx_e1 = lst_channels_to_save.index('EEG ' + elec_1 + '-REF')
                    idx_e2 = lst_channels_to_save.index('EEG ' + elec_2 + '-REF')
                    #finding pearson correlation coefficient for the given window length time interval
                    temp_corr, p_value = pearsonr(fil_win_dur_inc_onset[idx_e1].ravel(),
                                                  fil_win_dur_inc_onset[idx_e2].ravel())
                    ns_corr_mean = ns_df.query("patient == @pat_id and (nn == @elec_comb or nn_rev == @elec_comb)")['ns_corr_mean'].values[0]
                    tem_cen_nn_vals.append(abs(round((temp_corr-ns_corr_mean), 2)))
                for i in range(len(tem_cen_nn)):
                    df.at[index, tem_cen_nn[i]] = tem_cen_nn_vals[i]
                #df['label'] = seiz_onset
        except Exception as ex:
            print(ex)
            ctr += 1
            lst_excp_pstrst.append(row.pstrst)
            pass
    return df, ctr, lst_excp_pstrst
    


# function to read edf file and extract/crop image of seizure-onset/non-seizure without and with shift
def save_eeg_channel_images(raw, output_file_path, start_time, end_time):
    try:
        # Set the time period for which you want to save the images
        duration = end_time - start_time
        raw.crop(tmin=start_time, tmax=end_time)
        
        # Disable the interactive toolbar
        plt.rcParams['toolbar'] = 'none'
        
        # Plot the EEG channels, disabling the interactive mode and color bars
        fig = raw.plot(duration=duration, show=False, remove_dc=False, scalings='auto', show_scrollbars=False)
        #plt.show()
        # Save the plot to a file
        fig.savefig(output_file_path)
        
        # Close the figure after saving to avoid displaying it
        plt.close(fig)
        
    except Exception as ex:
       print('Exception came for start & end times: ', start_time, end_time)
       print(ex)
       pass
       
    
#func to save Visual EEG
# for a dataframe of format pstrst	start_time	stop_time	file_path, we save images of specific seizur eonset duration
def save_eeg_extracted_images(df,output_folder,lst_channels_to_save,seiz_non_seiz,win_dur):
    # together means with and without shift
    for row in df.itertuples(index=False):
        output_file_path = output_folder+row[4]+'.png'
        if not os.path.isfile(output_file_path):
           edf_file_path = row.file_path.replace('edf_bi','edf').replace('csv_bi','edf').replace('csv','edf')
           raw = load_notch_filtered_eeg(edf_file_path)
           raw.filter(l_freq=0.5, h_freq=40) 
           raw.pick_channels(lst_channels_to_save)
           save_eeg_channel_images(raw, output_file_path, row[2], row[2]+win_dur) 
           
           
           
# func to compute energy of imfs
def compute_energy(imfs):
    energy = np.sum(imfs**2, axis=1)  # Sum of squares for each IMF
    return energy
    
    
#func to get filtered eeg segments
def get_filtered_eeg_segments(edf_file_path, lst_channels_to_save, start_time, window_duration, lowcut, highcut):
    """
    Loads an 8-second segment of EEG data from specified lst_channels_to_save in an EDF file, applies a bandpass filter, 
    and returns the filtered signals for each channel.
    
    Parameters:
        edf_file_path (str): Path to the EDF file.
        start_time (float): Start time of the segment in seconds.
        window_duration (int): Duration of the segment in seconds. Default is 8 seconds.
        lst_channels_to_save (list): List of channel names to retrieve and filter. Default is ['EEG C3-REF', 'EEG C4-REF'].
        lowcut (float): Low cutoff frequency for bandpass filter. Default is 8 Hz.
        highcut (float): High cutoff frequency for bandpass filter. Default is 12 Hz.

    Returns:
        filtered_signals (dict): Dictionary of filtered EEG segments for each specified channel.
    """
    
    # Open the EDF file and read signal
    with pyedflib.EdfReader(edf_file_path) as f:
        # Retrieve sampling rate from the first EEG signal
        sampling_rate = f.getSampleFrequency(0)
        
        # Dictionary to store the filtered signal for each channel
        filtered_signals = {}
        
        for channel_name in lst_channels_to_save:
            try:
                # Get the index of the channel based on its name
                channel_idx = f.getSignalLabels().index(channel_name)
                
                # Read the raw EEG signal for this channel
                eeg_signal = f.readSignal(channel_idx)
                
                # Calculate the number of data points in the desired window
                start_index = int(start_time * sampling_rate)
                end_index = start_index + int(window_duration * sampling_rate)
                
                # Extract the specified segment
                raw_segment = eeg_signal[start_index:end_index]
                
                # Define a bandpass Butterworth filter
                sos = butter(N=4, Wn=[lowcut, highcut], btype='bandpass', fs=sampling_rate, output='sos')
                
                # Apply the bandpass filter to the raw EEG segment
                filtered_signal = sosfilt(sos, raw_segment)
                
                # Store the filtered signal in the dictionary
                filtered_signals[channel_name] = filtered_signal
            
            except ValueError:
                print(f"Channel '{channel_name}' not found in the EDF file.")
    
    return filtered_signals



#function to fetch imfs for an electrode e
def get_e_lf_hf_imfs(edf_file_path, lst_channels_to_save, st, win_dur, e1):

    try: 
       fil_sig_lf_d = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 0.5, 4)
       fil_sig_lf_t = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 4, 8)
       fil_sig_lf_a = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 8, 12)
       fil_sig_hf_b = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 12, 30)
       
       # Perform EMD on electrode e's signal where e is of form eg - 'EEG C3-REF'
       emd = EMD(extrema_detection='parabol') # extrema_detection='parabol' parabol can help by smoothening out noise while preserving important oscillatory patterns and it may provide more robust identification of seizure-related oscillations
       IMFs_e1_lf_d = emd(fil_sig_lf_d[e1])

       IMFs_e1_lf_t = emd(fil_sig_lf_t[e1])

       IMFs_e1_lf_a = emd(fil_sig_lf_a[e1])

       IMFs_e1_hf_b = emd(fil_sig_hf_b[e1])

    except Exception as ex:
       IMFs_e1_lf_d,IMFs_e1_lf_t,IMFs_e1_lf_a,IMFs_e1_hf_b=[],[],[],[]
    finally:
       return IMFs_e1_lf_d, IMFs_e1_lf_t, IMFs_e1_lf_a, IMFs_e1_hf_b
       
       
#Compute energy for IMFs of e1
def get_imfs_based_energy(edf_file_path, lst_channels_to_save, st, win_dur):
    i_e1_d, i_e1_t, i_e1_a, i_e1_b = get_e_lf_hf_imfs(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur,e1)
    if len(i_e1_d) > 0:
       energy_i_e1_d = compute_energy(i_e1_d)
       energy_i_e1_t = compute_energy(i_e1_t)
       energy_i_e1_a = compute_energy(i_e1_a)
       energy_i_e1_b = compute_energy(i_e1_b)
    else:
       energy_i_e1_d, energy_i_e1_t, energy_i_e1_a, energy_i_e1_b = [],[],[],[]
    
    return energy_i_e1_d, energy_i_e1_t, energy_i_e1_a, energy_i_e1_b
    
    
 # func to get phase synschronization plv values for a pair of nodes (electrodes)   
# Phase Locking Value (PLV): Quantifies synchronization between electrodes, e.g., between e1 and e2 during a seizure.
def get_plv(edf_file_path, lst_channels_to_save, st, win_dur, e1, e2):
    # eg e1= 'EEG C3-REF'

    try: 
       fil_sig_lf_d = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 0.5, 4)
       fil_sig_lf_t = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 4, 8)
       fil_sig_lf_a = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 8, 12)
       fil_sig_hf_b = get_filtered_eeg_segments(edf_file_path.replace('.csv','.edf').replace('.csv_bi','.edf'), lst_channels_to_save, st, win_dur, 12, 30)
       
       # compute the analytical signal (Hilbert transform) for e1 and e2
       # and then compute the instantaneous phase for each signal
       phase_e1_d = angle(hilbert(fil_sig_lf_d[e1]))
       phase_e2_d = angle(hilbert(fil_sig_lf_d[e2]))
       phase_e1_t = angle(hilbert(fil_sig_lf_t[e1]))
       phase_e2_t = angle(hilbert(fil_sig_lf_t[e2]))
       phase_e1_a = angle(hilbert(fil_sig_lf_a[e1]))
       phase_e2_a = angle(hilbert(fil_sig_lf_a[e2]))
       phase_e1_b = angle(hilbert(fil_sig_hf_b[e1]))
       phase_e2_b = angle(hilbert(fil_sig_hf_b[e2]))
       
       # Compute phase difference between e1 and e2
       phase_diff_d = phase_e1_d - phase_e2_d
       phase_diff_t = phase_e1_t - phase_e2_t
       phase_diff_a = phase_e1_a - phase_e2_a
       phase_diff_b = phase_e1_b - phase_e2_b
       
       # Compute PLV (Phase Locking Value) for each of teh three bands
       plv_d = abs(np.mean(np.exp(1j * phase_diff_d)))
       plv_t = abs(np.mean(np.exp(1j * phase_diff_t)))
       plv_a = abs(np.mean(np.exp(1j * phase_diff_a)))
       plv_b = abs(np.mean(np.exp(1j * phase_diff_b)))
    except Exception as ex:
       plv_d, plv_t, plv_a, plv_b = 0,0,0,0
    #print('Phase Locking Value (PLV) between e1 and e2 for delta(0.5-4), theta(4-8), alpha(8-12) and beta(12-30) bands are:', plv_d, plv_t, plv_a, plv_b)
    return plv_d, plv_t, plv_a, plv_b
    

### function to add imf based hilbert energy feature (of first 4 imfs) corresponding to a channel & 4 phase synchronization features corresponding to a node pair in a df for each freq band)
def add_e_imfs_plv(df,seiz):
    #this function to be run when the last column if the input df contains the label
    new_df = df.iloc[:,:-1]
    lst_i_e1_d1, lst_i_e1_d2, lst_i_e1_d3, lst_i_e1_d4 = [],[],[],[]
    lst_i_e1_t1, lst_i_e1_t2, lst_i_e1_t3, lst_i_e1_t4 = [],[],[],[]
    lst_i_e1_a1, lst_i_e1_a2, lst_i_e1_a3, lst_i_e1_a4 = [],[],[],[] 
    lst_i_e1_b1, lst_i_e1_b2, lst_i_e1_b3, lst_i_e1_b4 = [],[],[],[]
    lst_uid, lst_plv_d, lst_plv_t, lst_plv_a, lst_plv_b = [],[],[],[],[]
    for row in new_df.itertuples(index=False):
        #current df format of row nos row[1] - onset, 2-win_st, 3-file, 4-uid
        #format of input in function get_imfs_based_energy(edf_file_path, lst_channels_to_save, st, win_dur)
        i_e1_d_en, i_e1_t_en, i_e1_a_en, i_e1_b_en = get_imfs_based_energy(row[3], lst_channels_to_save, row[2], 6)
        plv_d, plv_t, plv_a, plv_b = get_plv(row[3], lst_channels_to_save, row[2], 6)
        
        lst_uid.append(row[4])
        
        lst_i_e1_d1.append(get_val_wE(i_e1_d_en,0))
        lst_i_e1_d2.append(get_val_wE(i_e1_d_en,1))
        lst_i_e1_d3.append(get_val_wE(i_e1_d_en,2))
        lst_i_e1_d4.append(get_val_wE(i_e1_d_en,3))     
        lst_i_e1_t1.append(get_val_wE(i_e1_t_en,0))
        lst_i_e1_t2.append(get_val_wE(i_e1_t_en,1))
        lst_i_e1_t3.append(get_val_wE(i_e1_t_en,2))
        lst_i_e1_t4.append(get_val_wE(i_e1_t_en,3))
        lst_i_e1_a1.append(get_val_wE(i_e1_a_en,0))
        lst_i_e1_a2.append(get_val_wE(i_e1_a_en,1))
        lst_i_e1_a3.append(get_val_wE(i_e1_a_en,2))
        lst_i_e1_a4.append(get_val_wE(i_e1_a_en,3))
        lst_i_e1_b1.append(get_val_wE(i_e1_b_en,0))
        lst_i_e1_b2.append(get_val_wE(i_e1_b_en,1))
        lst_i_e1_b3.append(get_val_wE(i_e1_b_en,2))
        lst_i_e1_b4.append(get_val_wE(i_e1_b_en,3))
        
        lst_plv_d.append(round(plv_d,3))
        lst_plv_t.append(round(plv_t,3))
        lst_plv_a.append(round(plv_a,3))
        lst_plv_b.append(round(plv_b,3))
    
    new_df['i_e1_d1'] = lst_i_e1_d1
    new_df['i_e1_d2'] = lst_i_e1_d2
    new_df['i_e1_d3'] = lst_i_e1_d3
    new_df['i_e1_d4'] = lst_i_e1_d4
    new_df['i_e1_t1'] = lst_i_e1_t1
    new_df['i_e1_t2'] = lst_i_e1_t2
    new_df['i_e1_t3'] = lst_i_e1_t3
    new_df['i_e1_t4'] = lst_i_e1_t4
    new_df['i_e1_a1'] = lst_i_e1_a1
    new_df['i_e1_a2'] = lst_i_e1_a2
    new_df['i_e1_a3'] = lst_i_e1_a3
    new_df['i_e1_a4'] = lst_i_e1_a4
    new_df['i_e1_b1'] = lst_i_e1_b1
    new_df['i_e1_b2'] = lst_i_e1_b2
    new_df['i_e1_b3'] = lst_i_e1_b3
    new_df['i_e1_b4'] = lst_i_e1_b4
    

    new_df['plv_d'] = lst_plv_d
    new_df['plv_t'] = lst_plv_t
    new_df['plv_a'] = lst_plv_a
    new_df['plv_b'] = lst_plv_b 
    new_df['label'] = seiz 
    return new_df
    
    
    
# function to apply Discrete Wavelet Transform of a certain DB (e.g. 4 or 10) and level (e.g. 4) to the filtered eeg signal
def apply_wavelet_transform(signal, db, level):
    # Daubechies 10 wavelet transform with given level
    coeffs = pywt.wavedec(signal, db, level=level)
    #print('coeffs after apply_wavelet_transform function are: ', coeffs)
    # cA4, cD4, cD3, cD2, cD1 = coeffs
    return coeffs
    
    
# function to collect DWT related features of a filtered EEG signal data, returns list of 35 lists
def get_dwt_features(raw, out_fol, lst_channels_to_save, start_time, end_time, win_dur):
    try:
       fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, out_fol, lst_channels_to_save, start_time, end_time) #even 0 can be passed as start_time but in that case check what should be passed as end_time
       #fil_sig_for_elec = get_filtered_eeg_segments(edf_file_path, lst_channels_to_save, start_time, win_dur, 0.5, 40)
       # for all channels, take last win_dur time interval number of seconds data points for applying debauchies
       fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
       #fil_win_dur_inc_onset = fil_sig_for_elec
       # Apply wavelet transform DB10 Level 4 only for getting permuatation entropy
       wc_db10l4 = apply_wavelet_transform(fil_win_dur_inc_onset, 'db10', 4)
       pd1, pd2, pd3, pd4, pd5 = get_permut_entropy(wc_db10l4[4], wc_db10l4[3], wc_db10l4[2], wc_db10l4[1], wc_db10l4[0])
       # Apply wavelet transform DB4 Level 4 for other entropy and energy features
       wc_db4l4 = apply_wavelet_transform(fil_win_dur_inc_onset, 'db4', 4)
       #                                        D1,          D2,          D3,          D4,          A4
       sh1, sh2, sh3, sh4, sh5 = get_shan_entropy(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       sk1, sk2, sk3, sk4, sk5 = get_skeww(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       kt1, kt2, kt3, kt4, kt5 = get_kurtosiss(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       mn1, mn2, mn3, mn4, mn5 = get_meann(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       st1, st2, st3, st4, st5 = get_stdd(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       en1, en2, en3, en4, en5 = get_energy(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
       X = [pd1, pd2, pd3, pd4, pd5, sh1, sh2, sh3, sh4, sh5, sk1, sk2, sk3, sk4, sk5, kt1, kt2, kt3, kt4, kt5,
            mn1, mn2, mn3, mn4, mn5, st1, st2, st3, st4, st5, en1, en2, en3, en4, en5]
       # each item of this list has 19 values in each item i.e. 1 value for each electrode
    except Exception as ex:
       print(ex)
    return X
    
    
# function to get combination of features per electrode
def get_lst_chnlwise_features(lst_channels_to_save):
    # permutation entropy, shannon entropy, skewness, kurtosis, mean, standard deviation, energy
    lst_features = ['pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'sh1', 'sh2', 'sh3', 'sh4', 'sh5', 'sk1', 'sk2', 'sk3', 'sk4',
                    'sk5',
                    'kt1', 'kt2', 'kt3', 'kt4', 'kt5', 'mn1', 'mn2', 'mn3', 'mn4', 'mn5', 'st1', 'st2', 'st3', 'st4',
                    'st5',
                    'en1', 'en2', 'en3', 'en4', 'en5']
    # constructing a list of new features to be made as columns of each of the dfs to store their values
    lst_chnlwise_feat = []  # lst_channels_to_save is total electrodes being considered, max=19
    for feature in lst_features:
        for elec_ref in lst_channels_to_save:
            elec = elec_ref.split(' ')[-1].split('-')[0]
            lst_chnlwise_feat.append(elec + '_' + feature)
    return lst_chnlwise_feat
    
    
# function to add features upon the train and test set dfs
def add_dwt_features_to_df(df, lst_channels_to_save, lst_chnlwise_feat, win_dur, seiz_onset):
    lst_excp, temp_uid = [], ''
    # first we add all 665 columns as empty columns and later update values row-wise
    df = df.reindex(df.columns.tolist() + lst_chnlwise_feat, axis=1)
    for index, row in df.iterrows():
        temp_uid = row.uid
        raw = load_notch_filtered_eeg(row.file_path.replace('.csv','.edf').replace('.csv_bi','.edf')) #commented on 8th Nov
        try:
            #commented on 8th Nov
            F = get_dwt_features(raw, 'none', lst_channels_to_save, math.floor(row.win_start_tm), math.floor(row.win_start_tm + win_dur), win_dur)
            lst_chnlwise_feat_val = []
            for lst_item in F:
                for ch_feat_val in lst_item:
                    lst_chnlwise_feat_val.append(ch_feat_val)
            for i in range(len(lst_chnlwise_feat)):
                df.at[index, lst_chnlwise_feat[i]] = lst_chnlwise_feat_val[i]
        except Exception as ex:
            lst_excp.append(temp_uid)
            pass
    df['label'] = seiz_onset
    return df, lst_excp
    
    
# function to get dwt features added in the dfs
def get_dwt_added_features_in_dfs(fnsz_onset_features_test_df, fnsz_onset_features_train_df, non_seiz_features_test_df,
                                  non_seiz_features_train_df, lst_channels_to_save, output_folder_path, win_dur):
    # list of features based on dwt column names per electrode for all selected electrodes
    lst_chnlwise_feat = get_lst_chnlwise_features(lst_channels_to_save)
    
    # get dfs with added dwt features, additionally get exception description if got any while making the features
    # sz test
    fnsz_onset_dwt_feat_test_df, lst_excp_seiz_test = add_dwt_features_to_df(fnsz_onset_features_test_df,
                                                                             lst_channels_to_save, lst_chnlwise_feat,
                                                                             win_dur, 1)
    # due to exceptions, some columns may get NaN values, so we need to remove such rows from the dfs
    valid_fnsz_onset_dwt_feat_test_df = fnsz_onset_dwt_feat_test_df.loc[
        ~fnsz_onset_dwt_feat_test_df['uid'].isin(lst_excp_seiz_test)]  # 1 record with NaN removed
    # save the dwt features for each test train dfs of the seizure non-seizure
    valid_fnsz_onset_dwt_feat_test_df.to_pickle(output_folder_path + 'w6_s2_sz_test_dwt.pkl')

    #sz train
    fnsz_onset_dwt_feat_train_df, lst_excp_seiz_train = add_dwt_features_to_df(fnsz_onset_features_train_df, lst_channels_to_save, lst_chnlwise_feat, win_dur, 1)
    valid_fnsz_onset_dwt_feat_train_df = fnsz_onset_dwt_feat_train_df.loc[
        ~fnsz_onset_dwt_feat_train_df['uid'].isin(lst_excp_seiz_train)]  # 8 record with NaN removed
    valid_fnsz_onset_dwt_feat_train_df.to_pickle(output_folder_path + 'w6_s2_sz_train_dwt.pkl')

    #ns test
    non_seiz_dwt_feat_test_df, lst_excp_non_seiz_test = add_dwt_features_to_df(non_seiz_features_test_df, lst_channels_to_save, lst_chnlwise_feat, win_dur, 0)
    valid_non_seiz_dwt_feat_test_df = non_seiz_dwt_feat_test_df.loc[
        ~non_seiz_dwt_feat_test_df['uid'].isin(lst_excp_non_seiz_test)]  # 81 record with NaN removed
    valid_non_seiz_dwt_feat_test_df.to_pickle(output_folder_path + 'w6_s2_ns_test_dwt.pkl')

    #ns train
    non_seiz_dwt_feat_train_df, lst_excp_non_seiz_train = add_dwt_features_to_df(non_seiz_features_train_df,lst_channels_to_save, lst_chnlwise_feat, win_dur, 0)
    valid_non_seiz_dwt_feat_train_df = non_seiz_dwt_feat_train_df.loc[
        ~non_seiz_dwt_feat_train_df['uid'].isin(lst_excp_non_seiz_train)]  # 213 record with NaN removed
    valid_non_seiz_dwt_feat_train_df.to_pickle(output_folder_path + 'w6_s2_ns_train_dwt.pkl')
    
    return valid_fnsz_onset_dwt_feat_test_df, valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df
    
    


