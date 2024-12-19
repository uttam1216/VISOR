import mne
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
from utilities import *


# function to insert all data from TUH Seizure corpus to postgreSQL database in local for better EDA and data processing
def insert_data_in_postgres_tables(pat_ses_tcps):
    all_ctr = 0
    num_rows_inserted = 0
    num_rows_inserted_chnl_ann = 0
    num_rows_inserted_term_ann = 0
    try:
        for f in glob.glob(pat_ses_tcps):
            fol_id = f.split('/')[-1]
            pat_ses_tcps = f + '/*'
            for fin in glob.glob(pat_ses_tcps):
                pat_id = fin.split('/')[-1]
                ses_tcps = fin + '/*'
                for pat_fin in glob.glob(ses_tcps):
                    full_ses_id = pat_fin.split('/')[-1]
                    arr = full_ses_id.split('_')
                    ses_id = arr[0]
                    ses_date = arr[1] + '-' + arr[2] + '-' + arr[3]
                    ses_tcps = pat_fin + '/*'
                    cnt_tcps = 0
                    lst_dis_tcps = []
                    for pat_ses_fin in glob.glob(ses_tcps):
                        tcp_id = pat_ses_fin.split('/')[-1]
                        end_files = pat_ses_fin + '/*'
                        #we skip first 5 rows as they contain comments and metadata
                        for end_file in glob.glob(end_files):
                            t_id = end_file.split('/')[-1].split('.')[0].split('_')[-1]
                            file_type = ''
                            if end_file.split('.')[-1] == 'csv':
                                file_type = 'channel_annotation'
                                channel_df = pd.read_csv(end_file, skiprows=5)
                                for row in channel_df.itertuples(index=False):
                                    # insert data in channel_annotations table
                                    num_rows_inserted_chnl_ann += insert_in_channel_annotations(row.channel,
                                                                                                row.start_time,
                                                                                                row.stop_time,
                                                                                                row.label,
                                                                                                row.confidence, pat_id,
                                                                                                ses_id, t_id, tcp_id,
                                                                                                end_file, ses_date)
                            elif end_file.split('.')[-1] == 'csv_bi':
                                file_type = 'term_annotation'
                                term_df = pd.read_csv(end_file, skiprows=5)
                                for row in term_df.itertuples(index=False):
                                    # insert data in term_annotations table
                                    num_rows_inserted_term_ann += insert_in_term_annotations(row.channel,
                                                                                             row.start_time,
                                                                                             row.stop_time, row.label,
                                                                                             row.confidence, pat_id,
                                                                                             ses_id, t_id, tcp_id,
                                                                                             end_file, ses_date)
                            elif end_file.split('.')[-1] == 'edf':
                                file_type = 'edf'

                            # insert data in patient table
                            num_rows_inserted += insert_in_patient(fol_id, pat_id, ses_id, tcp_id, t_id, ses_date,
                                                                   end_file, file_type)
                            all_ctr += 1
    except Exception as ex:
        print(ex)


# function for data clean up and also to add some statistical columns for further EDA
def process_data_cleanup_and_stats(fnsz_channel_df, term_df, output_folder_path):
    fnsz_channel_df = fnsz_channel_df.loc[fnsz_channel_df['start_time'] < fnsz_channel_df['stop_time']]
    # followig removes faulty patients records having seizure stop time earlier than seizure start time for same seizure e.g. issue for one such file has been reported by us to TUH EEG dataset provider by e-mail
    fnsz_channel_df = fnsz_channel_df.loc[fnsz_channel_df['start_time'] < fnsz_channel_df['stop_time']]
    fnsz_stats_df = get_seiz_stats_df(fnsz_channel_df)
    fnsz_stats_df['patient'] = fnsz_stats_df['pstr'].apply(lambda x: x.split('$')[0])
    fnsz_stats_df['ar_le'] = fnsz_stats_df.apply(lambda row: 'le' if 'le' in (row['pstr'].split('$')[-1]) else 'ar',
                                                 axis=1)
    # for each seiz_duration_top_chnl, find median and min and also at end for all
    fnsz_stats_df['dur_all_chnl_median'] = fnsz_stats_df['seiz_duration_all_chnl'].apply(lambda x: statistics.median(x))
    fnsz_stats_df['dur_all_chnl_min'] = fnsz_stats_df['seiz_duration_all_chnl'].apply(lambda x: min(x))
    # EDA done in another notebook, could not be shared due to presence of data on it and NDA with dataset provider
    # median duration of seizure over all session channels  46.6
    # min. duration of seizure over all session channels  1.85
    # mean duration of seizure over all session channels 58.809522703273494
    # remove seizures of duration less than equal to 3 seconds considering them as noise
    fnsz_stats_df = fnsz_stats_df.loc[fnsz_stats_df['dur_all_chnl_min'] > 3]
    # taking data with only Averaged reference montage and removing linked ear reference montage data
    fnsz_stats_df = fnsz_stats_df.loc[fnsz_stats_df['ar_le'] == 'ar']
    ## call function to frame a df with valid seizure onset times with-
    # contents: pstr, lst_seiz_start_times, unq_chnls_in_val_cnt having seiz_duration_all_chnl < 3 sec removed
    fnsz_onset_df = frame_seiz_onset_df(fnsz_stats_df, fnsz_channel_df, 'fnsz')
    # call function to formulate final seizure df with start and stop times
    final_fnsz_seiz_onset_df = formulate_final_seiz_onset_df(fnsz_onset_df, 'fnsz')
    # next we filter out this dataframe to have patients for whom normal control(non-seizure sessions) data is present
    final_fnsz_seiz_onset_df['patient_id'] = final_fnsz_seiz_onset_df.pstrst.apply(lambda x: x.split('$')[0])
    lst_unq_pat_with_fnsz_ar = list(final_fnsz_seiz_onset_df['patient_id'].unique())
    bckg_term_df = term_df.loc[(term_df['label'] == 'bckg') & (term_df['patient_id'].isin(lst_unq_pat_with_fnsz_ar))]
    bckg_term_df['ref_ar'] = bckg_term_df.tcp_ref.apply(
        lambda x: 1 if (x.split('_')[1] + '_' + x.split('_')[2]) == 'tcp_ar' else 0)
    bckg_term_ar_df = bckg_term_df.loc[(bckg_term_df['ref_ar'] == 1)]
    lst_pat_fnsz_bckg_ar = list(bckg_term_ar_df['patient_id'].unique())
    final_fnsz_seiz_onset_df['patient_id'] = final_fnsz_seiz_onset_df.pstrst.apply(lambda x: x.split('$')[0])
    final_fnsz_seiz_onset_df = final_fnsz_seiz_onset_df.loc[
        final_fnsz_seiz_onset_df.patient_id.isin(lst_pat_fnsz_bckg_ar)]
    # selecting only relevant columns for next step of train test split
    final_fnsz_seiz_onset_df = final_fnsz_seiz_onset_df[['pstrst', 'start_time', 'stop_time', 'file_path']]
    # we also make the non seizure df ready now
    bckg_term_ar_df['pstr'] = bckg_term_ar_df.apply(
        lambda row: formulate_pstr(row.patient_id, row.session_id, row.session_date, row.t_id, row.tcp_ref), axis=1)
    bckg_term_ar_df['dur'] = round((bckg_term_ar_df['stop_time'] - bckg_term_ar_df['start_time']), 2)
    # we remove non_seizure records less than 16 seconds in duration
    bckg_term_ar_df = bckg_term_ar_df.loc[
        bckg_term_ar_df['dur'] >= 16]  # 16 sec is our max time interval for extracting features
    # we sort the df in ascending order of duration and then start taking 16 sec records
    bckg_term_ar_df = bckg_term_ar_df.sort_values(by=['dur'])
    bckg_term_ar_df = bckg_term_ar_df[['pstr', 'start_time', 'stop_time', 'file_path', 'dur']]
    bckg_term_ar_df['posbl_rec'] = bckg_term_ar_df[
                                       'dur'] / 16  # 16 sec is our max time interval for extracting features but this does not mean that windows of > 16 sec are not possible
    bckg_term_ar_df['posbl_rec'] = bckg_term_ar_df['posbl_rec'].apply(lambda x: math.floor(x))
    bckg_term_ar_df['lst_s_e'] = bckg_term_ar_df.apply(
        lambda row: fetch_non_seiz_start_ends(row.start_time, row.posbl_rec), axis=1)
    # we call a function to prepare 10 times more non seizure time interval frames for our non seizure set than the fnsz seizure time intervals
    fnsz_final_non_seiz_df = formulate_final_non_seiz_onset_df(bckg_term_ar_df, 'fnsz')
    fnsz_final_non_seiz_df = fnsz_final_non_seiz_df.drop_duplicates()
    # revise the saved file
    fnsz_final_non_seiz_df.to_csv(output_folder_path + 'fnsz' + '_final_term_non_seiz_times.csv',
                                  index=False)
    fnsz_final_non_seiz_df = fnsz_final_non_seiz_df[['pstrst', 'start_time', 'stop_time', 'file_path']]
    lst_unq_non_seiz_pstrst = fnsz_final_non_seiz_df.pstrst.unique()
    # we intend to have only unique records in fnsz_final_non_seiz_df so we drop duplicates in next step
    fnsz_final_unq_non_seiz_df = pd.DataFrame()
    for item in lst_unq_non_seiz_pstrst:
        sel_df = fnsz_final_non_seiz_df.loc[fnsz_final_non_seiz_df['pstrst'] == item].iloc[0:1, :]
        # concat all such dfs to form one big df of size 20728
        fnsz_final_unq_non_seiz_df = pd.concat([fnsz_final_unq_non_seiz_df, sel_df])
    fnsz_final_unq_non_seiz_df['pstrst'] = fnsz_final_unq_non_seiz_df['pstrst'].apply(lambda x: x.replace('$', '__'))
    return final_fnsz_seiz_onset_df, fnsz_final_unq_non_seiz_df


if __name__ == '__main__':
    # Check if the correct number of arguments is provided
    try:
        if len(sys.argv) != 3:
            print("Usage: python3 data_preprocessing.py input_folder_path output_folder_path")
        else:
            input_folder_path = sys.argv[1]
            output_folder_path = sys.argv[2]
    except Exception as ex:
        input_folder_path = '/media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*'  # some data source input path
        output_folder_path = '/media/data/fol/visor/data_files/'  # some output path
        pass
    # load all data from TUH EEG website into our local postgreSQL database
    insert_data_in_postgres_tables(
        input_folder_path)  # local path of the TUH EEG Seizure data folder as downloaded from the TUH website
    # load all patients data
    patient_df = load_df('patient')
    # load all channel annotations data
    channel_df = load_df('channel_annotations')
    # load all term annotations data
    term_df = load_df('term_annotations')
    # load all focal seizure channel annotations data into a dataframe
    fnsz_channel_df = channel_df.loc[channel_df['label'] == 'fnsz']  # ('spsz','cpsz','fnsz')
