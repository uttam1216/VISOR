import mne
import numpy as np
import os
import glob
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt, sosfilt
from scipy.stats import pearsonr
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import pywt
import matplotlib.pylab as plt
import operator
from collections import Counter
import pyedflib
import operator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statistics
import math
from functools import reduce
import csv
import random
from scipy.stats import kurtosis,skew

# function to load data(TUHEEG Metadata) from postgresql table into pandas data frames
def load_df(table_name):
    try:
        #conn = psycopg2.connect(host='localhost', dbname='tuheeg', user='tuheeg_user', password='tuheeg', port=5432)
        engine = create_engine('postgresql://tuheeg_user:tuheeg@localhost:5432/tuheeg')
        #loading patients data in a dataframe
        try:
            query = f"SELECT * FROM public.{table_name};".format()
        except Exception as ex:
            num_rows_before_ins = 0
            print(ex)
            pass
        #df = pd.read_sql(query, conn)
        df = pd.read_sql_query(query, con=engine)
        return df
    except Exception as ex:
        print(ex)
        return None
   

# function to take a particular labeld seizure channel dataframe and compute stats related to that type of seizure
def get_seiz_stats_df(seizType_channel_df):
    lst_pat_sess_with_seizType_chnl = list(seizType_channel_df.pstr.unique())
    seizType = seizType_channel_df.label.values[0]
    #count no of distinct seizure events present in 687 seizure sessions since more than one seizrue can exist 
    #frame pstr_wise_fnsz_chnl_stats_df 
    lst_pstr, lst_temp_top_chnl_cnt, lst_cnt_unq_chnls, lst_temp_unq_start_time_cnt, lst_lst_unq_chnls_in_val_cnt, lst_lst_unq_chnl_vals_in_val_cnt, lst_lst_unq_start_times, lst_lst_top_chnl_unq_start_times, lst_lst_seiz_duration_top_chnl, lst_lst_seiz_duration_all_chnl = [], [], [], [], [], [], [], [], [], []
    for pstr in lst_pat_sess_with_seizType_chnl:
        lst_pstr.append(pstr)
        #print('\n\n')
        temp_top_chnl_cnt, temp_cnt_unq_chnls, temp_unq_start_time_cnt, lst_unq_chnls_in_val_cnt, lst_unq_chnl_vals_in_val_cnt, lst_unq_start_times, lst_top_chnl_unq_start_times, lst_seiz_duration_top_chnl, lst_seiz_duration_all_chnl = 0, 0, 0, [], [], [], [], [], []
        pstr_seizType_channel_df = seizType_channel_df.loc[seizType_channel_df['pstr']==pstr]
        pstr_seizType_channel_df = pstr_seizType_channel_df[['channel','start_time','stop_time','pstr']]
        pstr_seizType_channel_df['dur'] = round((pstr_seizType_channel_df['stop_time'] - pstr_seizType_channel_df['start_time']),2)
        #compute session wise mean, median and minimum duration of a fnsz seizure grouped by each unq start time
        lst_unq_start_times = list(pstr_seizType_channel_df.start_time.unique())
        temp_unq_start_time_cnt = len(lst_unq_start_times)
        temp_top_chnl_cnt = pstr_seizType_channel_df['channel'].value_counts()[0]
        #print(temp_top_chnl_cnt)
        ch = pstr_seizType_channel_df['channel'].value_counts()
        lst_unq_chnls_in_val_cnt = ch.index.tolist()
        #print('topmost channel of lst_unq_chnls_in_val_cnt: ',lst_unq_chnls_in_val_cnt[0])
        red_df = pstr_seizType_channel_df.loc[pstr_seizType_channel_df['channel']==lst_unq_chnls_in_val_cnt[0]]
        lst_top_chnl_unq_start_times = list(red_df.start_time)
        # list of individual duration of all seiz channels in this session (uniqued..)
        lst_seiz_duration_all_chnl = list(pstr_seizType_channel_df['dur'])
        # duration of topmost(most frequently occuring channel in seizures for this session)
        lst_seiz_duration_top_chnl = list(red_df.dur)
        temp_cnt_unq_chnls = len(lst_unq_chnls_in_val_cnt) 
        #print(lst_unq_chnls_in_val_cnt)
        lst_unq_chnl_vals_in_val_cnt = ch.values.tolist()
        #print(lst_unq_chnl_vals_in_val_cnt)
        #print(pstr_seizType_channel_df)
        #print('###################################################################')
        lst_temp_top_chnl_cnt.append(temp_top_chnl_cnt)
        lst_cnt_unq_chnls.append(temp_cnt_unq_chnls)
        lst_temp_unq_start_time_cnt.append(temp_unq_start_time_cnt)
        lst_lst_unq_chnls_in_val_cnt.append(lst_unq_chnls_in_val_cnt)
        lst_lst_unq_chnl_vals_in_val_cnt.append(lst_unq_chnl_vals_in_val_cnt)
        lst_lst_unq_start_times.append(lst_unq_start_times)
        lst_lst_top_chnl_unq_start_times.append(lst_top_chnl_unq_start_times)
        lst_lst_seiz_duration_top_chnl.append(lst_seiz_duration_top_chnl)
        lst_lst_seiz_duration_all_chnl.append(lst_seiz_duration_all_chnl)
    #create dataframe from the lists
    pdf_data = {'pstr': lst_pstr, 'top_chnl_cnt': lst_temp_top_chnl_cnt, 'unq_chnl_cnt': lst_cnt_unq_chnls, 'unq_start_time_cnt': lst_temp_unq_start_time_cnt, 'unq_chnls_in_val_cnt': lst_lst_unq_chnls_in_val_cnt, 'unq_chnl_vals_in_val_cnt': lst_lst_unq_chnl_vals_in_val_cnt, 'unq_start_times':lst_lst_unq_start_times, 'top_chnl_unq_start_times':lst_lst_top_chnl_unq_start_times, 'seiz_duration_top_chnl':lst_lst_seiz_duration_top_chnl, 'seiz_duration_all_chnl':lst_lst_seiz_duration_all_chnl}
    seizType_stats_df = pd.DataFrame(pdf_data, columns = ['pstr','top_chnl_cnt','unq_chnl_cnt','unq_start_time_cnt','unq_chnls_in_val_cnt','unq_chnl_vals_in_val_cnt','unq_start_times','top_chnl_unq_start_times','seiz_duration_top_chnl','seiz_duration_all_chnl'])
    file_path = '../data_files/'+seizType+'_seiz_stats.csv'
    seizType_stats_df.to_csv(file_path, index=False)    
    return seizType_stats_df
    
    
def get_seiz_stats_df_with_prints(seizType_channel_df,num_records):
    ctr = 0
    lst_pat_sess_with_seizType_chnl = list(seizType_channel_df.pstr.unique())
    seizType = seizType_channel_df.label.values[0]
    #count no of distinct seizure events present in 687 seizure sessions since more than one seizrue can exist 
    #frame pstr_wise_fnsz_chnl_stats_df 
    lst_pstr, lst_temp_top_chnl_cnt, lst_cnt_unq_chnls, lst_temp_unq_start_time_cnt, lst_lst_unq_chnls_in_val_cnt, lst_lst_unq_chnl_vals_in_val_cnt, lst_lst_unq_start_times, lst_lst_top_chnl_unq_start_times, lst_lst_seiz_duration_top_chnl, lst_lst_seiz_duration_all_chnl = [], [], [], [], [], [], [], [], [], []
    for pstr in lst_pat_sess_with_seizType_chnl:
        lst_pstr.append(pstr)
        #print('\n\n')
        temp_top_chnl_cnt, temp_cnt_unq_chnls, temp_unq_start_time_cnt, lst_unq_chnls_in_val_cnt, lst_unq_chnl_vals_in_val_cnt, lst_unq_start_times, lst_top_chnl_unq_start_times, lst_seiz_duration_top_chnl, lst_seiz_duration_all_chnl = 0, 0, 0, [], [], [], [], [], []
        pstr_seizType_channel_df = seizType_channel_df.loc[seizType_channel_df['pstr']==pstr]
        pstr_seizType_channel_df = pstr_seizType_channel_df[['channel','start_time','stop_time','pstr']]
        pstr_seizType_channel_df['dur'] = round((pstr_seizType_channel_df['stop_time'] - pstr_seizType_channel_df['start_time']),2)
        #compute session wise mean, median and minimum duration of a fnsz seizure grouped by each unq start time
        lst_unq_start_times = list(pstr_seizType_channel_df.start_time.unique())
        temp_unq_start_time_cnt = len(lst_unq_start_times)
        temp_top_chnl_cnt = pstr_seizType_channel_df['channel'].value_counts()[0]
        #print(temp_top_chnl_cnt)
        ch = pstr_seizType_channel_df['channel'].value_counts()
        lst_unq_chnls_in_val_cnt = ch.index.tolist()
        #print('topmost channel of lst_unq_chnls_in_val_cnt: ',lst_unq_chnls_in_val_cnt[0])
        red_df = pstr_seizType_channel_df.loc[pstr_seizType_channel_df['channel']==lst_unq_chnls_in_val_cnt[0]]
        lst_top_chnl_unq_start_times = list(red_df.start_time)
        # list of individual duration of all seiz channels in this session (uniqued..)
        lst_seiz_duration_all_chnl = list(pstr_seizType_channel_df['dur'])
        # duration of topmost(most frequently occuring channel in seizures for this session)
        lst_seiz_duration_top_chnl = list(red_df.dur)
        temp_cnt_unq_chnls = len(lst_unq_chnls_in_val_cnt) 
        #print(lst_unq_chnls_in_val_cnt)
        lst_unq_chnl_vals_in_val_cnt = ch.values.tolist()
        #print(lst_unq_chnl_vals_in_val_cnt)
        #print(pstr_seizType_channel_df)
        #print('###################################################################')
        lst_temp_top_chnl_cnt.append(temp_top_chnl_cnt)
        lst_cnt_unq_chnls.append(temp_cnt_unq_chnls)
        lst_temp_unq_start_time_cnt.append(temp_unq_start_time_cnt)
        lst_lst_unq_chnls_in_val_cnt.append(lst_unq_chnls_in_val_cnt)
        lst_lst_unq_chnl_vals_in_val_cnt.append(lst_unq_chnl_vals_in_val_cnt)
        lst_lst_unq_start_times.append(lst_unq_start_times)
        lst_lst_top_chnl_unq_start_times.append(lst_top_chnl_unq_start_times)
        lst_lst_seiz_duration_top_chnl.append(lst_seiz_duration_top_chnl)
        lst_lst_seiz_duration_all_chnl.append(lst_seiz_duration_all_chnl)
        if temp_unq_start_time_cnt>temp_top_chnl_cnt: 
           ctr+=1
           print('\n\ntemp_top_chnl_cnt: ',temp_top_chnl_cnt)
           print('temp_unq_start_time_cnt: ',temp_unq_start_time_cnt)
           print('topmost channel of lst_unq_chnls_in_val_cnt: ',lst_unq_chnls_in_val_cnt[0])
           print('temp_cnt_unq_chnls: ',temp_cnt_unq_chnls)
           print('lst_unq_chnls_in_val_cnt: ',lst_unq_chnls_in_val_cnt)
           print('lst_unq_chnl_vals_in_val_cnt: ',lst_unq_chnl_vals_in_val_cnt)
           print('lst_lst_seiz_duration_top_chnl: ',lst_lst_seiz_duration_top_chnl)
           print('lst_lst_seiz_duration_all_chnl: ',lst_seiz_duration_all_chnl)
           print(pstr_seizType_channel_df)
           print('###################################################################')
           if ctr == num_records:
              break
              

def fetch_lst_unq_stop_times(lst_of_lst):
    lst, lst_unq_items = [], []
    for item in lst_of_lst:
        lst+=item
    lst_unq_items = reduce(lambda acc,elem: acc+[elem] if not elem in acc else acc , lst, [])
    return lst_unq_items
    
    
# function to make a list of items merged within 3 sec of the start time
def agg_st_tm(lst):
    aggregated_list = []
    aggregated_list.append(lst[0])
    lst = [x for x in lst if x not in aggregated_list]
    for item in lst:
        if not (item>=aggregated_list[-1] and item<aggregated_list[-1]+12):
           aggregated_list.append(item)
           lst = [x for x in lst if x not in aggregated_list]
    return aggregated_list


## function to frame a df with valid seizure onset times 
# contents: pstr, lst_seiz_start_times, unq_chnls_in_val_cnt having seiz_duration_all_chnl < 12 sec removed
def frame_seiz_onset_df(seizType_stats_df,seizType_channel_df,seizType):
    ctr = 0
    lst_pstr, lst_file_path, lst_lst_unq_chnls_desc, lst_lst_seiz_start_times, lst_lst_lst_chnls, lst_lst_lst_stop_times = [], [], [], [], [], []
    # for fnsz type of seizure, this iterates over 687 patient sessions
    for row in seizType_stats_df.itertuples(index=False): 
        lst_unq_chnls_desc, lst_seiz_start_times, lst_lst_stop_times, lst_lst_chnls = [], [], [], []
        lst_pstr.append(row.pstr)
        # importance wise or higher value count wise channels
        lst_unq_chnls_desc.append(row.unq_chnls_in_val_cnt)
        lst_seiz_start_times = row.unq_start_times
        # now we go inside channel_df to extract pstr_start_time wise rec 
        pstr_seizType_channel_df = seizType_channel_df.loc[seizType_channel_df['pstr']==row.pstr]
        for i in range(len(lst_seiz_start_times)):
            unq_time_pstr_seizType_channel_df = pstr_seizType_channel_df.loc[pstr_seizType_channel_df['start_time']==lst_seiz_start_times[i]]
            lst_lst_stop_times.append(list(unq_time_pstr_seizType_channel_df.stop_time))
            lst_lst_chnls.append(list(unq_time_pstr_seizType_channel_df.channel))
            temp_file_path = unq_time_pstr_seizType_channel_df.file_path.values[0]
            ctr+=1
        lst_lst_lst_stop_times.append(lst_lst_stop_times)
        lst_lst_lst_chnls.append(lst_lst_chnls)
        lst_file_path.append(temp_file_path)
        lst_seiz_start_times.sort()
        lst_lst_seiz_start_times.append(lst_seiz_start_times)
        lst_lst_unq_chnls_desc.append(lst_unq_chnls_desc)
    #create dataframe based on these stats
    print('Total no of distinct seizure onset times for this type of seizure over all sessions: ',ctr)
    pdf_data = {'pstr': lst_pstr, 'seiz_start_tm': lst_lst_seiz_start_times, 'lst_seiz_stop_tm':lst_lst_lst_stop_times, 'lst_chnls':lst_lst_lst_chnls,  'unq_chnls_in_val_cnt_desc': lst_lst_unq_chnls_desc, 'file_path':lst_file_path}
    seizType_onset_df = pd.DataFrame(pdf_data, columns = ['pstr','seiz_start_tm','lst_seiz_stop_tm','lst_chnls','unq_chnls_in_val_cnt_desc','file_path'])
    # adding some features
    seizType_onset_df['unq_sz_stop_tm'] = seizType_onset_df['lst_seiz_stop_tm'].apply(lambda x:fetch_lst_unq_stop_times(x))
    seizType_onset_df['cnt_sz_start_tm'] = seizType_onset_df['seiz_start_tm'].apply(lambda x: len(x))
    seizType_onset_df['cnt_unq_sz_stop_tm'] = seizType_onset_df['unq_sz_stop_tm'].apply(lambda x: len(x))
    seizType_onset_df['agg_unq_st_tm_in_12_sec'] = seizType_onset_df['seiz_start_tm'].apply(lambda x: agg_st_tm(x))
    seizType_onset_df['cnt_agg_unq_st_tm_in_12_sec'] = seizType_onset_df['agg_unq_st_tm_in_12_sec'].apply(lambda x: len(x))
    seizType_onset_df['seiz_prop_tm'] = seizType_onset_df.apply(lambda row: ([x for x in row.seiz_start_tm if x not in row.agg_unq_st_tm_in_12_sec]),axis=1)
    seizType_onset_df = seizType_onset_df[['pstr','seiz_start_tm','lst_seiz_stop_tm','lst_chnls','unq_chnls_in_val_cnt_desc','unq_sz_stop_tm','cnt_sz_start_tm','cnt_unq_sz_stop_tm','agg_unq_st_tm_in_12_sec','cnt_agg_unq_st_tm_in_12_sec','seiz_prop_tm','file_path']]
    seizType_onset_df.to_csv('../data_files/'+seizType+'_seiz_onset_times.csv', index=False)
    return seizType_onset_df
    
    
# function to fetch channel combination and independent electrodes for a type of seizure in order of significance in seizure participation 
def get_dict_imp_chnls_elec(seizType_stats_df):
    lst_unq_chn_comb, lst_unq_chnl_comb_val, lst_unq_elec, lst_unq_elec_vals = [], [], [], []
    #first collect all lists of the channel combinations into a list and vals in another list
    for row in seizType_stats_df.itertuples(index=False): 
        lst_unq_chn_comb+=row.unq_chnls_in_val_cnt
        lst_unq_chnl_comb_val+=row.unq_chnl_vals_in_val_cnt
    # take unique items of that list and make an empty list for corresponding items
    set_unq_chn_comb = list(set(lst_unq_chn_comb))
    set_unq_chn_comb_vals = []
    # for each item in the set list of channel comb, add up their weights to form a dict and sort in desc
    for item in set_unq_chn_comb:
        #get indexes of same item in lst_unq_chn_comb and then using those indexes get vals from lst_unq_chnl_comb_val
        indices = [i for i, x in enumerate(lst_unq_chn_comb) if x == item]
        item_vals_sum = 0
        for i in range(len(indices)):
            item_vals_sum += lst_unq_chnl_comb_val[indices[i]]
        set_unq_chn_comb_vals.append(item_vals_sum)
        
    scaler = MinMaxScaler()
    rescaled_set_unq_chn_comb_vals_arr = scaler.fit_transform(np.array(set_unq_chn_comb_vals).reshape(-1,1))
    lst_rescaled_set_unq_chn_comb_vals = [round(x[0],2) for x in rescaled_set_unq_chn_comb_vals_arr]

    dct_vals = dict(zip(set_unq_chn_comb, set_unq_chn_comb_vals))
    sorted_dct_desc_vals = dict(sorted(dct_vals.items(), key=operator.itemgetter(1), reverse=True))
    dct_wgts = dict(zip(set_unq_chn_comb, lst_rescaled_set_unq_chn_comb_vals))
    sorted_dct_desc_wgts = dict(sorted(dct_wgts.items(), key=operator.itemgetter(1), reverse=True))
    
    # process significance for single electrodes
    for key, value in sorted_dct_desc_vals.items():
        if not key.split('-')[0] in lst_unq_elec:
           lst_unq_elec.append(key.split('-')[0])
        if not key.split('-')[-1] in lst_unq_elec:
           lst_unq_elec.append(key.split('-')[-1])
    
    for item in lst_unq_elec:
        temp_sum = 0
        for key, value in sorted_dct_desc_vals.items():
            if item in key:
               temp_sum+=value
        lst_unq_elec_vals.append(temp_sum) 
    
    #electrodes with their values for this type of seizure for this reference system are
    scaler = MinMaxScaler()
    rescaled_lst_unq_elec_vals_arr = scaler.fit_transform(np.array(lst_unq_elec_vals).reshape(-1,1))
    lst_rescaled_lst_unq_elec_vals = [round(x[0],2) for x in rescaled_lst_unq_elec_vals_arr]

    dct_elec_vals = dict(zip(lst_unq_elec, lst_unq_elec_vals))
    sorted_dct_desc_elec_vals = dict(sorted(dct_elec_vals.items(), key=operator.itemgetter(1), reverse=True))
    dct_elec_wgts = dict(zip(lst_unq_elec, lst_rescaled_lst_unq_elec_vals))
    sorted_dct_desc_elec_wgts = dict(sorted(dct_elec_wgts.items(), key=operator.itemgetter(1), reverse=True))
    
    return sorted_dct_desc_vals, sorted_dct_desc_wgts, sorted_dct_desc_elec_vals, sorted_dct_desc_elec_wgts    
    

# function to write dict to csv
def write_dict_to_csv(file_path, mydict):
    with open(file_path, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
           writer.writerow([key, value])
           
           
# function to read csv into dict 
def read_csv_into_dict(file_path):
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
    return mydict
    

# mean, stdev, cutoff threshold(mean-1std) of freq of occurence and lst of the chnls/electrodes above threshold
def get_dct_stats(mydict,thres_decider):
    lst,lst_keys_above_thres,dct_above_thres = [],[],{}
    for key, value in mydict.items():
        lst.append(value)
    mn=round(statistics.mean(lst),2)
    std=round(statistics.stdev(lst),2)
    if thres_decider == 'mean':
       thres=mn  
    else:
       thres=round(mn-std,2)
    for key, value in mydict.items():
        if value>thres:
           lst_keys_above_thres.append(key)
           dct_above_thres[key]=value
    return mn,std,thres,lst_keys_above_thres,dct_above_thres
    
#function to formulate final seizure onset df for a type of seizure being passed in seizType...for AR or LE reference
def formulate_final_seiz_onset_df(seizType_onset_df,seizType):
    import math
    # lst_pstrst: pstr + math.floor(start_time), lst_start_time, 
    # lst_stop_time (max of stop time corresponding to common start time), file_path
    lst_pstrst, lst_start_time, lst_stop_time, lst_file_path = [], [], [], []
    for row in seizType_onset_df.itertuples(index=False): 
        for item in row.agg_unq_st_tm_in_12_sec:
            lst_pstrst.append(row.pstr+'$'+str(math.floor(item)))
            # find index of same time in list of row.seiz_start_tm
            lst_start_time.append(item)
            idx = row.seiz_start_tm.index(item)
            # fetch the particular item at this index of lst_seiz_stop_tm
            temp_lst_stop_time = row.lst_seiz_stop_tm[idx]
            #suppose 3 chnls os eiz start at 82.03 sec and end separately at 97.3, 98.2, 99.6 then stop_time=99.6
            lst_stop_time.append(max(temp_lst_stop_time))
            lst_file_path.append(row.file_path)
    #create df for seiz onset times of this type of seizure
    pdf_data = {'pstrst': lst_pstrst, 'start_time': lst_start_time, 'stop_time': lst_stop_time, 'file_path': lst_file_path}
    seizType_final_onset_df = pd.DataFrame(pdf_data, columns = ['pstrst','start_time','stop_time','file_path'])
    seizType_final_onset_df['seiz_dur'] = seizType_final_onset_df.stop_time - seizType_final_onset_df.start_time
    seizType_final_onset_df.to_csv('../data_files/'+ seizType +'_final_seiz_onset_times.csv', index=False)
    return seizType_final_onset_df


# function to read edf file and extract/crop image of seizure-onset/non-seizure 
def save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, start_time, end_time):
    try:
       raw = mne.io.read_raw_edf(edf_file_path)
       #raw.resample(128)
       raw.load_data()
       raw.filter(l_freq=0.1, h_freq=64) 
       raw.notch_filter(freqs=50)
       raw.pick_channels(lst_channels_to_save)
       
       # Set the time period for which you want to save the images
       duration = end_time - start_time
       raw.crop(tmin=start_time, tmax=end_time)
       
       # Plot all EEG channels together
       plt = raw.plot(duration=duration)
       #plt.figure(figsize=(15, 8))
       plt.savefig(output_file_path)
       #plt.close()
    except Exception as ex:
       print('Exception came for file and start & end times: ',edf_file_path, start_time, end_time)
       print(ex)
       pass
        

# for a dataframe of format pstrst	start_time	stop_time	file_path, we save images of specific seizur eonset duration
def save_eeg_extracted_images(df,output_folder,lst_channels_to_save,seiz_non_seiz):
    for row in df.itertuples(index=False):
        edf_file_path = row.file_path.replace('csv_bi','edf').replace('csv','edf')
        output_file_path = output_folder+row.pstrst+'.png'
        output_file_path = output_file_path.replace('$','__')
        if seiz_non_seiz == 'seiz':
           save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, (row.start_time-6), (row.start_time+6)) 
        else:
           save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, row.start_time, row.stop_time)


# function to create folders named upon patients and move all files in respective folder
# before this, also keep all images at one place after sanity check for full black outs..which should be deleted
def arrange_images(orig_folder,parent_folder):
    #e.g. orig_folder = all_fnsz_onset_train_images, parent_folder = fnsz_onset_train
    # copy it to patient wise folders to check patterns
    lst_pat = []
    for fin in glob.glob('../data_files/'+orig_folder+'/*'):
        #get a list of all patients to create folders with those names
        temp_pat = fin.split('/')[-1].split('__')[0]
        lst_pat.append(temp_pat)
    lst_unq_pat = reduce(lambda acc,elem: acc+[elem] if not elem in acc else acc , lst_pat, [])
    for item in lst_unq_pat:
        os.makedirs('../data_files/'+parent_folder+'/'+item, exist_ok=True)
    for fin in glob.glob('../data_files/'+orig_folder+'/*.png'):
        file_name = fin.split('/')[-1]
        patient_id_folder = file_name.split('__')[0]
        new_path='../data_files/'+parent_folder+'/'+patient_id_folder+'/'+file_name
        shutil.copy(fin, new_path)
        
        
#function to align number of files in dataframes to that after removal of images from df respective folders
def fetch_lst_existing_files(folder_name):
    #folder_name = 'test/sz'
    lst_files = []
    for fin in glob.glob('../data_files/'+folder_name+'/*'):
        temp_file_name = fin.split('/')[-1]
        if folder_name.split('/')[-1] == 'sz':
           temp_file_name = temp_file_name.replace('__','$')
        temp_file_name = temp_file_name.replace('.png','')
        lst_files.append(temp_file_name)
    return lst_files


#evaluate result, using precision, recall and F1 score as metrics by passing y_true and y_pred
def evaluate_seizure_result(actual,predicted):
    #create a df of actual and pred for ease of work
    pdf_data = {'actual':list(actual),'predicted':list(predicted)}
    actl_pred_df = pd.DataFrame(pdf_data,columns=['actual','predicted'])
    #actuals
    tot_act_seiz = sum(actl_pred_df.actual)
    print('Total no. of actual seizures: ',tot_act_seiz)
    tot_act_nonSeiz = (len(actl_pred_df) - tot_act_seiz)
    print('Total no. of actual non-seizures: ',tot_act_nonSeiz)

    #predicted
    tot_recog_seiz = sum(actl_pred_df['predicted'])
    print('Total no. of predicted seizures: ',tot_recog_seiz)
    tot_recog_nonSeiz = len(actl_pred_df) - tot_recog_seiz
    print('Total no. of predicted non-seizures: ',tot_recog_nonSeiz)

    #total no. of correctly predicted seizures and correctly predicted nonSeizures
    corr_seiz = actl_pred_df.query('actual==1 & predicted==1')['predicted'].sum()
    corr_nonSeiz = len(actl_pred_df.query('actual==0 & predicted==0')['predicted'])
    print('total count of test set: ',len(actual))
    print('total no. of correctly predicted seizures onset windows: ',corr_seiz)
    print('total no. of correctly predicted nonSeizure windows:',corr_nonSeiz)

    precision_seiz = round((corr_seiz/tot_recog_seiz),2)
    precision_nonSeiz = round((corr_nonSeiz/tot_recog_nonSeiz),2)
    recall_seiz = round((corr_seiz/tot_act_seiz),2)
    recall_nonSeiz = round((corr_nonSeiz/tot_act_nonSeiz),2)

    seizF1 = round(((2*precision_seiz*recall_seiz)/(precision_seiz+recall_seiz)),2)
    nonSeizF1 = round(((2*precision_nonSeiz*recall_nonSeiz)/(precision_nonSeiz+recall_nonSeiz)),2)

    return precision_seiz,precision_nonSeiz,recall_seiz,recall_nonSeiz,seizF1,nonSeizF1
    
    
# func. to read edf & apply notch & butterworth filter & return digitalized signal inclusive of all chnls in list
def fetch_filtered_eeg_lst_chnls(edf_file_path, output_file_path, lst_channels_to_save, start_time, end_time):
    try:
       raw = mne.io.read_raw_edf(edf_file_path,preload=True)
       #raw.set_eeg_reference()  #since our data is already having AR, we do not need to re-reference
       raw.load_data()    #converts to numpy array
       #print('After load_data: ',raw)
       
       raw.notch_filter(freqs=50)
       #print('After applying notch_filter: ',raw)
       raw.pick_channels(lst_channels_to_save)
       #print('After picking sel. chnls: ',raw)
    
       # Set the time period for which you want to save the images
       duration = end_time - start_time
       raw.crop(tmin=start_time, tmax=end_time)
       #print('After cropping: ',raw)
    
       ##raw.filter(l_freq=0.1, h_freq=64) 
       ##print('After 0.1-64: ',raw)
    
       #data_ch, times = raw.get_data(picks=ref_electrode, return_times=True, start=start_time, stop=end_time)
       #above one gives of single electrode, below one gives for a list of electrodes
       data_ch, times = raw.get_data(picks=lst_channels_to_save, return_times=True, start=start_time, stop=end_time)
       #print('data_ch.shape: ',data_ch.shape)
       #print('data_ch', data_ch) 17 channels of duration 0 to start_time+6 sec
    
       lowcut, highcut, nyquist_freq, b_order = 0.1, 64, (raw.info['sfreq'] / 2.0) , 2
       #print('nyquist_freq: ',nyquist_freq)
       # b_order is order of butterworth filter
       sos = butter(b_order, [lowcut/nyquist_freq, highcut/nyquist_freq], btype='band', output='sos')
       #print('sos: ',sos)
       ## Apply the filter to the signal
       filtered_signal = sosfilt(sos, data_ch)
       #print('filtered_signal shape: ',filtered_signal.shape) #17 channels of duration 0 to start_time+6 sec
       print(filtered_signal)
       #break
       #np_arr = np.array(filtered_signal[0])
       return filtered_signal
    except Exception as ex:
       print('Exception came for file and start & end times: ',edf_file_path, start_time, end_time)
       print(ex)
       pass
       
       
# func. to read edf & apply notch & butterworth filter & return digitalized signal inclusive of one electrode
def fetch_filtered_eeg_for_elec(edf_file_path, output_file_path, ref_electrode, start_time, end_time):
    try:
       raw = mne.io.read_raw_edf(edf_file_path,preload=True)
       raw.load_data()       
       raw.notch_filter(freqs=50)    
       # Set the time period for which you want to save the images
       duration = end_time - start_time
       raw.crop(tmin=start_time, tmax=end_time)    
       data_ch, times = raw.get_data(picks=ref_electrode, return_times=True, start=start_time, stop=end_time)    
       lowcut, highcut, nyquist_freq, b_order = 0.1, 64, (raw.info['sfreq'] / 2.0), 2
       # b_order is the order of butterworth filter
       sos = butter(b_order, [lowcut/nyquist_freq, highcut/nyquist_freq], btype='band', output='sos')
       # Apply the filter to the signal
       filtered_signal = sosfilt(sos, data_ch)
       #print('filtered_signal shape: ',filtered_signal.shape) #1 channel of duration 0 to start_time+6 sec
       print(filtered_signal)
       return filtered_signal
    except Exception as ex:
       print('Exception came for file and start & end times: ',edf_file_path, start_time, end_time)
       print(ex)
       pass
       
       
#function to apply DB10 to the filtered eeg signal
def apply_wavelet_transform(signal, db, level):
    #Daubechies 10 wavelet transform with given level
    coeffs = pywt.wavedec(signal, db, level=level)
    #cA4, cD4, cD3, cD2, cD1 = coeffs 
    return coeffs
    
    
#function to save and plot DB10A4 level 4 images for EEG from a given dataframe 
def save_db10A4_eeg_images(df,output_file_path,lst_channels_to_save):
    lst_excp_pstrst, ctr = [], 0
    for row in df.itertuples(index=False):
        try: 
            file_name = row.pstrst.replace('$','__') + '.png'
            if not os.path.isfile(output_file_path+file_name):
               # i.e. only if file does not already exist, do following
               fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(row.file_path, 'none', lst_channels_to_save, 0, math.floor(row.start_time+8))   
               #taking total 16 sec, 8 sec before and 8 sec after the start time
               fil_16_sec_inc_onset = fil_sig_for_elec[:,fil_sig_for_elec.shape[-1]-16:fil_sig_for_elec.shape[-1]]
               # Apply wavelet transform
               wavelet_coeffs_db10 = apply_wavelet_transform(fil_16_sec_inc_onset, 'db10', 4)
               cA4 = wavelet_coeffs_db10[0]
               x = np.linspace(0, 15, 18)
               y = np.asarray(cA4)
               plt.plot(x, y.T)
               plt.xlabel("Time(s)")
               plt.ylabel("Elec (µV)")  
               plt.savefig(output_file_path+file_name)
               plt.show() 
               plt.close()
               ctr+=1
        except Exception as ex:
            print(ex)
            lst_excp_pstrst.append(row.pstrst)
            pass
    return ctr, lst_excp_pstrst



#function to apply DB10 to the filtered eeg signal
def apply_wavelet_transform(signal, db, level):
    #Daubechies 10 wavelet transform with given level
    coeffs = pywt.wavedec(signal, db, level=level)
    #cA4, cD4, cD3, cD2, cD1 = coeffs 
    return coeffs
    
    
#function to save and plot DB10A4 level 4 images for EEG from a given dataframe 
def save_db10A4_eeg_images(df,output_file_path,lst_channels_to_save):
    lst_excp_pstrst, ctr = [], 0
    for row in df.itertuples(index=False):
        try: 
            file_name = row.pstrst.replace('$','__') + '.png'
            if not os.path.isfile(output_file_path+file_name):
               # i.e. only if file does not already exist, do following
               fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(row.file_path, 'none', lst_channels_to_save, 0, math.floor(row.start_time+8))   
               #taking total 16 sec, 8 sec before and 8 sec after the start time
               fil_16_sec_inc_onset = fil_sig_for_elec[:,fil_sig_for_elec.shape[-1]-16:fil_sig_for_elec.shape[-1]]
               # Apply wavelet transform
               wavelet_coeffs_db10 = apply_wavelet_transform(fil_16_sec_inc_onset, 'db10', 4)
               cA4 = wavelet_coeffs_db10[0]
               x = np.linspace(0, 15, 18)
               y = np.asarray(cA4)
               plt.plot(x, y.T)
               plt.xlabel("Time(s)")
               plt.ylabel("Elec (µV)")  
               plt.savefig(output_file_path+file_name)
               plt.show() 
               plt.close()
               ctr+=1
        except Exception as ex:
            print(ex)
            lst_excp_pstrst.append(row.pstrst)
            pass
    return ctr, lst_excp_pstrst


#function to get Shannon Entropy, returns list of sh's
def get_shan(d1,d2,d3,d4,a4):
    sh1,sh2,sh3,sh4,sh5=[],[],[],[],[]
    [sh1.append(entropy.shannon_entropy(d1[i])) for i in range(0,d1.shape[0])]
    [sh2.append(entropy.shannon_entropy(d2[i])) for i in range(0,d2.shape[0])]
    [sh3.append(entropy.shannon_entropy(d3[i])) for i in range(0,d3.shape[0])]
    [sh4.append(entropy.shannon_entropy(d4[i])) for i in range(0,d4.shape[0])]
    [sh5.append(entropy.shannon_entropy(a4[i])) for i in range(0,a4.shape[0])]
    return sh1,sh2,sh3,sh4,sh5
    
    
#function to get Energy of EEG Signal, returns list of sh's
def get_energy(d1,d2,d3,d4,a4):
    en1,en2,en3,en4,en5=[],[],[],[],[]
    for i in range(0,d1.shape[0]):
        X=d1[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en1.append(summ)
    for i in range(0,d2.shape[0]):
        X=d2[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en2.append(summ)
    for i in range(0,d3.shape[0]):
        X=d3[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en3.append(summ)
    for i in range(0,d4.shape[0]):
        X=d4[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en4.append(summ)
    for i in range(0,a4.shape[0]):
        X=a4[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en5.append(summ)

    return en1,en2,en3,en4,en5


#function to get Renyi Entropy, returns list of ry's
def get_renyi_entropy(d1,d2,d3,d4,a4):
    rend1,rend2,rend3,rend4,rend5=[],[],[],[],[]
    alpha=2    
    for i in range(0,d1.shape[0]):
        X=d1[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend1.append(Renyi_En)

    for i in range(0,d2.shape[0]):
        X=d2[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend2.append(Renyi_En)

    for i in range(0,d3.shape[0]):
        X=d3[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend3.append(Renyi_En)

    for i in range(0,d4.shape[0]):
        X=d4[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend4.append(Renyi_En)

    for i in range(0,a4.shape[0]):
        X=a4[i]
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))
        rend5.append(Renyi_En)
    return rend1,rend2,rend3,rend4,rend5
    
    
#function to get Permutation Entropy, returns list of pd's
def get_permut_entropy(d1,d2,d3,d4,a4):
    pd1,pd2,pd3,pd4,pd5=[],[],[],[],[]
    [pd1.append(round(entropy.permutation_entropy(d1[i],3,normalize=True),2)) for i in range(0,d1.shape[0])]
    [pd2.append(round(entropy.permutation_entropy(d2[i],3,normalize=True),2)) for i in range(0,d2.shape[0])]
    [pd3.append(round(entropy.permutation_entropy(d3[i],3,normalize=True),2)) for i in range(0,d3.shape[0])]
    [pd4.append(round(entropy.permutation_entropy(d4[i],3,normalize=True),2)) for i in range(0,d4.shape[0])]
    [pd5.append(round(entropy.permutation_entropy(a4[i],3,normalize=True),2)) for i in range(0,a4.shape[0])]
    return pd1,pd2,pd3,pd4,pd5
    
    
#function to get Shannon Entropy, returns list of sh's
def get_shan_entropy(d1,d2,d3,d4,a4):
    sh1,sh2,sh3,sh4,sh5=[],[],[],[],[]
    [sh1.append(entropy.shannon_entropy(d1[i])) for i in range(0,d1.shape[0])]
    [sh2.append(entropy.shannon_entropy(d2[i])) for i in range(0,d2.shape[0])]
    [sh3.append(entropy.shannon_entropy(d3[i])) for i in range(0,d3.shape[0])]
    [sh4.append(entropy.shannon_entropy(d4[i])) for i in range(0,d4.shape[0])]
    [sh5.append(entropy.shannon_entropy(a4[i])) for i in range(0,a4.shape[0])]
    return sh1,sh2,sh3,sh4,sh5
    
    
#function to get Energy of EEG Signal, returns list of sh's
def get_energy(d1,d2,d3,d4,a4):
    en1,en2,en3,en4,en5=[],[],[],[],[]
    for i in range(0,d1.shape[0]):
        X=d1[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en1.append(summ)
    for i in range(0,d2.shape[0]):
        X=d2[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en2.append(summ)
    for i in range(0,d3.shape[0]):
        X=d3[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en3.append(summ)
    for i in range(0,d4.shape[0]):
        X=d4[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en4.append(summ)
    for i in range(0,a4.shape[0]):
        X=a4[i]
        summ=0
        for i in X:
            summ+=(i**2)
        en5.append(summ)

    return en1,en2,en3,en4,en5
    
    
#function to get statistical feature - mean of the DB10/DB4 with level4 of the EEG Signal
def get_meann(d1,d2,d3,d4,a4):
    mean1,mean2,mean3,mean4,mean5=[],[],[],[],[]
    [mean1.append(np.mean(d1[i])) for i in range(0,d1.shape[0])]
    [mean2.append(np.mean(d2[i])) for i in range(0,d2.shape[0])]
    [mean3.append(np.mean(d3[i])) for i in range(0,d3.shape[0])]
    [mean4.append(np.mean(d4[i])) for i in range(0,d4.shape[0])]
    [mean5.append(np.mean(a4[i])) for i in range(0,a4.shape[0])]
    return mean1,mean2,mean3,mean4,mean5
    
    
#function to get statistical feature - standard deviation of the DB10/DB4 with level4 of the EEG Signal
def get_stdd(d1,d2,d3,d4,a4):
    std1,std2,std3,std4,std5=[],[],[],[],[]
    [std1.append(np.std(d1[i])) for i in range(0,d1.shape[0])]
    [std2.append(np.std(d2[i])) for i in range(0,d2.shape[0])]
    [std3.append(np.std(d3[i])) for i in range(0,d3.shape[0])]
    [std4.append(np.std(d4[i])) for i in range(0,d4.shape[0])]
    [std5.append(np.std(a4[i])) for i in range(0,a4.shape[0])]
    return std1,std2,std3,std4,std5
    
    
#function to get statistical feature - skewness of the DB10/DB4 with level4 of the EEG Signal
def get_skeww(d1,d2,d3,d4,a4):
    skew1,skew2,skew3,skew4,skew5=[],[],[],[],[]
    [skew1.append(round(skew(d1[i]),2)) for i in range(0,d1.shape[0])]
    [skew2.append(round(skew(d2[i]),2)) for i in range(0,d2.shape[0])]
    [skew3.append(round(skew(d3[i]),2)) for i in range(0,d3.shape[0])]
    [skew4.append(round(skew(d4[i]),2)) for i in range(0,d4.shape[0])]
    [skew5.append(round(skew(a4[i]),2)) for i in range(0,a4.shape[0])]
    return skew1,skew2,skew3,skew4,skew5
    
    
#function to get statistical feature - kurtosiss of the DB10/DB4 with level4 of the EEG Signal
def get_kurtosiss(d1,d2,d3,d4,a4):
    kurtosis1,kurtosis2,kurtosis3,kurtosis4,kurtosis5=[],[],[],[],[]
    [kurtosis1.append(round(kurtosis(d1[i]),2)) for i in range(0,d1.shape[0])]
    [kurtosis2.append(round(kurtosis(d2[i]),2)) for i in range(0,d2.shape[0])]
    [kurtosis3.append(round(kurtosis(d3[i]),2)) for i in range(0,d3.shape[0])]
    [kurtosis4.append(round(kurtosis(d4[i]),2)) for i in range(0,d4.shape[0])]
    [kurtosis5.append(round(kurtosis(a4[i]),2)) for i in range(0,a4.shape[0])]
    return kurtosis1,kurtosis2,kurtosis3,kurtosis4,kurtosis5
    
    




#function to formulate unique patient sessions
def formulate_pstr(patient_id,session_id,session_date,t_id,tcp_ref):
    pstr = patient_id + '$' + session_id + '_' + str(session_date).split('T')[0].replace('-','_').replace(' 00:00:00','') + '$' + t_id + '$' + tcp_ref
    return pstr
    

# function to save specific channels of the EEG in a single .png file
def save_eeg_channel_images_OLD(edf_file_path, output_file_path, lst_channels_to_save, start_time, end_time):
    # Load the .edf file
    raw_data = mne.io.read_raw_edf(edf_file_path, preload=True)

    # Select the EEG channels you want to save
    raw_data.pick_channels(lst_channels_to_save)

    # Set the time period for which you want to save the images
    duration = end_time - start_time
    raw_data.crop(tmin=start_time, tmax=end_time)
    
     # Plot all EEG channels together
    plt.figure(figsize=(15, 8))
    raw_data.plot(duration=duration, scalings='auto', show=False)

    # Save the plot as an image at output_file_path in .png format
    plt.savefig(output_file_path)
    plt.close()
    '''
    # Plot and save the images for each EEG channel
    for channel in lst_channels_to_save:
        plt.figure(figsize=(10, 5))
        raw_data.plot(duration=duration, scalings='auto', show=False)

        # Save the plot as an image
        output_file_path = f"{output_folder}/{channel}_eeg_{start_time}_{end_time}.png"
        plt.savefig(output_file_path)
        plt.close() '''


# create mapping, for each electrode, list down all its neighbor in a df
def fetch_neighbor_elec_mapping(mapping_file_path):
    cleaned_lst_electrodes, lst_elct_cnt_in_oneHop, lst_neighbor_electrodes = [], [], []
    try:
        ch_df = pd.read_csv(mapping_file_path)
        lst_electrodes = list(ch_df['electrodes'])
        cleaned_lst_electrodes = [x for x in lst_electrodes if str(x) != 'nan']
        lst_elct_cnt_in_oneHop, lst_neighbor_electrodes = [], []
        lst_one_hop_bipMon = list(ch_df['one_hop_bipMon'])
        for item in cleaned_lst_electrodes:
            item_cnt, lst_neighbor = 0, []
            for row in ch_df.itertuples(index=False):
                if item in str(row.one_hop_bipMon):
                   item_cnt+=1
                   neighbor_elec = str(row.one_hop_bipMon).replace(item,'').replace('-','')
                   lst_neighbor.append(neighbor_elec)
            lst_elct_cnt_in_oneHop.append(item_cnt)
            lst_neighbor_electrodes.append(lst_neighbor)
        # print(len(lst_elct_cnt_in_oneHop))
    except Exception as ex:
        pass
    pdf_data = {'electrodes': cleaned_lst_electrodes, 'count_in_1_hop': lst_elct_cnt_in_oneHop, 'neighbor':lst_neighbor_electrodes}
    s_df = pd.DataFrame(pdf_data, columns = ['electrodes','count_in_1_hop', 'neighbor'])
    return s_df
    
    
def fetch_neighbor_elec(electrode,electrodes_neighbor_df=None):
    lst_neighbor_electrodes = []
    red_electrodes_neighbor_df = electrodes_neighbor_df.loc[electrodes_neighbor_df['electrodes']==electrode]
    if not red_electrodes_neighbor_df.empty:
       lst_neighbor_electrodes = red_electrodes_neighbor_df.neighbor.values[0]
    return lst_neighbor_electrodes
    
    
# function to fetch value of remarks (whether it is from train , test or validation folder)
def get_dataset_type(file_path):
    dataset_type = ' '
    if 'train' in file_path:
        dataset_type = 'train'
    if 'dev' in file_path:
        dataset_type = 'test'
    if 'eval' in file_path:
        dataset_type = 'val'
    return dataset_type
    
    
# function to get values of nn1, nn2, nn3, nn4 for an electrode
def get_nn_values(electrode):
    lst_neighbor_elec = fetch_neighbor_elec(electrode)
    nn1,nn2,nn3,nn4 = ' ', ' ', ' ', ' '
    try:
        nn1 = lst_neighbor_elec[0]
        nn2 = lst_neighbor_elec[1]
        nn3 = lst_neighbor_elec[2]
        nn4 = lst_neighbor_elec[3]
    except:
        pass
    finally: 
        return nn1,nn2,nn3,nn4
        
        
# apply freq_band== [0.5, 40] in Hz to raw EEG Signal
def get_filtered_eeg(raw,freq_band=[0.5, 40]):   
    try:
       # Filter the data in the frequency band of interest
       raw = raw.filter(freq_band[0], freq_band[1], fir_design='firwin');
    except Exception as ex:
       print('Exception happened in applying freq_band 0.5 to 40 in get_filtered_eeg function') 
       pass
    finally:
        return raw
        

#function to get discretized raw signal data points in form of array
def fetch_array_discrete_electrode_raw_signal_old(raw, ref_electrode, start_time, stop_time):
    np_arr = np.array([0,0,0,0])
    try:
        # get raw signal only of a particular period
        data_ch_4sec, times = raw.get_data(picks=ref_electrode, return_times=True, start=start_time, stop=stop_time)
        # set notch filter to remove line noise using butterworth filter for the channel
        notch_freq = 50
        nyquist_freq = raw.info['sfreq'] / 2.0 
        b, a = butter(4, [(notch_freq - 2.0) / nyquist_freq, (notch_freq + 2.0) / nyquist_freq], btype='bandstop')
        # converting the continuous signal into a discrete sequence
        data_ch_4sec = filtfilt(b, a, data_ch_4sec, padtype=None, padlen=None)
        #print(data_ch_4sec[0])
        #break
        np_arr = np.array(data_ch_4sec[0])
    except Exception as ex:
        print('Exception in function fetch_array_discrete_electrode_raw_signal during discretizing the raw signal: ')
        print(ex)
    finally:
        return np_arr

        
#function to get discretized raw signal data points in form of array
def fetch_array_discrete_electrode_raw_signal(raw, ref_electrode, b_order=2, start_time=None, stop_time=None):
    np_arr = np.array([0,0,0,0])
    try:
        # get raw signal only of a particular period
        if start_time is None:
           data_ch, times = raw.get_data(picks=ref_electrode, return_times=True)
        else:
           data_ch, times = raw.get_data(picks=ref_electrode, return_times=True, start=start_time, stop=stop_time)
        # set loc cut and high cut filter
        lowcut = 0.1
        highcut = 64
        nyquist_freq = raw.info['sfreq'] / 2.0 
        ##b, a = butter(4, [lowcut / nyquist_freq, highcut / nyquist_freq], btype='bandpass')
        # converting the continuous signal into a discrete sequence
        ##data_ch_4sec = filtfilt(b, a, data_ch_4sec, padtype=None, padlen=None)
        sos = butter(b_order, [lowcut/nyquist_freq, highcut/nyquist_freq], btype='band', output='sos')
        # Apply the filter to the signal
        filtered_signal = sosfilt(sos, data_ch)
        #print(filtered_signal[0])
        #break
        np_arr = np.array(filtered_signal[0])
    except Exception as ex:
        print('Exception in function fetch_array_discrete_electrode_raw_signal during discretizing the raw signal: ')
        print(ex)
    finally:
        return np_arr
        
    
def get_eeg_stop_time(file_path):
    try:
       duration,sample_frequency = 0,256
       # Open the .edf file
       edf_file = pyedflib.EdfReader(file_path)
   
       # Get the number of signals in the file
       num_signals = edf_file.signals_in_file
   
       # Get the duration of the EEG signal (in seconds)
       duration = edf_file.file_duration
       # since the above gives ceil value, we suvbtract 1 to always be sure for presence of last second.
       duration = duration - 1
   
       ## Get the sample frequency of the first signal
       sample_frequency = edf_file.samplefrequency(0)
    
       #print('Total no. of data points: ', (duration*sample_frequency))
       # Close the .edf file
       edf_file.close()
    except Exception as ex:
       print(ex)
       edf_file.close()
    finally:
       return duration,sample_frequency
       
       
def sig_plot(data,max_t=1):
    time=np.linspace(0,max_t,max_t)
    plt.plot(time,data)
    plt.show()
    
    
# formulate electrode node in correct format based on reference system used 
def fetch_ref_electrode(tcp_ref,electrode):
    if 'tcp_ar' in str(tcp_ref):
        ref_electrode = 'EEG '+electrode+'-REF'
    else:
        ref_electrode = 'EEG '+electrode+'-LE'
    return ref_electrode
    
    
# formulate start times for all 3 phases - before seizure onset, during and after
def get_bda_st_et(start_time):   
    bos_st = math.floor(start_time-6)
    bos_et = math.floor(start_time-2)
    dos_st = math.floor(start_time-2)
    ### to address what if no signal exist for framed extended end time 
    ### since seizure goes on for atleast 10 sec(assumption as per DB property), 2+4 sec after seizure should not be an issue
    dos_et = math.floor(start_time+2)
    aos_st = math.floor(start_time+2)
    aos_et = math.floor(start_time+6)
    return bos_st, bos_et, dos_st, dos_et, aos_st, aos_et
    
    
# function that calls a function to discretize raw signals for multiple electrodes 
def get_signal_vectors(raw, ref_electrode, start_time): 
    global lst_exception 
    bos_vec, dos_vec, aos_vec = np.array([0,0,0,0]),np.array([0,0,0,0]),np.array([0,0,0,0])
    bos_st, bos_et, dos_st, dos_et, aos_st, aos_et = get_bda_st_et(start_time)
    if ref_electrode in raw.ch_names:
       try:
           #print('GETTING SIGNAL VECTORS FOR ref_electrode: ',ref_electrode)
           #print('GETTING SIGNAL VECTORS FOR start_time: ',start_time)
           #bos_vec = fetch_array_discrete_electrode_raw_signal(raw, ref_electrode, bos_st, bos_et)
           #dos_vec = fetch_array_discrete_electrode_raw_signal(raw, ref_electrode, dos_st, dos_et)
           #aos_vec = fetch_array_discrete_electrode_raw_signal(raw, ref_electrode, aos_st, aos_et)
           # changes done on 12th July
           aos_dos_bos_vec = fetch_array_discrete_electrode_raw_signal(raw, ref_electrode)
           #print(len(aos_dos_bos_vec))
           #print('timings for electrode: ',ref_electrode,' are: ',bos_st,bos_et,dos_st,dos_et,aos_st,aos_et)
           bos_vec = aos_dos_bos_vec[bos_st:bos_et]
           dos_vec = aos_dos_bos_vec[dos_st:dos_et]
           aos_vec = aos_dos_bos_vec[aos_st:aos_et]
           print(bos_vec, dos_vec, aos_vec)
           print('                         ')
       except Exception as ex:
           ## to address what if no signal exist for framed extended end time
           lst_exception.append(ex)
           print('Exception occured in function get_signal_vectors: ')
           print(ex)
       finally:
           return bos_vec, dos_vec, aos_vec
    else:
       return bos_vec, dos_vec, aos_vec
       
       
#core func to find correlations between 3 phases before seizure onset, during and after b/w current elec & nebr
def fetch_corr_bw_vecs(raw,lst_ch_names,tcp_ref,bos_vec,dos_vec,aos_vec,nn,start_time):
    c_bos_nn, c_dos_nn, c_aos_nn = 0, 0, 0
    try:
        ref_electrode_for_nn = fetch_ref_electrode(tcp_ref,nn)
        bos_vec_nn,dos_vec_nn,aos_vec_nn = get_signal_vectors(raw, ref_electrode_for_nn, start_time)
        if nn!= ' ' and ref_electrode_for_nn in lst_ch_names:
               c_bos_nn, p_value = pearsonr(bos_vec.ravel(), bos_vec_nn.ravel())
               c_dos_nn, p_value = pearsonr(dos_vec.ravel(), dos_vec_nn.ravel())
               c_aos_nn, p_value = pearsonr(aos_vec.ravel(), aos_vec_nn.ravel())
        else:
               c_bos_nn, c_dos_nn, c_aos_nn = 0, 0, 0
    except Exception as ex:
        print('Exception occurred in function fetch_corr_bw_vecs: ')
        print(ex)
    finally:
        return c_bos_nn,c_dos_nn,c_aos_nn
        
        
#func. to find correlations between 3 phases before seizure onset, during and after b/w current elec & nebr
def fetch_corr_with_nn(raw,lst_ch_names,tcp_ref,bos_vec,dos_vec,aos_vec,nn1,nn2,nn3,nn4,start_time):
    c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4=0,0,0,0,0,0,0,0,0,0,0,0
    try:
        c_bos_nn1, c_dos_nn1, c_aos_nn1 = fetch_corr_bw_vecs(raw,lst_ch_names,row.tcp_ref,bos_vec,dos_vec,aos_vec,nn1,start_time)
        c_bos_nn2, c_dos_nn2, c_aos_nn2 = fetch_corr_bw_vecs(raw,lst_ch_names,row.tcp_ref,bos_vec,dos_vec,aos_vec,nn2,start_time)
        c_bos_nn3, c_dos_nn3, c_aos_nn3 = fetch_corr_bw_vecs(raw,lst_ch_names,row.tcp_ref,bos_vec,dos_vec,aos_vec,nn3,start_time)
        c_bos_nn4, c_dos_nn4, c_aos_nn4 = fetch_corr_bw_vecs(raw,lst_ch_names,row.tcp_ref,bos_vec,dos_vec,aos_vec,nn4,start_time)
    except Exception as ex:
        print('Exception occurred in function fetch_corr_with_nn: ')
        print(ex)
    finally: 
        return c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4
        
        
        
def formulate_pstr(patient_id,session_id,session_date,t_id,tcp_ref):
    pstr = patient_id + '$' + session_id + '_' + str(session_date).split('T')[0].replace('-','_').replace(' 00:00:00','') + '$' + t_id + '$' + tcp_ref
    return pstr
    
    
# new func created with logic to take non seizure period much before seizure period
def get_non_seiz_start_stop_time(red_df,bs_or_as,sz_start_or_stop_time,duration):
    global lst_pstrl_non_seiz_st_time
    # if bs_or_as = bs then sz_start_or_stop_time has sz_start_time else sz_stop_time
    ## can sort the red_df start time wise, but doesn't effect
    #collect list of paired start stop times of seizures 
    lst_seiz_start_stop_time = list(zip(red_df.start_time, red_df.stop_time))
    final_non_seiz_time = 0 # to be returned if nothing fits
    if bs_or_as == 'bs':
       non_seiz_start_time = sz_start_or_stop_time - (6+5+6)   # before seizure 'bs'
    else:
       non_seiz_start_time = sz_start_or_stop_time + (6+5+6) # after 1 seizure 'as'  eg 189
    if non_seiz_start_time > 6 and non_seiz_start_time<duration:
       # check if non_seiz_start_time_bs is same from being overlapped by a seizure
       for s,e in lst_seiz_start_stop_time:
           if non_seiz_start_time > (s-6) and non_seiz_start_time < (e+6):
              if bs_or_as == 'bs':
                 non_seiz_start_time = s - (6+5+6)
              else:
                 non_seiz_start_time = e + (6+5+6)
       # no need to give any breaks, remove duplicates later by loading in a df before training
       pstrls = red_df.pstr.values[0] + '$' + red_df.label.values[0] + '$' + str(non_seiz_start_time)
       if pstrls in lst_pstrl_non_seiz_st_time:
          final_non_seiz_time = 0
       else:
          lst_pstrl_non_seiz_st_time.append(pstrls)
          final_non_seiz_time = non_seiz_start_time
    return final_non_seiz_time
    
    
def get_non_seiz_start_stop_time_OLD(red_df,bs_or_as,sz_start_or_stop_time):
    # if bs_or_as = bs then sz_start_or_stop_time has sz_start_time else sz_stop_time
    ## can sort the red_df start time wise, but doesn't effect
    #collect list of paired start stop times of seizures 
    lst_seiz_start_stop_time = list(zip(red_df.start_time, red_df.stop_time))
    final_non_seiz_time = 0 # to be returned if nothing fits
    if bs_or_as == 'bs':
       non_seiz_start_time = sz_start_or_stop_time - (6+5+6)   # before seizure 'bs'
    else:
       non_seiz_start_time = sz_start_or_stop_time + (6+5+6) # after 1 seizure 'as'  eg 189
    if non_seiz_start_time > 6:
       # check if non_seiz_start_time_bs is same from being overlapped by a seizure
       for s,e in lst_seiz_start_stop_time:
           if non_seiz_start_time > (s-6) and non_seiz_start_time < (e+6):
              if bs_or_as == 'bs':
                 non_seiz_start_time = s - (6+5+6)
              else:
                 non_seiz_start_time = e + (6+5+6)
              # no need to give any breaks, remove duplicates later by loading in a df before training
       final_non_seiz_time = non_seiz_start_time
    return final_non_seiz_time
    
    
def create_non_seiz_rec_for_pstr_wd_sz_st_sp(df,pstr,label,confidence,sz_start_time,sz_stop_time,duration,channel=None):
    #WE DONT PASS CHANNEL INFO, SO NON SEIZ IS CHECKED THROUGHOUT ON ALL CHNLS FOR SAME PSTR
    red_df = df.loc[(df['pstr']==pstr) & (df['label']==label) & (df['confidence']==confidence)]
    ###print(len(red_df))
    '''
    # non_seiz_start_time should be atleast 12+6 sec before seizure start time & should not fall in prev ictal
     let presumed non_seiz_time be 25 sec before first start of seiz time after sorting the red_df start_time wise 
    .. 6sec - non_seiz - 6 sec | gap of 5 sec | 6 sec -seiz - 6 sec | gap of 5 sec | 6 sec - non_seiz - 6 sec
    # let non_seiz_time be an instance just like seizure start_time we presume 2 non-seizure times & try insertion 
    for each electrode '''
    if red_df.empty:
       non_seiz_st_bs,non_seiz_st_as = 0,0
    else:
       # if for same pstr & label, multiple records exist for same electrode, 
       #    the non-seizure time taken should not be in between any of them
       # one non_seiz before a seiz onset
       non_seiz_st_bs = get_non_seiz_start_stop_time(red_df,'bs',sz_start_time,duration)
       # one non_seiz after a seiz onset
       non_seiz_st_as = get_non_seiz_start_stop_time(red_df,'as',sz_stop_time,duration)
    return non_seiz_st_bs, non_seiz_st_as
    
    
def get_all_elec_dependent_params(raw, ref_electrode, start_time, electrode, lst_ch_names, tcp_ref, seiz_onset, non_seizure_occurrence, remarks):
    #fetch t1...t12
    bos_vec, dos_vec, aos_vec = get_signal_vectors(raw, ref_electrode, start_time)
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12 = bos_vec[0],bos_vec[1],bos_vec[2],bos_vec[3],dos_vec[0],dos_vec[1],dos_vec[2],dos_vec[3],aos_vec[0],aos_vec[1],aos_vec[2],aos_vec[3]
    #fetch nn1....nn4
    nn1,nn2,nn3,nn4 = get_nn_values(electrode)
    #fetch c_bos_nn1...c_aos_nn4
    c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4 = fetch_corr_with_nn(raw,lst_ch_names,tcp_ref,bos_vec,dos_vec,aos_vec,nn1,nn2,nn3,nn4,start_time) 
    # target variable seizure means 1, no seizure means 0
    seiz_onset = seiz_onset
    non_seizure_occurrence = non_seizure_occurrence # will store 'bs' or 'as' only for non_seizure, 'bg' fo bckg, 'ts' for term seizure
    remarks = remarks
    return bos_vec, dos_vec, aos_vec, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, nn1,nn2,nn3,nn4, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, seiz_onset, non_seizure_occurrence, remarks 
    
    
    
# function to insert non seizure records in seiz_non_seiz table
def process_insertion_for_non_seiz(pstr, label, electrode, non_seiz_bs_or_as, raw, ref_electrode, lst_ch_names, tcp_ref, seizure_onset, ns_or_bs, remarks, confidence, file_path, dataset_type):
    num_rows_inserted = 0
    global lst_pstr_label_elec_nsbst_nsast_done
    try:
       # DOING FOR non_seiz_bs 
       start_time = non_seiz_bs_or_as # only for passing in function get_all_elec_dependent_params
       pstr_label_elec_nsbst_or_nsast = pstr + '#' + label + '#' + electrode + '#' + str(non_seiz_bs_or_as)
       if non_seiz_bs_or_as >6 and pstr_label_elec_nsbst_or_nsast not in lst_pstr_label_elec_nsbst_nsast_done:
          # then only do this non seizure insertion for this electrode else move ahead
          #call function get_all_elec_dependent_params to get all params values
          bos_vec, dos_vec, aos_vec, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, nn1,nn2,nn3,nn4, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, seiz_onset, non_seizure_occurrence, remarks = get_all_elec_dependent_params(raw, ref_electrode, start_time, electrode, lst_ch_names, row.tcp_ref, seizure_onset, ns_or_bs, remarks)
          print(pstr, label, electrode, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, nn1, nn2, nn3, nn4, seiz_onset, start_time, stop_time, confidence, file_path, non_seizure_occurrence, dataset_type, remarks)
          ## insert values in table for each electrode
          start_time, stop_time = non_seiz_bs_or_as -6, non_seiz_bs_or_as + 6 # only for inserting non_seiz duration in table
          num_rows_inserted = insert_in_seiz_non_seiz(pstr, label, electrode, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, nn1, nn2, nn3, nn4, seiz_onset, start_time, stop_time, confidence, file_path, non_seizure_occurrence, dataset_type, remarks) 
          ## if insertion happens successfully, then only insert pstr_label_elec_nsbst_nsast in the list
       if num_rows_inserted > 0:
          lst_pstr_label_elec_nsbst_nsast_done.append(pstr_label_elec_nsbst_or_nsast)
    except Exception as ex:
       num_rows_inserted = 0
       pass
    finally:
       return num_rows_inserted
       
       
def get_non_depenedent_elect_params(label, start_time, stop_time, confidence, file_path):
    # type of seizure 
    label = label
    # actual start time, end time and confidence of having seizure
    start_time, stop_time, confidence = start_time, stop_time, confidence
    #file path to load as raw signal
    file_path = file_path.replace('csv','edf')
    # remarks store train dev or val - to denote type of data for use in model
    dataset_type = get_dataset_type(file_path)
    return label, start_time, stop_time, confidence, file_path, dataset_type
    
    
## TERM based data entry to database related functions
# function to create timings for which bckg records are to be discretized, return a list of such start timings
def create_non_seiz_rec_for_pstrt_bckg(bckg_start_time,bckg_stop_time):
    lst_bckg_timings = []
    dur_bckg = bckg_stop_time - bckg_start_time
    # let one non seiz period be of 12 sec, so with buffer we need - 6+12+6 sec for one record i.e. 25 sec
    # for each record we need 4 timings 
    quartile_dur = math.floor(dur_bckg/4)
    # if bckg occurs for less than 25 sec in a record, we just ignore it
    if dur_bckg < 25:
       st_1, st_2, st_3, st_4 = 0, 0, 0, 0
    elif quartile_dur < 25:   #(min will be )
       num_valid_pts = math.floor(dur_bckg/25)
       if num_valid_pts == 1:
          st_1, st_2, st_3, st_4 = bckg_start_time+12.5, 0, 0, 0
       elif num_valid_pts == 2:
          st_1, st_2, st_3, st_4 = bckg_start_time+12.5, bckg_start_time+ 37.5, 0, 0
       else:
          st_1, st_2, st_3, st_4 = bckg_start_time+12.5, bckg_start_time+ 37.5, bckg_start_time+ 62.5, 0
    else:
       st_1 = bckg_start_time + quartile_dur/2                          # min 12.5 
       st_2 = bckg_start_time + quartile_dur + (quartile_dur/2)         #     37.5
       st_3 = bckg_start_time + (2*quartile_dur) + (quartile_dur/2)     #     62.5
       st_4 = bckg_start_time + (3*quartile_dur) + (quartile_dur/2)     #     87.5
    lst_bckg_timings.append(st_1)
    lst_bckg_timings.append(st_2)
    lst_bckg_timings.append(st_3)
    lst_bckg_timings.append(st_4)
    return lst_bckg_timings
    # bckg_start_time should be atleast 12+6 sec before seizure start time if there exist a seiz & should not fall in previou sictal 
    
    
def process_insertion_for_bckg(pstrt, patient_id, pstr, label, electrode, st, raw, ref_electrode, lst_ch_names, tcp_ref, seizure_onset, ns_or_bs, remarks, confidence, file_path, dataset_type):
    num_rows_inserted = 0
    try:
       # DOING FOR bckg 
       start_time = st # only for passing in function get_all_elec_dependent_params
       if term_seiz == 0:
          # then only do this bckg insertion for this electrode else move ahead
          #call function get_all_elec_dependent_params to get all params values
          bos_vec, dos_vec, aos_vec, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, nn1,nn2,nn3,nn4, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, seiz_onset, non_seizure_occurrence, remarks = get_all_elec_dependent_params(raw, ref_electrode, start_time, electrode, lst_ch_names, row.tcp_ref, 0, 'bg', remarks)
          '''print(pstrt, patient_id, pstr, label, electrode, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, c_bos_nn1,
                c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,
                c_dos_nn4,c_aos_nn4, nn1, nn2, nn3, nn4, seiz_onset, start_time, stop_time, confidence, 
                file_path, non_seizure_occurrence, dataset_type, term_seiz, remarks)    '''
          ## insert values in table for each electrode
          num_rows_inserted = insert_in_term_seiz_non_seiz(pstrt, patient_id, pstr, label, electrode, t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12, c_bos_nn1,c_dos_nn1,c_aos_nn1,c_bos_nn2,c_dos_nn2,c_aos_nn2,c_bos_nn3,c_dos_nn3,c_aos_nn3,c_bos_nn4,c_dos_nn4,c_aos_nn4, nn1, nn2, nn3, nn4, seiz_onset, start_time, stop_time, confidence, file_path, non_seizure_occurrence, dataset_type, remarks) 
          # if insertion happens successfully, then only insert pstr_label_elec_nsbst_nsast in the list
    except Exception as ex:
       num_rows_inserted = 0
       pass
    finally:
       return num_rows_inserted
       
       
# Mean releated function
# Function to fetch the aggregated split matrix for mean of correlation
def get_agg_split_nn_chnl_dfs(df):
    nn1_df = df[['patient_id', 'label', 'electrode', 'nn1','c_bos_nn1', 'c_dos_nn1', 'c_aos_nn1']]
    nn2_df = df[['patient_id', 'label', 'electrode', 'nn2','c_bos_nn2', 'c_dos_nn2', 'c_aos_nn2']]
    nn3_df = df[['patient_id', 'label', 'electrode', 'nn3','c_bos_nn3', 'c_dos_nn3', 'c_aos_nn3']]
    nn4_df = df[['patient_id', 'label', 'electrode', 'nn4','c_bos_nn4', 'c_dos_nn4', 'c_aos_nn4']]
    # check and remove if blank nn occurs, since nn2, nn3, nn4 at times also has no value and 0 corr
    nn1_df = nn1_df.loc[nn1_df['nn1']!=' ']   # non_seiz - 0 blanks nn1
    nn2_df = nn2_df.loc[nn2_df['nn2']!=' ']   # non_seiz - 43779-41614 blanks nn1 removed
    nn3_df = nn3_df.loc[nn3_df['nn3']!=' ']   # non_seiz - 43779-41614 blanks nn2 removed
    nn4_df = nn4_df.loc[nn4_df['nn4']!=' ']   # non_seiz - 43779-24327 blanks nn3 removed
    # do aggregations on correaltions
    agg_nn1_df = nn1_df.groupby(['patient_id','label','electrode','nn1']).agg({'c_bos_nn1': np.mean,'c_dos_nn1': np.mean,'c_aos_nn1': np.mean }).reset_index() #3063
    agg_nn2_df = nn2_df.groupby(['patient_id','label','electrode','nn2']).agg({'c_bos_nn2': np.mean,'c_dos_nn2': np.mean,'c_aos_nn2': np.mean }).reset_index() #2853
    agg_nn3_df = nn3_df.groupby(['patient_id','label','electrode','nn3']).agg({'c_bos_nn3': np.mean,'c_dos_nn3': np.mean,'c_aos_nn3': np.mean }).reset_index() #2853
    agg_nn4_df = nn4_df.groupby(['patient_id','label','electrode','nn4']).agg({'c_bos_nn4': np.mean,'c_dos_nn4': np.mean,'c_aos_nn4': np.mean }).reset_index() #1617
    # RENAME ALL COLUMNS TO MERGE LATER
    agg_nn1_df.rename(columns={'nn1':'nn', 'c_bos_nn1':'c_bos_nn', 'c_dos_nn1':'c_dos_nn', 'c_aos_nn1':'c_aos_nn'}, inplace=True)
    agg_nn2_df.rename(columns={'nn2':'nn', 'c_bos_nn2':'c_bos_nn', 'c_dos_nn2':'c_dos_nn', 'c_aos_nn2':'c_aos_nn'}, inplace=True)
    agg_nn3_df.rename(columns={'nn3':'nn', 'c_bos_nn3':'c_bos_nn', 'c_dos_nn3':'c_dos_nn', 'c_aos_nn3':'c_aos_nn'}, inplace=True)
    agg_nn4_df.rename(columns={'nn4':'nn', 'c_bos_nn4':'c_bos_nn', 'c_dos_nn4':'c_dos_nn', 'c_aos_nn4':'c_aos_nn'}, inplace=True)
    return agg_nn1_df, agg_nn2_df, agg_nn3_df, agg_nn4_df
    
    
# function to spit a main df into nn dfs and then aggregate them after removing blank nns
def get_agg_split_nn_dfs(df):
    nn1_df = df[['patient_id', 'electrode', 'nn1','c_bos_nn1', 'c_dos_nn1', 'c_aos_nn1']] #len-274039 for bckg term
    nn2_df = df[['patient_id', 'electrode', 'nn2','c_bos_nn2', 'c_dos_nn2', 'c_aos_nn2']] #len-274039 for bckg term
    nn3_df = df[['patient_id', 'electrode', 'nn3','c_bos_nn3', 'c_dos_nn3', 'c_aos_nn3']] #len-274039 for bckg term
    nn4_df = df[['patient_id', 'electrode', 'nn4','c_bos_nn4', 'c_dos_nn4', 'c_aos_nn4']] #len-274039 for bckg term
    #### Since nn2, nn3, nn4 at times also has no value and 0 corr, so we remove them from df for correct mean calculation
    nn1_df = nn1_df.loc[nn1_df['nn1']!=' '] # no reduction as all electrodes have at least 1 nn
    nn2_df = nn2_df.loc[nn2_df['nn2']!=' '] # corr_descrtz_bckg_term_nn2_df - 274039 reduced to 252503
    nn3_df = nn3_df.loc[nn3_df['nn3']!=' '] # corr_descrtz_bckg_term_nn3_df - 274039 reduced to 252503
    nn4_df = nn4_df.loc[nn4_df['nn4']!=' '] # corr_descrtz_bckg_term_nn4_df - 274039 reduced to 119637
    # Aggregating over nn corr values to get mean for respective dfs
    agg_nn1_df = nn1_df.groupby(['patient_id','electrode','nn1']).agg({'c_bos_nn1': np.mean,'c_dos_nn1': np.mean,'c_aos_nn1': np.mean }).reset_index() # 7338
    agg_nn2_df = nn2_df.groupby(['patient_id','electrode','nn2']).agg({'c_bos_nn2': np.mean,'c_dos_nn2': np.mean,'c_aos_nn2': np.mean }).reset_index() # 6764
    agg_nn3_df = nn3_df.groupby(['patient_id','electrode','nn3']).agg({'c_bos_nn3': np.mean,'c_dos_nn3': np.mean,'c_aos_nn3': np.mean }).reset_index() # 6764
    agg_nn4_df = nn4_df.groupby(['patient_id','electrode','nn4']).agg({'c_bos_nn4': np.mean,'c_dos_nn4': np.mean,'c_aos_nn4': np.mean }).reset_index() # 3204  
    ## rename all columns of these 4 dfs to make common and then merge
    agg_nn1_df.rename(columns={'nn1':'nn', 'c_bos_nn1':'c_bos_nn', 'c_dos_nn1':'c_dos_nn', 'c_aos_nn1':'c_aos_nn'}, inplace=True)
    agg_nn2_df.rename(columns={'nn2':'nn', 'c_bos_nn2':'c_bos_nn', 'c_dos_nn2':'c_dos_nn', 'c_aos_nn2':'c_aos_nn'}, inplace=True)
    agg_nn3_df.rename(columns={'nn3':'nn', 'c_bos_nn3':'c_bos_nn', 'c_dos_nn3':'c_dos_nn', 'c_aos_nn3':'c_aos_nn'}, inplace=True)
    agg_nn4_df.rename(columns={'nn4':'nn', 'c_bos_nn4':'c_bos_nn', 'c_dos_nn4':'c_dos_nn', 'c_aos_nn4':'c_aos_nn'}, inplace=True)
    return agg_nn1_df, agg_nn2_df, agg_nn3_df, agg_nn4_df
    
    
# frame one df from there two df with proper names of mean columns to be compared
def merge_seiz_non_seiz_corr_mean_dfs(df1,df2):
    #df be the seizure corr mean df for patient, df2 be bckg corr mean for same patient
    lst_bckg_mean_vals = []
    for row in df1.itertuples(index=False):
        nn_nn = row.nn_nn
        nn_nn_rev = nn_nn.split('_')[-1] + '_' + nn_nn.split('_')[0]
        lst_bckg_mean_vals.append(df2.loc[(df2['nn_nn']==nn_nn) | (df2['nn_nn']==nn_nn_rev)]['mean_corr'].values[0])
    df1['non_seiz_mean_corr'] = lst_bckg_mean_vals
    return df1
    
    
# function to print the correlations mean of seiz and bckg using seaborn
def plot_corr_mean_comparison(df):
    rcParams['figure.figsize'] = 16, 5
    print('Seiz - Non-Seiz Mean Corr. Graph for patient: ', df.patient_id.values[0])
    df = df.reset_index()
    df = df[['nn_nn','seiz_mean_corr','non_seiz_mean_corr']]
    # convert to long (tidy) form
    dfm = df.melt('nn_nn', var_name='cols', value_name='mean_corr')
    #g = sns.catplot(x="nn_nn", y="vals", hue='cols', data=dfm, kind='point')
    sns.pointplot(x="nn_nn", y="mean_corr", hue='cols', data=dfm, height=15, aspect=0.8)
    plt.show()
    print('---------------------------------------------------------------')
    print('                                                               ')
    print('                                                               ')
    
    
    
## Other functions related to EDA
def fetch_concat_stt(a,b,c):
    return a+'_'+b+'_'+c
    
    
def get_filtered_df(seiz_channel_df, seiz_type):
    lst_fnsz_pat, lst_unq_stt = [], []
    seiz_fnsz_channel_df = seiz_channel_df.loc[seiz_channel_df['label']==seiz_type] #'fnsz'
    lst_fnsz_pat = list(set(list(seiz_fnsz_channel_df['patient_id'])))
    seiz_fnsz_channel_df['idx_unq_pat'] = seiz_fnsz_channel_df['patient_id'].apply(lambda x: lst_fnsz_pat.index(x))
    #selecting records of just 5 patients and 5 sessions and 4 tids and tcp_ref
    seiz_fnsz_channel_stt_df = seiz_fnsz_channel_df.loc[seiz_fnsz_channel_df['idx_unq_pat']<5]
    seiz_fnsz_channel_stt_df['stt'] = seiz_fnsz_channel_stt_df.apply(lambda row: fetch_concat_stt(row['session_id'], row['t_id'], row['tcp_ref']), axis=1)
    lst_unq_stt = seiz_fnsz_channel_stt_df['stt'].unique()
    lst_unq_stt = list(set(lst_unq_stt))
    seiz_fnsz_channel_stt_df['idx_unq_stt'] = seiz_fnsz_channel_stt_df['stt'].apply(lambda x: lst_unq_stt.index(x))
    seiz_fnsz_channel_stt_red_df = seiz_fnsz_channel_stt_df.loc[seiz_fnsz_channel_stt_df['idx_unq_stt']<5]
    seiz_fnsz_channel_stt_red_df = seiz_fnsz_channel_stt_red_df.sort_values(by=['patient_id', 'start_time'])
    lst_5_unq_stt = seiz_fnsz_channel_stt_red_df['stt'].unique() 
    return seiz_fnsz_channel_stt_red_df
    
    
def fetch_lst_unq_elec_seiz_together(df):
    lst_unq_elec = list(df.electrode.value_counts().index)
    lst_elec, lst_lst_pat, lst_pat_cnt, lst_lst_pstr, lst_lst_pstr_count, lst_lst_pat_wise_elec, lst_lst_pat_wise_all_elec, lst_lst_elec_cnt = [], [], [], [], [], [], [], []
    lst_unq_pat = list(df.patient_id.unique())
    for curr_electrode in lst_unq_elec:  # will go in desc order of value counts for this df
        lst_elec.append(curr_electrode)
        lst_pat, lst_pstr, lst_pstr_count, lst_pat_wise_elec, lst_elec_cnt, lst_all_elec = [], [], [], [], [], []
        for item in lst_unq_pat:  # 173 patients one by one
            pat_wise_df = df.loc[df['patient_id']==item]
            lst_pat_elec, lst_pat_all_elec, lst_pat_pstr = [], [], []
            curr_electrode_presence = False
            for row in pat_wise_df.itertuples(index=False):
                if not row.electrode in lst_pat_elec:
                   lst_pat_elec.append(row.electrode)
                if row.electrode == curr_electrode:
                   curr_electrode_presence = True
                if not row.pstr in lst_pat_pstr: 
                   lst_pat_pstr.append(row.pstr)
                lst_pat_all_elec.append(row.electrode)
            if curr_electrode_presence:
               lst_pat.append(item)
               lst_pstr.append(lst_pat_pstr)
               lst_pstr_count.append(len(lst_pat_pstr))
               lst_pat_wise_elec.append(lst_pat_elec)
               lst_elec_cnt.append(len(lst_pat_elec))
               lst_all_elec.append(lst_pat_all_elec)
        lst_lst_pat.append(lst_pat) 
        lst_pat_cnt.append(len(lst_pat))
        lst_lst_pstr.append(lst_pstr)
        lst_lst_pstr_count.append(lst_pstr_count)
        lst_lst_pat_wise_elec.append(lst_pat_wise_elec)
        lst_lst_elec_cnt.append(lst_elec_cnt)
        lst_lst_pat_wise_all_elec.append(lst_all_elec)
    pdf_data = {'electrode': lst_elec, 'lst_patient_ids': lst_lst_pat, 'cnt_pat':lst_pat_cnt, 'lst_pstr': lst_lst_pstr, 'pstr_count':lst_lst_pstr_count, 
                'lst_pat_wise_unq_elec': lst_lst_pat_wise_elec,  'unq_elec_count': lst_lst_elec_cnt, 'lst_pat_wise_all_elec':lst_lst_pat_wise_all_elec}
    pat_wise_elec_rel_df = pd.DataFrame(pdf_data, columns = ['electrode','lst_patient_ids','cnt_pat','lst_pstr','pstr_count',
                                                               'lst_pat_wise_unq_elec','unq_elec_count','lst_pat_wise_all_elec'])
    return pat_wise_elec_rel_df
    
    
# function to get electrodes present and their value counts when one electrode is present patient wise
def get_rel_elec_val_counts(curr_elec, lst_lst_pat_wise_all_elec):
    merged_list_of_all_elec, lst_rel_elec, lst_rel_elec_val_counts = [], [], []
    for lst_pat_wise_all_elec in lst_lst_pat_wise_all_elec:
        merged_list_of_all_elec += lst_pat_wise_all_elec
    merged_list_of_all_elec = list(filter(lambda a: a != curr_elec, merged_list_of_all_elec))
    lst_rel_elec = Counter(merged_list_of_all_elec).keys() # equals to list(set(words))
    lst_rel_elec_val_counts = Counter(merged_list_of_all_elec).values() # counts the elements' frequency
    return dict(zip(lst_rel_elec,lst_rel_elec_val_counts)) 
    
    
# function to plot current electrode with related electrodes and their frequency of occurence
def plot_elec_pat_rel_elec(dict_rel_elec_cnt,electrode):
    #for row in df.itertuples(index=False):
    print('\n\n')
    print('related electrodes presence for ',electrode)
    lists = sorted(row.dict_rel_elec_cnt.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()
    print('################################################################################')
    
    
#function to fetch mean and std of the dict
def get_mean_and_std(dct,mean_or_std):
    listr = []
    # appending all the values in the list
    for value in dct.values():
        listr.append(value)
    if mean_or_std == 'mean':
       # calculating mean using np.mean
       mean = np.mean(listr)
       return mean
    else: 
       # calculating standard deviation using np.std
       std = np.std(listr)
       return std
       
       
# function to return rel electrodes of current electrode with count > mean value in desc order as list of elec
def get_higher_rel_elec(dct,threshold=None):
    lst_key = []
    sorted_dct_desc = dict(sorted(dct.items(), key=operator.itemgetter(1),reverse=True))
    for key, value in sorted_dct_desc.items():
        if threshold is None:
           lst_key.append(key)
        elif value>threshold:
           lst_key.append(key)
    return lst_key
    
    
# function to get normalized occurence count for rel electrodes
def get_rel_elec_weights(dct):
    sorted_dct_desc = dict(sorted(dct.items(), key=operator.itemgetter(1),reverse=True))
    values = sorted_dct_desc.values()
    min_ = min(values)
    max_ = max(values)
    normalized_d = {key: round(((v - min_ ) / (max_ - min_) ),2)  for (key, v) in sorted_dct_desc.items() }
    return normalized_d
    
    
def get_elec_features(df):
    df = fetch_lst_unq_elec_seiz_together(df)
    df['dict_rel_elec_cnt'] = df.apply(lambda row: get_rel_elec_val_counts(row.electrode,row.lst_pat_wise_all_elec), axis=1)
    df['rel_elec_cnt_mean'] = df['dict_rel_elec_cnt'].apply(lambda x: get_mean_and_std(x,'mean'))
    df['rel_elec_with_high_occ'] = df.apply(lambda row: get_higher_rel_elec(row.dict_rel_elec_cnt,row.rel_elec_cnt_mean),axis=1)
    df['rel_elec_with_occ'] = df.apply(lambda row: get_higher_rel_elec(row.dict_rel_elec_cnt),axis=1)
    df['rel_elec_weights'] = df['dict_rel_elec_cnt'].apply(lambda x: get_rel_elec_weights(x))
    df['rel_elec_cnt_std'] = df['dict_rel_elec_cnt'].apply(lambda x: get_mean_and_std(x,'std'))
    df['rel_elec_cnt_mean_plus_std'] = df.apply(lambda row: (row.rel_elec_cnt_mean + row.rel_elec_cnt_std),axis=1)
    df['rel_elec_cnt_mean_minus_std'] = df.apply(lambda row: (row.rel_elec_cnt_mean - row.rel_elec_cnt_std),axis=1)
    return df
    
    
def merge_high_occ_with_1hop_neighbor(high_occ_df):
    nn_df = fetch_neighbor_elec_mapping('../data_files/channels.csv')
    lst_electrodes = list(high_occ_df['electrode'])
    lst_lst_nn_elec = []
    for item in lst_electrodes:
        fil_nn_df = nn_df.loc[nn_df['electrodes']==item]
        if not fil_nn_df.empty:
           lst_lst_nn_elec.append(fil_nn_df.neighbor.values[0])
        else:
           lst_lst_nn_elec.append([])
    high_occ_df['hop1_neighbors'] = lst_lst_nn_elec
    high_occ_df['nn_abs_in_rel_elec_with_occ'] = high_occ_df.apply(lambda row: ([x for x in row.hop1_neighbors if x not in row.rel_elec_with_occ]),axis=1)
    high_occ_df['nn_abs_in_rel_elec_with_high_occ'] = high_occ_df.apply(lambda row: ([x for x in row.hop1_neighbors if x not in row.rel_elec_with_high_occ]),axis=1)
    return high_occ_df
    
    
# function to return a list of high occurence electrodes
def get_list_high_occ_elec(df):
    # finalize a common rel_elec_with_high_occ based on scores of rel_elec_weights summing over all 19 rows
    lst = []
    for row in df.itertuples(index=False): 
        lst += row.rel_elec_with_high_occ
    x = Counter(lst)
    lst_desc_elec_counts = [(l,k) for k,l in sorted([(j,i) for i,j in x.items()], reverse=True)]
    print(lst_desc_elec_counts)
    lst_desc_elec_counts_key = [x for (x,y) in lst_desc_elec_counts]
    return lst_desc_elec_counts_key
    
    
def fetch_image_of_arrays():
    # Normalize the arrays to be in the range of 0 to 255
    array1 = (array1 * 255).astype(np.uint8)
    array2 = (array2 * 255).astype(np.uint8)
    array3 = (array3 * 255).astype(np.uint8)
    
    # Create an empty image with the size of the grid
    image = Image.new('RGB', (grid_size[0] * 100, grid_size[1] * 100))
    
    # Paste the arrays onto the image
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x_offset = i * 100
            y_offset = j * 100
            array = [array1, array2, array3][(i+j)%3]  # alternate the arrays
            sub_image = Image.fromarray(array)
            image.paste(sub_image, (x_offset, y_offset))
    
    # Display the image
    image.show()
    
    
def fetch_arr_descrtzd_vector_of_channel(raw, start_time, stop_time, ref_type=None): #'EEG C4-REF', 'EEG P4-REF'
    arr_chnl_vectors = np.zeros((21,4))
    lst_ch_names = raw.ch_names[:21]
    i = 0
    for item in lst_ch_names:
        #print(item)
        # get raw signal only of a particular period
        data_ch_4sec, times = raw.get_data(picks=item, return_times=True, start=start_time, stop=stop_time)
        # set notch filter to remove line noise and remove noise using butterworth filter for the channel
        notch_freq = 50 
        nyquist_freq = raw.info['sfreq'] / 2.0 
        b, a = butter(4, [(notch_freq - 2.0) / nyquist_freq, (notch_freq + 2.0) / nyquist_freq], btype='bandstop')
        # converting the continuous signal into a discrete sequence
        data_ch_4sec = filtfilt(b, a, data_ch_4sec, padtype=None, padlen=None)
        #print(data_ch_4sec[0])
        #break
        np_arr = np.array(data_ch_4sec[0])
        for j in range(len(np_arr)):
            arr_chnl_vectors[i][j]= np_arr[j]  
        #lst_chnl_vectors.append(data_ch_4sec)
        i+=1
    return arr_chnl_vectors
    
    
def fetch_descrtzd_vector_of_channel(raw, start_time, stop_time, ref_type=None): #'EEG C4-REF', 'EEG P4-REF'
    lst_chnl_vectors = []
    lst_ch_names = raw.ch_names[:21]
    for item in lst_ch_names:
        #print(item)
        # get raw signal only of a particular period
        data_ch_4sec, times = raw.get_data(picks=item, return_times=True, start=start_time, stop=stop_time)
        # set notch filter to remove line noise and remove noise using butterworth filter for the channel
        notch_freq = 50 
        nyquist_freq = raw.info['sfreq'] / 2.0 
        b, a = butter(4, [(notch_freq - 2.0) / nyquist_freq, (notch_freq + 2.0) / nyquist_freq], btype='bandstop')
        # converting the continuous signal into a discrete sequence
        data_ch_4sec = filtfilt(b, a, data_ch_4sec, padtype=None, padlen=None)
        lst_chnl_vectors.append(data_ch_4sec)
    return lst_chnl_vectors
    
    
def fetch_corr_bw_channels(raw, ch_name1, ch_name2, start_time, stop_time): #'EEG C4-REF', 'EEG P4-REF'
    data_ch1_4sec, times = raw.get_data(picks=ch_name1, return_times=True, start=start_time, stop=stop_time)
    data_ch2_4sec, times = raw.get_data(picks=ch_name2, return_times=True, start=start_time, stop=stop_time)
    notch_freq = 50 # notch filter to remove line noise
    nyquist_freq = raw.info['sfreq'] / 2.0 
    b, a = butter(4, [(notch_freq - 2.0) / nyquist_freq, (notch_freq + 2.0) / nyquist_freq], btype='bandstop')
    # converting the continuous signal into a discrete sequence
    data_ch1_4sec = filtfilt(b, a, data_ch1_4sec, padtype=None, padlen=None)
    data_ch2_4sec = filtfilt(b, a, data_ch2_4sec, padtype=None, padlen=None)
    corr, p_value = pearsonr(data_ch1_4sec.ravel(), data_ch2_4sec.ravel())
    return round(corr,2)
    
    
#function to fetch diff of correlations between different nearest neighbors during and before onset of seizures
def fetch_diff_corr_dos_bos_nn(df):
    df['diff_corr_dos_bos_nn1'] = df['c_dos_nn1'] - df['c_bos_nn1']
    df['diff_corr_dos_bos_nn2'] = df['c_dos_nn2'] - df['c_bos_nn2']
    df['diff_corr_dos_bos_nn3'] = df['c_dos_nn3'] - df['c_bos_nn3']
    df['diff_corr_dos_bos_nn4'] = df['c_dos_nn4'] - df['c_bos_nn4']
    return df
    
    
#function to fetch diff of correlations between different nearest neighbors after and during onset of seizures
def fetch_diff_corr_aos_dos_nn(df):
    df['diff_corr_aos_dos_nn1'] = df['c_aos_nn1'] - df['c_dos_nn1']
    df['diff_corr_aos_dos_nn2'] = df['c_aos_nn2'] - df['c_dos_nn2']
    df['diff_corr_aos_dos_nn3'] = df['c_aos_nn3'] - df['c_dos_nn3']
    df['diff_corr_aos_dos_nn4'] = df['c_aos_nn4'] - df['c_dos_nn4']
    return df
    
    
# Define function for feature extraction
def extract_features(data):
    # Apply time-frequency analysis using Morlet wavelets
    freqs = np.arange(1, 40, 1)
    n_cycles = freqs / 2.0
    power, phase_lock = mne.time_frequency.tfr_array_morlet(data, freqs=freqs, n_cycles=n_cycles, sfreq=250)
    
    # Calculate average power across frequency bands
    delta_power = np.mean(power[:, :, :, 0:4], axis=3)
    theta_power = np.mean(power[:, :, :, 4:8], axis=3)
    alpha_power = np.mean(power[:, :, :, 8:13], axis=3)
    beta_power = np.mean(power[:, :, :, 13:30], axis=3)
    gamma_power = np.mean(power[:, :, :, 30:40], axis=3)
    
    # Calculate other features, such as variance, skewness, kurtosis, etc.
    var = np.var(data, axis=1)
    skew = np.apply_along_axis(lambda x: mne.features.skewness(x), 1, data)
    kurt = np.apply_along_axis(lambda x: mne.features.kurtosis(x), 1, data)
    
    # Combine features into a single matrix
    features = np.hstack((delta_power, theta_power, alpha_power, beta_power, gamma_power, var, skew, kurt))
    return features
    
    

    
    
def print_brain_corr(ls,lst):
    print('                                 '+ls[0]+':'+str(lst[0])+'                                     ')
    print('                                                                                               ')
    print('        '+ls[1]+':'+str(lst[1])+'    '+ls[2]+':'+str(lst[2])+'          '+ls[3]+':'+str(lst[3])+'    '+ls[4]+':'+str(lst[4])+'     ')
    print('                                                                                               ')
    print('              '+ls[5]+':'+str(lst[5])+'    '+ls[6]+':'+str(lst[6])+'    '+ls[7]+':'+str(lst[7])+'    '+ls[8]+':'+str(lst[8])+'        ')
    print('                                                                                               ')
    print('       '+ls[9]+':'+str(lst[9])+'    '+ls[10]+':'+str(lst[10])+'    '+ls[11]+':'+str(lst[11])+'    '+ls[12]+':'+str(lst[12])+'    '+ls[13]+':'+str(lst[13])+'   ')
    print('                                                                                               ')
    print(ls[14]+':'+str(lst[14])+'    '+ls[15]+':'+str(lst[15])+'    '+ls[16]+':'+str(lst[16])+'    '+ls[17]+':'+str(lst[17])+'    '+ls[18]+':'+str(lst[18])+'    '+ls[19]+':'+str(lst[19]))
    print('                                                                                               ')
    print('       '+ls[20]+':'+str(lst[20])+'    '+ls[21]+':'+str(lst[21])+'    '+ls[22]+':'+str(lst[22])+'    '+ls[23]+':'+str(lst[23])+'    '+ls[24]+':'+str(lst[24])+'   ')
    print('                                                                                               ')
    print('              '+ls[25]+':'+str(lst[25])+'    '+ls[26]+':'+str(lst[26])+'    '+ls[27]+':'+str(lst[27])+'    '+ls[28]+':'+str(lst[28])+'        ')
    print('                                                                                               ')
    print('       '+ls[29]+':'+str(lst[29])+'    '+ls[30]+':'+str(lst[30])+'          '+ls[31]+':'+str(lst[31])+'    '+ls[32]+':'+str(lst[32])+'     ')
    print('                                                                                               ')
    print('                                 '+ls[33]+':'+str(lst[33])+'               ')
    
    
    

### Functions  to do Discrete Wavelet Transform (DWT)
def fetch_dwt(w,x,y,z,db,lvl,cA_or_cD=None):
    if lvl > 1:
        coeffs = pywt.wavedec([w,x,y,z], db, level=lvl)
        return coeffs
    else:
        cA, cD = pywt.dwt([w,x,y,z], db)
        if cA_or_cD == 'cA':
           return cA
        elif cA_or_cD == 'cD':
           return cD
           
           
def fetch_dwt_8s(q,r,s,t,w,x,y,z,db,lvl,cA_or_cD=None):
    if lvl > 1:
        coeffs = pywt.wavedec([q,r,s,t,w,x,y,z], db, level=lvl)
        return coeffs
    else:
        cA, cD = pywt.dwt([q,r,s,t,w,x,y,z], db)
        if cA_or_cD == 'cA':
           return cA
        elif cA_or_cD == 'cD':
           return cD    


    
##Classifiers related functions
#function to get predictions
def process_pred(X,y,z,clf,i_number_trees=None):
    y=y.astype('int')
    #regr_1.fit(X, np.ravel(y,order='C'))  
    clf.fit(X, y)  
    y_pred = clf.predict(z)
    return clf, y_pred
    
    
# function to train and test model
def train_test_model(clf,train_df,test_df,i_number_trees=1):
    ohne_pat_train_df = train_df.iloc[:,1:-1]
    ohne_pat_test_df = test_df.iloc[:,1:-1]
    X = np.array(ohne_pat_train_df)
    y = np.array(list(train_df['seiz_onset']))
    z = np.array(ohne_pat_test_df)
    y_act = np.array(list(test_df['seiz_onset']))
    clsfr, y_pred = process_pred(X,y,z,clf,i_number_trees)
    y_act = np.array(y_act).ravel()
    return clsfr, y_act, y_pred
    
    
#evaluate result, using precision, recall and F1 score as metrics
def evaluate_seizure_result(actual,predicted):
    #create a df of actual and pred for ease of work
    pdf_data = {'actual':list(actual),'predicted':list(predicted)}
    actl_pred_df = pd.DataFrame(pdf_data,columns=['actual','predicted'])
    #actuals
    tot_act_seiz = sum(actl_pred_df.actual)
    print('Total no. of actual seizures: ',tot_act_seiz)
    tot_act_nonSeiz = (len(actl_pred_df) - tot_act_seiz)
    print('Total no. of actual non-seizures: ',tot_act_nonSeiz)

    #predicted
    tot_recog_seiz = sum(actl_pred_df['predicted'])
    print('Total no. of predicted seizures: ',tot_recog_seiz)
    tot_recog_nonSeiz = len(actl_pred_df) - tot_recog_seiz
    print('Total no. of predicted non-seizures: ',tot_recog_nonSeiz)

    #total no. of correctly predicted seizures and correctly predicted nonSeizures
    corr_seiz = actl_pred_df.query('actual==1 & predicted==1')['predicted'].sum()
    corr_nonSeiz = len(actl_pred_df.query('actual==0 & predicted==0')['predicted'])
    print('total count of test set: ',len(actual))
    print('total no. of correctly predicted seizures onset windows: ',corr_seiz)
    print('total no. of correctly predicted nonSeizure windows:',corr_nonSeiz)

    precision_seiz = round((corr_seiz/tot_recog_seiz),2)
    precision_nonSeiz = round((corr_nonSeiz/tot_recog_nonSeiz),2)
    recall_seiz = round((corr_seiz/tot_act_seiz),2)
    recall_nonSeiz = round((corr_nonSeiz/tot_act_nonSeiz),2)

    seizF1 = round(((2*precision_seiz*recall_seiz)/(precision_seiz+recall_seiz)),2)
    nonSeizF1 = round(((2*precision_nonSeiz*recall_nonSeiz)/(precision_nonSeiz+recall_nonSeiz)),2)

    return precision_seiz,precision_nonSeiz,recall_seiz,recall_nonSeiz,seizF1,nonSeizF1
    
    


