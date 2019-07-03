

import os
import sys
import argparse
import glob
import re
import zipfile
import numpy as np
import pandas as pd
from numpy import nan as NaN

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        Input: Folder
        """)

    parser.add_argument(
        "-d", "--idir",
        required=True, nargs="+",
        help="Folder to be sorted",
        )

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output    folder - if doesnt exist it will be created",
        )

    parser.add_argument(
        "-v", "--verbose",
        required=False, nargs="+",
        help="Verbose to    get more information about what's going on",
        )

    args =  parser.parse_args()
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def get_all_ids(iDir):
    if not os.path.exists(iDir):
        sys.exit('This folder doesnt exist: {}'.format(iDir))
        return
    ids = []
    allZipFiles = glob.glob(os.path.join(iDir,'*.zip'))
    for currZipFile in allZipFiles:
        currZipFile = os.path.basename(currZipFile)
        ids.append((currZipFile.split('_')[0],currZipFile.split('_')[1]))
    return ids

def set_subject_data(idBeh, datadir, output):
    print(idBeh)
    sub_files = []
    s_dir = glob.glob(os.path.join(datadir, idBeh+'*IRM.zip'))
    if len(s_dir) != 1:
        print('Multiple directories match subject id '+idBeh)
    else:
        s_path = os.path.join(output, idBeh+'*')
        s_out = glob.glob(s_path)
        if len(s_out)<1:
            z_ref = zipfile.ZipFile(s_dir[0], 'r')
            z_ref.extractall(output)
            z_ref.close()
            s_out = glob.glob(s_path)
        prefix = ['Output-Responses-Encoding_CIMAQ_*', 'Onset-Event-Encoding_CIMAQ_*',
        'Output_Retrieval_CIMAQ_*']
        if len(s_out)==1:
            for i in range (0, 3):
                file = glob.glob(os.path.join(s_out[0], prefix[i]))
                if len(file)==1:
                    sub_files.append(file[0])
    return sub_files

def cleanMain(mainFile):
    #remove first three junk rows (blank trials): CTL0, Enc00 and ENc000
    mainFile.drop([0, 1, 2], axis=0, inplace=True)
    #re-label columns
    mainFile.rename(columns={'TrialNumber':'trial_number', 'Category':'trial_type',
    'OldNumber':'stim_id', 'CorrectSource':'position_correct', 'Stim_RESP':'response',
    'Stim_RT':'response_time'}, inplace=True)
    #remove redundant columns
    mainFile.drop(['TrialCode', 'Stim_ACC'], axis=1, inplace=True)
    #re-order columns
    cols = ['trial_number', 'trial_type', 'response', 'response_time',
    'stim_id', 'position_correct']
    mainFile = mainFile[cols]
    #change in-scan reaction time from ms to s
    mainFile['response_time']=mainFile['response_time'].astype('float64', copy=False)
    mainFile['response_time'] = mainFile['response_time'].div(1000)
    #insert new columns
    colNames = ['onset', 'duration', 'offset', 'stim_file', 'stim_category', 'stim_name',
    'recognition_accuracy', 'recognition_responsetime', 'position_response', 'position_accuracy',
    'position_responsetime']
    dtype = [NaN, NaN, NaN, 'None', 'None', 'None', -1, NaN, -1, -1, NaN]
    colIndex = [0, 1, 2, 8, 9, 10, 11, 12, 14, 15, 16]
    for i in range (0, 11):
        mainFile.insert(loc=colIndex[i], column=colNames[i], value=dtype[i], allow_duplicates=True)
    return mainFile #modified in-place

def cleanOnsets(onsets):
    #add column headers
    onsets.columns = ["TrialNum", "Condition", "TrialNum_perCondi",
    "ImageID", "Trial_part", "onsetSec", "durationSec"]
    #remove first six junk rows (3 junk trials; 2 rows per trial)
    onsets.drop([0, 1, 2, 3, 4, 5], axis=0, inplace=True)
    return onsets

def cleanRetriev(ret):
    #Change column headers
    ret.rename(columns={'category':'old_new', 'Stim':'stim_file',
    'OldNumber':'stim_id', 'Recognition_ACC':'recognition_accuracy',
    'Recognition_RESP':'recognition_response', 'Recognition_RT':'recognition_responsetime',
    'Spatial_RESP':'position_response', 'Spatial_RT':'position_responsetime',
    'Spatial_ACC(à corriger voir output-encodage)':'position_accuracy'}, inplace=True)
    #re-order columns
    cols = ['old_new', 'stim_file', 'stim_id', 'recognition_response',
    'recognition_accuracy', 'recognition_responsetime', 'position_response',
    'position_accuracy', 'position_responsetime']
    ret = ret[cols]
    #Transform reaction time columns from ms to s
    ret[['recognition_responsetime']]=ret[['recognition_responsetime']].astype('float64', copy=False) #string is object in pandas, str in Python
    ret[['position_responsetime']]=ret[['position_responsetime']].astype('float64', copy=False)
    ret['recognition_responsetime'] = ret['recognition_responsetime'].div(1000)
    ret['position_responsetime'] = ret['position_responsetime'].div(1000)
    #Clean up eprime mistake: replace position_response and position_responsetime
    #to NaN if subject perceived image as new (not probed for position)
    #There should not be a response or an RT value there, it is carried over from previous trial (not reset)
    #VERIFY: cannot flag trials were person answered OLD but failed to give position answer when probed
    i = ret[ret['recognition_response']==2].index
    ret.loc[i, 'position_responsetime']= NaN
    ret.loc[i, 'position_response'] = -1
    #clean up eprime mistake (change Old67 condition ('old_new') from New to OLD)
    q = ret[ret['stim_id']=='Old67'].index
    ret.loc[q, 'old_new'] = 'OLD'
    #insert new columns
    colNames = ['trial_number', 'stim_category', 'stim_name',
    'recognition_performance', 'position_correct']
    dtype = [-1, 'None', 'None', 'None', -1]
    colIndex = [0, 4, 5, 9, 10]
    for j in range (0, 5):
        ret.insert(loc=colIndex[j], column=colNames[j], value=dtype[j], allow_duplicates=True)#FIX efficienty: add new columns; change order?
    #Extract info and fill trial_number, stim_category and stim_name columns
    k = ret.index
    ret.loc[k, 'trial_number'] = k+1
    #format: category_imageName.bmp w some space, _ and - in image names
    stimInfo = ret.loc[k, 'stim_file']
    for s in k:
        ret.loc[s, 'stim_category'] = re.findall('(.+?)_', stimInfo[s])[0]
        ret.loc[s, 'stim_name'] = re.findall('_(.+?)[.]', stimInfo[s])[0]
    #Fill recognition_performance column based on actual and perceived novelty
    m = ret[ret['old_new']=='OLD'].index.intersection(ret[ret['recognition_accuracy']==1].index)
    ret.loc[m, 'recognition_performance']='Hit'
    n = ret[ret['old_new']=='OLD'].index.intersection(ret[ret['recognition_accuracy']==0].index)
    ret.loc[n, 'recognition_performance']='Miss'
    o = ret[ret['old_new']=='New'].index.intersection(ret[ret['recognition_accuracy']==1].index)
    ret.loc[o, 'recognition_performance']='CR'
    p = ret[ret['old_new']=='New'].index.intersection(ret[ret['recognition_accuracy']==0].index)
    ret.loc[p, 'recognition_performance']='FA'
    #return cleaned up input Dataframe
    return ret

def addOnsets(main, enc):
    #make main file indexable by trial number:
    main.set_index('trial_number', inplace = True)
    #copy trial onset and offset times from enc into main
    #note: fixation's onset time is the trial task's offset time
    for i in enc.index:
        trialNum = enc.loc[i, 'TrialNum']
        if enc.loc[i, 'Trial_part']=='Fixation':
            main.loc[trialNum, 'offset'] = enc.loc[i, 'onsetSec']
        else :
            main.loc[trialNum, 'onset'] = enc.loc[i, 'onsetSec']
    #Calculate trial duration time from onset and offset times
    main['duration'] = main['offset']-main['onset']
    #reset main's searchable index to default
    main.reset_index(level=None, drop=False, inplace=True)
    return main

def addPostScan(main, ret):
    #split main's rows (trials) into sublist based on Condition
    mainEnc = main[main['trial_type']=='Enc'].copy()
    mainCTL = main[main['trial_type']=='CTL'].copy()
    #make mainEnc indexable by picture id
    mainEnc.set_index('stim_id', inplace = True)
    #import post-scan data from ret into mainEnc
    for i in ret[ret['old_new']=='OLD'].index:
        stimID = ret.loc[i, 'stim_id']
        mainEnc.loc[stimID, 'stim_category'] = ret.loc[i, 'stim_category']
        mainEnc.loc[stimID, 'stim_name'] = ret.loc[i, 'stim_name']
        mainEnc.loc[stimID, 'recognition_accuracy'] = ret.loc[i, 'recognition_accuracy']
        mainEnc.loc[stimID, 'recognition_responsetime'] = ret.loc[i, 'recognition_responsetime']
        mainEnc.loc[stimID, 'position_response'] = ret.loc[i, 'position_response']
        mainEnc.loc[stimID, 'position_responsetime'] = ret.loc[i, 'position_responsetime']
    #calculate post-scan source (position) accuracy;
    # -1 = control task; 0 = missed trial; 1 = wrong source (image recognized but wrong quadrant remembered);
    #2 = image recognized with correct source
    mainEnc['position_accuracy'] = 0
    for j in mainEnc[mainEnc['recognition_accuracy']==1].index:
        if mainEnc.loc[j, 'position_correct'] == mainEnc.loc[j, 'position_response']:
            mainEnc.loc[j, 'position_accuracy'] = 2
        else:
            mainEnc.loc[j, 'position_accuracy'] = 1
    #import source accuracy info from mainEnc into ret (in-place)
    for i in ret[ret['old_new']=='OLD'].index:
        picID = ret.loc[i, 'stim_id']
        ret.loc[i, 'position_correct'] = mainEnc.loc[picID, 'position_correct']
        ret.loc[i, 'position_accuracy'] = mainEnc.loc[picID, 'position_accuracy']
    #reset mainEnc searchable index to default
    #and re-order columns to match order in mainCTL
    mainEnc.reset_index(level=None, drop=False, inplace=True)
    cols = ['trial_number', 'onset', 'duration', 'offset', 'trial_type', 'response',
       'response_time', 'stim_id', 'stim_file', 'stim_category',
       'stim_name', 'recognition_accuracy', 'recognition_responsetime',
       'position_correct', 'position_response', 'position_accuracy', 'position_responsetime']
    mainEnc = mainEnc[cols]
    #Re-merge mainEnc and mainCTL and re-order by trial number
    mainMerged = mainEnc.append(mainCTL, ignore_index=True)
    mainMerged.sort_values('trial_number', axis=0, ascending=True, inplace=True)
    return mainMerged

def extract_taskFile(bID, sID, file_list, output):
    #import data from three text files into pandas DataFrames
    encMain = pd.read_csv(file_list[0], sep='\t')
    encOnsets = pd.read_fwf(file_list[1], infer_nrows=210,
    delim_whitespace=True, header=None)
    retriev = pd.read_csv(file_list[2], sep='\t', encoding = 'ISO-8859-1')
    #clean up each file
    encMain = cleanMain(encMain)
    encOnsets = cleanOnsets(encOnsets)
    retriev = cleanRetriev(retriev)
    #import onset times from encOnset into encMain
    encMain = addOnsets(encMain, encOnsets)
    #import post-scan performance data from retriev into encMain
    encMain = addPostScan(encMain, retriev)
    #export encMain and retriev into tsv files (output directorty)
    #encMain.to_csv(output+'/TaskFile_bID'+bID+'_mriID'+sID+'.tsv',
    #sep='\t', header=True, index=False)
    encMain.to_csv(output+'/sub-'+sID+'_ses-4_task-memory_events.tsv',
    sep='\t', header=True, index=False)
    retriev.to_csv(output+'/PostScanBehav_bID'+bID+'_mriID'+sID+'.tsv',
    sep='\t', header=True, index=False)
    #sub-658178_ses-4_task-memory_events.tsv

def main():
    args =  get_arguments()
    all_ids = get_all_ids(args.idir[0])
    temp_dir = os.path.join(args.odir[0], 'Temp')
    file_dir = os.path.join(args.odir[0], 'TaskFiles')
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir) #where unzip subject file, remove when done
    if not os.path.exists(file_dir):
        os.mkdir(file_dir) #where task files are saved
    for (idBehav, idMRI) in all_ids:
        s_files = set_subject_data(idBehav, args.idir[0], temp_dir)
        if(len(s_files)==3):
            extract_taskFile(idBehav, idMRI, s_files, file_dir)
        else:
            print('missing files for subject '+idBehav+', '+idMRI)
    #os.rmdir(temp_dir) gives error, remove temp directory manually

if __name__ == '__main__':
    sys.exit(main())
