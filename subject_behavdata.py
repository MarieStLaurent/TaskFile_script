
###Pending Qs for Samira: what purpose of first 3 trials? (signal equilibrium?)
### What is the control task (what looks like?)
###How are pictures labelled? Is the same image always old, or is
###the oldness or newness of the image randomized between participants?
###Can I have a directory with the stimulus images in it?
##How are images labelled? (old - new, number always matches same pic?)
##How to interpret response and response time to source question if post-scan pic identified as new?
##(do they just press whatever?)

#####################################

#Analysis steps:
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
    sub_files = []
    s_dir = glob.glob(datadir+'/'+idBeh+'*.zip')
    if len(s_dir) != 1:
        print('Multiple directories match subject id '+idBeh)
    else:
        z_ref = zipfile.ZipFile(s_dir[0], 'r')
        z_ref.extractall(output)
        z_ref.close()
        s_main = glob.glob(output+'/'+idBeh+'*/Output-Responses-Encoding_CIMAQ_*')
        s_onsets = glob.glob(output+'/'+idBeh+'*/Onset-Event-Encoding_CIMAQ_*')
        s_postscan = glob.glob(output+'/'+idBeh+'*/Output_Retrieval_CIMAQ_*')
        if(len(s_main)==len(s_onsets)==len(s_postscan)==1):
            sub_files.append(s_main[0])
            sub_files.append(s_onsets[0])
            sub_files.append(s_postscan[0])
    return sub_files

def cleanMain(mainFile):
    #remove first three junk rows (blank trials): CTL0, Enc00 and ENc000
    mainFile.drop([0, 1, 2], axis=0, inplace=True)
    #re-label columns
    mainFile.rename(columns={'Category':'Condition', 'OldNumber':'StimulusID',
    'Stim_RESP':'InScan_Resp', 'Stim_RT':'InScan_RT'}, inplace=True)
    #change in-scan reaction time from ms to s
    mainFile[['InScan_RT']]=mainFile[['InScan_RT']].astype('float64', copy=False)
    mainFile['InScan_RT'] = mainFile['InScan_RT'].div(1000)
    #remove redundant column (same info as column InScan_Resp)
    mainFile.drop('Stim_ACC', axis=1, inplace=True)

    ##TO FIX: make this section more efficient!
    ##Add new columns; Change column order?
    #For onsetTime and offsetTime columns, could instead copy entire columns from onset subfiles...
    mainFile.insert(loc=mainFile.shape[1], column='onsetTime', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=len(mainFile.columns), column='offsetTime', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=len(mainFile.columns), column='Duration', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='StimCategory', value='None', allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='StimulusImage', value='None', allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='PostScanReco_ACC', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='PostScanReco_RT', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceResp', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceRT', value=NaN, allow_duplicates=True)
    mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceACC', value=NaN, allow_duplicates=True)
    #cast float numerical columns as float64 type (pandas' float)
    #Overkill? (NaN type infered as float64 by pandas);
    mainFile[['onsetTime', 'offsetTime', 'Duration']]=mainFile[['onsetTime', 'offsetTime', 'Duration']].astype('float64', copy=False)
    mainFile[['PostScanReco_RT', 'PostScan_SourceRT']]=mainFile[['PostScanReco_RT', 'PostScan_SourceRT']].astype('float64', copy=False)
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
    ret.rename(columns={'category':'OLD_NEW', 'Stim':'Stimulus',
    'OldNumber':'StimulusID', 'Recognition_Resp':'Recognition_Resp_1old_2new',
    'Spatial_RESP':'SpatialSource_Resp', 'Spatial_RT':'SpatialSource_RT',
    'Spatial_ACC(à corriger voir output-encodage)':'SpatialSource_ACC'}, inplace=True)
    #Transform reaction time columns from ms to s
    ret[['Recognition_RT']]=ret[['Recognition_RT']].astype('float64', copy=False) #string is object in pandas, str in Python
    ret[['SpatialSource_RT']]=ret[['SpatialSource_RT']].astype('float64', copy=False)
    ret['Recognition_RT'] = ret['Recognition_RT'].div(1000)
    ret['SpatialSource_RT'] = ret['SpatialSource_RT'].div(1000)
    #FIX efficienty: add new columns; change order?
    ret.insert(loc=ret.shape[1], column='TrialNum', value=NaN, allow_duplicates=True) #type = int64 -> int
    ret.insert(loc=ret.shape[1], column='SpatialCorrectSource', value=NaN, allow_duplicates=True) #type = int64 -> int
    ret.insert(loc=ret.shape[1], column='StimCategory', value='None', allow_duplicates=True) #type = object -> string
    ret.insert(loc=ret.shape[1], column='StimulusImage', value='None', allow_duplicates=True) #type = object -> string
    ret.insert(loc=ret.shape[1], column='TrialPerfoType', value='None', allow_duplicates=True) #type = int64 -> int
    #Extract info and fill TrialNum, StimCategory and StimulusImage columns
    for i in ret.index:
        ret.loc[i, 'TrialNum'] = i+1
        #format: category_imageName.bmp w some space, _ and - in image names
        stimInfo = ret.loc[i, 'Stimulus']
        categ = re.findall('(.+?)_', stimInfo)[0]
        ima = re.findall('_(.+?)[.]', stimInfo)[0]
        ret.loc[i, 'StimCategory'] = categ
        ret.loc[i, 'StimulusImage'] = ima
    #Fix efficiency: Fill TrialPerfoType column based on old/new and trial type
    for i in ret.index:
        if ret.loc[i, 'OLD_NEW'] == 'OLD':
            if ret.loc[i, 'Recognition_ACC'] == 1:
                ret.loc[i, 'TrialPerfoType']='Hit'
            else:
                ret.loc[i, 'TrialPerfoType']='Miss'
        else:
            if ret.loc[i, 'Recognition_ACC'] == 1:
                ret.loc[i, 'TrialPerfoType']='CR'
            else:
                ret.loc[i, 'TrialPerfoType']='FA'
    return ret

def addOnsets(main, enc):
    #make main file indexable by trial number:
    main.set_index('TrialNumber', inplace = True)
    #copy trial onset and offset times from enc into main
    #note: fixation's onset time is the trial task's offset time
    for i in enc.index:
        trialNum = enc.loc[i, 'TrialNum']
        if(enc.loc[i, 'Trial_part']=='Fixation'):
            main.loc[trialNum, 'offsetTime'] = enc.loc[i, 'onsetSec']
        else:
            main.loc[trialNum, 'onsetTime'] = enc.loc[i, 'onsetSec']
    #Calculate trial duration time from onset and offset times
    main['Duration'] = main['offsetTime']-main['onsetTime']
    #reset main's searchable index to default
    main.reset_index(level=None, drop=False, inplace=True)
    return main

def addPostScan(main, ret):
    #split main's rows (trials) into sublist based on Condition
    mainEnc = main[main['Condition']=='Enc']
    mainCTL = main[main['Condition']=='CTL']
    #make mainEnc indexable by picture id
    mainEnc.set_index('StimulusID', inplace = True)
    #import post-scan data from ret into mainEnc
    for i in ret.index:
        if(ret.loc[i, 'OLD_NEW']=='OLD'):
            stimID = ret.loc[i, 'StimulusID']
            mainEnc.loc[stimID, 'StimCategory'] = ret.loc[i, 'StimCategory']
            mainEnc.loc[stimID, 'StimulusImage'] = ret.loc[i, 'StimulusImage']
            mainEnc.loc[stimID, 'PostScanReco_ACC'] = ret.loc[i, 'Recognition_ACC']
            mainEnc.loc[stimID, 'PostScanReco_RT'] = ret.loc[i, 'Recognition_RT']
            mainEnc.loc[stimID, 'PostScan_SourceResp'] = ret.loc[i, 'SpatialSource_Resp']
            mainEnc.loc[stimID, 'PostScan_SourceRT'] = ret.loc[i, 'SpatialSource_RT']
    #calculate post-scan source accuracy
    for i in mainEnc.index:
        if mainEnc.loc[i, 'CorrectSource'] == mainEnc.loc[i, 'PostScan_SourceResp']:
            mainEnc.loc[i, 'PostScan_SourceACC'] = 1
        else:
            mainEnc.loc[i, 'PostScan_SourceACC'] = 0
    #import source accuracy info from mainEnc into ret (in-place); global variable?
    for i in ret.index:
        if(ret.loc[i, 'OLD_NEW']=='OLD'):
            picID = ret.loc[i, 'StimulusID']
            ret.loc[i, 'SpatialCorrectSource'] = mainEnc.loc[picID, 'CorrectSource']
            ret.loc[i, 'SpatialSource_ACC'] = mainEnc.loc[picID, 'PostScan_SourceACC']
    #reset mainEnc searchable index to default
    #and re-order columns to match order in mainCTL
    mainEnc.reset_index(level=None, drop=False, inplace=True)
    cols = ['TrialNumber', 'Condition', 'TrialCode', 'StimulusID',
     'CorrectSource', 'InScan_Resp', 'InScan_RT', 'onsetTime', 'offsetTime', 'Duration',
     'StimCategory', 'StimulusImage', 'PostScanReco_ACC', 'PostScanReco_RT',
     'PostScan_SourceResp', 'PostScan_SourceRT', 'PostScan_SourceACC']
    mainEnc = mainEnc[cols]
    #Re-merge mainEnc and mainCTL and re-order by trial number
    mainMerged = mainEnc.append(mainCTL, ignore_index=True)
    mainMerged.sort_values('TrialNumber', axis=0, ascending=True, inplace=True)
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
    encMain.to_csv(output+'/TaskFile_bID'+bID+'_mriID'+sID+'.tsv',
    sep='\t', header=True, index=False)
    retriev.to_csv(output+'/PostScanBehav_bID'+bID+'_mriID'+sID+'.tsv',
    sep='\t', header=True, index=False)

def main():
    """Let's go"""
    args =  get_arguments()
    all_ids = get_all_ids(args.idir[0])
    temp_dir = args.odir[0]+'/Temp'
    file_dir = args.odir[0]+'/TaskFiles'
    os.mkdir(temp_dir) #where unzip subject file, remove when done
    os.mkdir(file_dir) #where task files are saved
    for (idBehav, idMRI) in all_ids:
        s_files = set_subject_data(idBehav, args.idir[0], temp_dir)
        if(len(s_files)==3):
            extract_taskFile(idBehav, idMRI, s_files, file_dir)
        else:
            print('missing files for subject '+idBehav)

        print(idBehav,idMRI)
    os.rmdir(temp_dir)

if __name__ == '__main__':
    sys.exit(main())
