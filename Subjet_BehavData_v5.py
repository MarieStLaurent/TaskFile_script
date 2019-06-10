import numpy as np
import pandas as pd
import os, re
import glob

#################
#Questions, draft code and notes
#load single subject's 3 datafiles into pandas DataFrames

#unzip subject's directory, access files within directory
#File1 (encOnsets): Onset-Event-Encoding_CIMAQ_id_session.txt
#File2 (encBehav): Output-Responses-Encoding_CIMAQ_id_session.txt
#File3 (retriev): Output_Retrieval_CIMAQ_id_session.txt

#I cannot manage to use the wild card *...
#IMPORTANT: behav ID is not the same as scanning id
bID = 3025432
sID = 658178
encOnsets = pd.read_csv('Onset-Event-Encoding_CIMAQ_'+str(bID)+'*.txt', sep='\s+', header=None)
encOnsets.columns = ["index", "Condi", "Trial", "Image", "Trial_seg", "onsetSec", "unknown"]
mainFile = pd.read_csv('Output-Responses-Encoding_CIMAQ_'+bID+'*.txt', sep='\t')
retriev = pd.read_csv('Output_Retrieval_CIMAQ_'+bID+'*.txt', sep='\t')

#useful pandas basics:
mainFile.dtypes #returns type of data inferred per column; make explicit?
mainFile.head()
mainFile.tail()
mainFile.iloc[0:10, 0:10]
mainFile.loc[:, 'TrialNumber']
#to explicitely convert dtypes from one to another; return copy unless copy=False
#convert subset of columns to dtype
mainFile[['Category', 'OldNumber']]=mainFile[['Category', 'OldNumber']].astype(object) #string is object in pandas, str in Python
#other types: bool (pandas and python), int64 (pandas) vs int (python), float64 (pandas) vs float (python)
#string is object in pandas, str in Python

###Qs for Samira: what purpose of first 3 trials? (signal equilibrium?)
### What is the control task (what looks like?)
###How are pictures labelled? Is the same image always old, or is
###the oldness or newness of the image randomized between participants?
###Can I have a directory with the stimulus images in it?
##How are images labelled? (old - new, number always matches same pic?)
##How to interpret response and response time to source question if post-scan pic identified as new?
##(do they just press whatever?)

#####################################

#Analysis steps:

#load main file: Output-Responses-Encoding_CIMAQ*txt, modify it in-place
mainFile = pd.read_csv('Output-Responses-Encoding_CIMAQ_'+bID+'*.txt', sep='\t')

#Modify main file in-place:
#remove first three junk rows (blank trials): CTL0, Enc00 and ENc000
mainFile.drop([0, 1, 2], axis=0, inplace=True)
#alternative way: mainFile = mainFile.iloc[3:,]

#Change column header names:
#Column1: 'Category', change to "Condition"
#Col3, OldNumber: change to "StimulusID"
#Col6, Stim_RESP: change to "InScan_Resp";
#Col7, Stim_RT: change to "InScan_RT", convert to float and change value to s (rather than ms)
mainFile.rename(columns={'Category':'Condition', 'OldNumber':'StimulusID',
'Stim_RESP':'InScan_Resp', 'Stim_RT':'InScan_RT'}, inplace=True)

#cast RT data as float64, then divide by 1000 (from ms to s)
mainFile[['InScan_RT']]=mainFile[['InScan_RT']].astype('float64', copy=False)
mainFile['InScan_RT'] = mainFile['InScan_RT'].div(1000)

#Remove Col8: 'Stim_ACC' (redundant)
mainFile.drop('Stim_ACC', axis=1, inplace=True) #axis 0 = rows, 1 = columns

#Load other enconding file with onset times (encOnsets): Onset-Event-Encoding_CIMAQ_*.txt
##Data is Fixed Width, padding with blank spaces
#Column format is estimated from first 100 rows by default; reset to avoid truncated values
encOnsets = pd.read_fwf('Onset-Event-Encoding_CIMAQ_'+str(bID)+'*.txt', infer_nrows=210, delim_whitespace=True, header=None)
#add column headers
encOnsets.columns = ["TrialNum", "Condition", "TrialNum_perCondi", "ImageID", "Trial_part", "onsetSec", "durationSec"]
#remove first six junk rows (3 junk trials)
encOnsets.drop([0, 1, 2, 3, 4, 5], axis=0, inplace=True)

#Create two sublists:
#one where Col 4 = Fixation, the other where it isn't (Col4 = Control or Encoding)
#creates copies distinct from encOnset
encFix = encOnsets[encOnsets['Trial_part']=='Fixation'] #displays sublist
encNoFix = encOnsets[encOnsets['Trial_part']!='Fixation']

#alternative 2-step method to split data (not used here)
#isFix = encOnsets.iloc[:, 4]=='Fixation' ##create boolean
#fixEnc = encOnsets[isFix]
#notFix = encOnsets.iloc[:, 4]!='Fixation' ##create boolean
#nfEnc = encOnsets[notFix]

#in MAIN FILE: create two new columns, "onsetTime" and "offsetTime"
#set default value to NaN??? (proper default value?)
mainFile.insert(loc=mainFile.shape[1], column='onsetTime', value=-1.0, allow_duplicates=True)
mainFile.insert(loc=len(mainFile.columns), column='offsetTime', value=-1.0, allow_duplicates=True)

#make sure data type is float from onset and offset times in mainFile before copying; if not, cast type (float)
mainFile[['onsetTime', 'offsetTime']]=mainFile[['onsetTime', 'offsetTime']].astype('float64', copy=False) #string is object in pandas, str in Python

#obtain timing values (trial onset and offset times) from encFix and encNoFix
#In mainFile, the TrialNumber (col0) corresponds to TrialNum (col0) in encFix and encNoFix.
#col5 = trial onset time value in encNoFix (col4 != Fixation , == Control or Encoding);
#col5= trial offset time value in encFix (col4 == Fixation)

#make mainFile indexable by Trial Number:
mainFile.set_index('TrialNumber', inplace = True)

##veridy with dtypes that the columns' dtype (copied from and to)= float64
mainFile.dtypes
encFix.dtypes
encNoFix.dtypes

#Better way: copy entire column of matching dimention (since trial order the same)
#import new column into mainFile
#and give it name (one step, no iterations)

#copy onset times from encNoFIx into mainFile:
for i in encNoFix.index:
    trialNum = encNoFix.loc[i, 'TrialNum']
    mainFile.loc[trialNum, 'onsetTime'] = encNoFix.loc[i, 'onsetSec']

#copy offset times from encFIx into mainFile:
for i in encFix.index:
    trialNum = encFix.loc[i, 'TrialNum']
    mainFile.loc[trialNum, 'offsetTime'] = encFix.loc[i, 'onsetSec']

#Add duration column, calculate its value by subtraction
#Note: each image is shown for 3 seconds each;
#the fixation time (inter-stim inverval) is jittered
mainFile.insert(loc=len(mainFile.columns), column='Duration', value=-1.0, allow_duplicates=True)
mainFile['Duration'] = mainFile['offsetTime']-mainFile['onsetTime']

#other manner to copy onset/offset values into mainFile, slower
#for index, row in encNoFix.iterrows():
#    mainFile.loc[row['TrialNum'], 'onsetTime'] = row['onsetSec']

#copying offsetTimes from enc sublist into main:
#for index, row in encFix.iterrows():
#    mainFile.loc[row['TrialNum'], 'offsetTime'] = row['onsetSec']


#In MAIN File, create these additional colums:
#StimCategory (what category the image belonged to)
#StimulusImage (what was the image)
#PostScanReco_ACC (was the image accurately recognized post-scan: boolean)
#PostScanReco_RT (reaction time)
#PostScan_SourceResp (what source was remembered by subject, if pic was recognized postscan)
#PostScan_SourceRT
#PostScan_SourceACC (to be calculated by cross-referencing encoding and retrieval output files)
mainFile.insert(loc=mainFile.shape[1], column='StimCategory', value='tbd', allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='StimulusImage', value='tbd', allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='PostScanReco_ACC', value=-1, allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='PostScanReco_RT', value=-1.0, allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceResp', value=-1, allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceRT', value=-1.0, allow_duplicates=True)
mainFile.insert(loc=mainFile.shape[1], column='PostScan_SourceACC', value=-1, allow_duplicates=True)

#Reset index of main file to default (not TrialNumber)
mainFile.reset_index(level=None, drop=False, inplace=True)

#Create two sublists from the MAIN file where col3 "StimulusID" is not Nan,
#and one where it is NaN (split dataframe between encoding and control trials)
mainEnc = mainFile[mainFile['Condition']=='Enc'] #displays sublist
mainCTL = mainFile[mainFile['Condition']=='CTL']

#In mainEnc sublist (StimulusID != NaN), make col3 ('StimulusID')
#the index so the array is searchable by picture ID
mainEnc.set_index('StimulusID', inplace = True)

#Load 3rd file (retriev): Output_Retrieval_CIMAQ_id_session.txt
retriev = pd.read_csv('Output_Retrieval_CIMAQ_'+bID+'*.txt', sep='\t', encoding = 'ISO-8859-1')

#In retriev, change following column headers
#col0 'category' becomes 'OLD_NEW'
#col1 'Stim' becomes 'Stimulus'
#col2 'OldNumber' becomes 'StimulusID'
#col4 'Recognition_Resp' becomes 'Recognition_Resp_1old_2new'
#col5 'Spatial_Resp' becomes 'SpatialSource_Resp'
#col6 'Spatial_RT' becomes 'SpatialSource_RT'
#col7 'Spatial_ACC' becomes 'SpatialSource_ACC'
retriev.rename(columns={'category':'OLD_NEW', 'Stim':'Stimulus',
'OldNumber':'StimulusID', 'Recognition_Resp':'Recognition_Resp_1old_2new',
'Spatial_RESP':'SpatialSource_Resp', 'Spatial_RT':'SpatialSource_RT',
'Spatial_ACC(à corriger voir output-encodage)':'SpatialSource_ACC'}, inplace=True)

#Convert the two RT columns into float64 and divide by 1000 (convert values from ms to s)
retriev[['Recognition_RT']]=retriev[['Recognition_RT']].astype('float64', copy=False) #string is object in pandas, str in Python
retriev[['SpatialSource_RT']]=retriev[['SpatialSource_RT']].astype('float64', copy=False)
retriev['Recognition_RT'] = retriev['Recognition_RT'].div(1000)
retriev['SpatialSource_RT'] = retriev['SpatialSource_RT'].div(1000)

#Within retriev, create four new columns: "TrialNum", "SpatialCorrectSource", "StimCategory", and "StimulusImage"
#Consider re-ordering the columns
retriev.insert(loc=retriev.shape[1], column='TrialNum', value=-1, allow_duplicates=True) #type = int64 -> int
retriev.insert(loc=retriev.shape[1], column='SpatialCorrectSource', value=-1, allow_duplicates=True) #type = int64 -> int
retriev.insert(loc=retriev.shape[1], column='StimCategory', value='tbd', allow_duplicates=True) #type = object -> string
retriev.insert(loc=retriev.shape[1], column='StimulusImage', value='tbd', allow_duplicates=True) #type = object -> string

#split col1's string entry (header = Stimulus) into two substrings:
#1. the segment before '_' is the StimCategory (copy to that column within retriev )
#2. the segment between '_' and '.' is the StimulusImage (copy to that column within retriev )
for i in retriev.index:
    retriev.loc[i, 'TrialNum'] = i+1
    stimInfo = retriev.loc[i, 'Stimulus'] #string format: category_imageName.bmp w some space, _ and - in names
    categ = re.findall('(.+?)_', stimInfo)[0]
    ima = re.findall('_(.+?)[.]', stimInfo)[0]
    retriev.loc[i, 'StimCategory'] = categ
    retriev.loc[i, 'StimulusImage'] = ima

#Within retriev, create four new columns: "Hit", "CorrectRej", "Miss" and "FAlarm"
retriev.insert(loc=retriev.shape[1], column='Hit', value=0, allow_duplicates=True) #type = int64 -> int
retriev.insert(loc=retriev.shape[1], column='CorrectRej', value=0, allow_duplicates=True) #type = object -> string
retriev.insert(loc=retriev.shape[1], column='Miss', value=0, allow_duplicates=True) #type = object -> string
retriev.insert(loc=retriev.shape[1], column='FAlarm', value=0, allow_duplicates=True) #type = object -> string
#IF col0 = "OLD" and col3 (Recognition_ACC) = 1, Hit = 1, else 0
#IF col0 = "OLD" and col3 (Recognition_ACC) = 0, Miss = 1, else 0
#IF col0 = "NEW" and col3 (Recognition_ACC) = 1, CorrectRej = 1, else 0
#IF col0 = "NEW" and col3 (Recognition_ACC) = 0, FAlarm = 1, else 0
for i in retriev.index:
    if retriev.loc[i, 'OLD_NEW'] == 'OLD':
        if retriev.loc[i, 'Recognition_ACC'] == 1:
            retriev.loc[i, 'Hit']=1
        else:
            retriev.loc[i, 'Miss']=1
    else:
        if retriev.loc[i, 'Recognition_ACC'] == 1:
            retriev.loc[i, 'CorrectRej']=1
        else:
            retriev.loc[i, 'FAlarm']=1

#From retriev, create two sublists : one where col0 (header = category) = OLD,
#the other where col0 = NEW
retriOLD = retriev[retriev['OLD_NEW']=='OLD']
retriNEW = retriev[retriev['OLD_NEW']=='New']

##Iterate through retriOLD (where col0 = OLD):
#For rows with matching imageIDs between retriOLD(col2) and mainEnc trials (searchable index) :
#Copy the following info from retriOLD to MainEnc:
#from StimCategory to MAIN sublist's StimCategory
#from StimulusImage to MAIN sublist's StimulusImage
#from Recognition_ACC to MAIN sublist's PostScanReco_ACC
#from Recognition_RT to MAIN sublist's PostScanReco_RT
#from SpatialSource_Resp to PostScan_SourceResp (quadrants = 5, 6, 8, 9)
#From SpatialSource_RT to PostScan_SourceRT
for i in retriOLD.index:
    stimID = retriOLD.loc[i, 'StimulusID']
    mainEnc.loc[stimID, 'StimCategory'] = retriOLD.loc[i, 'StimCategory']
    mainEnc.loc[stimID, 'StimulusImage'] = retriOLD.loc[i, 'StimulusImage']
    mainEnc.loc[stimID, 'PostScanReco_ACC'] = retriOLD.loc[i, 'Recognition_ACC']
    mainEnc.loc[stimID, 'PostScanReco_RT'] = retriOLD.loc[i, 'Recognition_RT']
    mainEnc.loc[stimID, 'PostScan_SourceResp'] = retriOLD.loc[i, 'SpatialSource_Resp']
    mainEnc.loc[stimID, 'PostScan_SourceRT'] = retriOLD.loc[i, 'SpatialSource_RT']

#in mainENc: if col4 ("CorrectSource") == PostScan_SourceResp, PostScan_SourceACC = 1, else 0
for i in mainEnc.index:
    if mainEnc.loc[i, 'CorrectSource'] == mainEnc.loc[i, 'PostScan_SourceResp']:
        mainEnc.loc[i, 'PostScan_SourceACC'] = 1
    else:
        mainEnc.loc[i, 'PostScan_SourceACC'] = 0

#For rows with matching imageIDs between mainEnc and retriOLD, copy values from mainEnc to retriOLD:
#from mainEnc's "CorrectSource" to retriOLD's "SpatialCorrectSource": copy value
#from mainEnc's "PostScan_SourceACC" to retriOLD's "SpatialSource_ACC": copy value
for i in retriOLD.index:
    picID = retriOLD.loc[i, 'StimulusID']
    retriOLD.loc[i, 'SpatialCorrectSource'] = mainEnc.loc[picID, 'CorrectSource']
    retriOLD.loc[i, 'SpatialSource_ACC'] = mainEnc.loc[picID, 'PostScan_SourceACC']

#Re-merge the two retriev sublists (OLD trials and NEW trials) into a single doc.
retriev = retriOLD.append(retriNEW, ignore_index=True)

#sort merged dataframe by trial number
retriev.sort_values('TrialNum', axis=0, ascending=True, inplace=True)
##Post-scan behavioural scan is complete and usable: export as tsv w decent name
retriev.to_csv('PostScanBehFile.tsv', sep='\t', header=True, index=False)

#Re-merge the two MAIN sublists (mainEnc and mainCTL) into a single doc
#with Encoding and Control trials included
mainEnc.reset_index(level=None, drop=False, inplace=True)

#re-order columns to match order between two files
cols = ['TrialNumber',
 'Condition',
 'TrialCode',
 'StimulusID',
 'CorrectSource',
 'InScan_Resp',
 'InScan_RT',
 'onsetTime',
 'offsetTime',
 'Duration',
 'StimCategory',
 'StimulusImage',
 'PostScanReco_ACC',
 'PostScanReco_RT',
 'PostScan_SourceResp',
 'PostScan_SourceRT',
 'PostScan_SourceACC']

mainEnc =mainEnc[cols]

#merge and re-order by trial number
mainFile = mainEnc.append(mainCTL, ignore_index=True)
mainFile.sort_values('TrialNumber', axis=0, ascending=True, inplace=True)
#Main fMRI task file is complete, export as tsv w decent name

mainFile.to_csv('TaskFile.tsv', sep='\t', header=True, index=False)

##Q for Arnaud: BIDS standard?
#ordering columns better 
#cleaning up placeholder default values (NaN?)...
