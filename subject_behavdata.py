
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
import argparse
import glob
import numpy as np
import pandas as pd
import os
import re
import sys
import zipfile

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
        help="Verbose to    get more information about whats going on",
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

def main():
    """Let's go"""
    args =  get_arguments()
    all_ids = get_all_ids(args.idir[0])
    for (idBehav, idMRI) in all_ids:
        print(idBehav,idMRI)

if __name__ == '__main__':
    sys.exit(main())
