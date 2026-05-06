import re,os,subprocess,shutil
from tqdm import tqdm 
import pandas as pd 

# NUPOS: https://morphadorner.northwestern.edu/documentation/nupos/
def adorn(tcpID): 
    folder = f"/Users/amycweng/DH"
    output_folder = f"/Users/amycweng/My Drive (aw3029@princeton.edu)/DH/POETRY"
    # shutil.copy(f"{output_folder}/APS_PLAIN/{tcpID}.txt",f"{folder}/APS_PLAIN/{tcpID}.txt")
    shutil.copy(f"{output_folder}/APS_PLAIN/{tcpID}.txt",f"{folder}/{tcpID}.txt")
    os.chdir(f'{folder}/morphadorner-2')
    subprocess.run(['./adornplainemetext', f"{folder}", f"{folder}/{tcpID}.txt"])
    # subprocess.run(['./adornplainemetext', f"{folder}/APS_ADORNED", f"{folder}/APS_PLAIN/{tcpID}.txt"])

adorn("DDC_FINAL")
# adorn("A88989_1")
# adorn("A88989_2")
# adorn("A68202_1")
# adorn("A68202_2")

# 100%|████████████████████████████████████████████████| 11456/11456 [37:44:45<00:00, 11.86s/it]
# tcpIDs = pd.read_csv(f"APS_CORPUS_TO_CLUSTER.csv")['id']
# tcpIDs = sorted(tcpIDs)
# for tcpID in tqdm(tcpIDs): 
#     shutil.copy(f"{output_folder}/APS_PLAIN/{tcpID}.txt",f"{folder}/APS_PLAIN/{tcpID}.txt")
#     adorn(tcpID)