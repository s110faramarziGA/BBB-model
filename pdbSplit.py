# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:14:15 2023

@author: sadegh-pc
"""

#% split concatenated pdb files to separate ones


#%% load libraries
import csv
import logging 
import os
import traceback 
import pandas
from tqdm import tqdm
from biopandas.pdb import PandasPdb
from Bio.PDB import *
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit
import pandas as pd
#%% read concatenated pdb

ppdb = PandasPdb().fetch_pdb('3eiy')
    #load file from the local desktop
fpath='E:\Professional\TheProject\Codes\Structures\pdb\\'

# filename=('E:\Professional\TheProject\Codes\Structures\pdb\\'+str(xx)+''.pdb')
# filename=fpath+str(1)+'.pdb'
filename=fpath+'TrainingSet'+'.pdb'

f = open(filename)
xx= f.readlines() 
print(xx)

#%%
# https://stackoverflow.com/questions/35916503/how-can-i-split-a-text-file-into-multiple-text-files-using-python
import re



buff = []
i = 1
for line in f:
    # print(line)
    if line.strip():  #skips the empty lines
       buff.append(line)
    if line.strip() == "END":
       print(i)
       output = open('%d.pdb' % i,'w')
       output.write(''.join(buff))
       output.close()
       i+=1
           
#%%
with open(filename, 'r') as f:
    data = f.read()
found = re.findall(r'\n*(HEADER.*?END)\n*', data, re.M | re.S)
#%%
# [open(str(i)+'.txt', 'w').write(found[i-1]) for i in range(1, len(found)+1)]
[open(str(i)+'.pdb', 'w').write(found[i-1]) for i in range(1, len(found)+1)]
