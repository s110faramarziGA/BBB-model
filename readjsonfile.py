# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:33:28 2023

@author: sadegh-pc
"""

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


#%%
import rdkit.Chem.rdMolInterchange
#%%
import json
f=open('E:\Professional\TheProject\Codes\Structures\Conformer3D_CID_3001055.json')
data=json.load(f)
print(data['PC_Compounds'])