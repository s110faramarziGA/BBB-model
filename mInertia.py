# -*- coding: utf-8 -*-
"""
calculate moments of iertia and principal axes of a molecule. Formulas can be found here
https://pycrawfordprogproj.readthedocs.io/en/latest/Project_01/Project_01.html
https://physics.stackexchange.com/questions/219076/how-can-i-determine-the-vector-parallel-to-the-long-molecular-axis
https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/

@author: s110f
"""

import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
# from scipy.integrate import dblquad
# from scipy.integrate import tblquad
import pandas as pd

from tqdm import tqdm
from biopandas.pdb import PandasPdb
from Bio.PDB import *
import sys
#%% loading the pdb and get coordinate. The same of the fp code. This section will be deleted later
ppdb = PandasPdb().fetch_pdb('3eiy')
#load file from the local desktop
fpath='D:\Professional\TheProject\Codes\Structures\pdb\\'

filename=fpath+str(558)+'.pdb'
p_1=ppdb.read_pdb(filename)
print('PDB Code: %s' % ppdb.code)
print('PDB Header Line: %s' % ppdb.header)
print('\nRaw PDB file contents:\n\n%s\n...' % ppdb.pdb_text[:1000])
# ATOM or HETATM names should correspond to the first column of the pdb file
pdbdf=ppdb.df['HETATM']
if len(pdbdf)==0:
    pdbdf=ppdb.df['ATOM']
#[ppdb.df['HETATM']['element_symbol'] != 'OH'].head()
pdbdf.shape
#%% list of atomic masses
MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                  'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                  'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                  'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                  'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                  'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                  'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                  'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                  'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                  'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                  'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                  'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                  'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                  '': 0}

#%%
#calculate moments of iertia and principal axes of a molecule. for more information look at mInertia.py
# calculate the center of mass (COM)
# x coordinate of the COM
xcm=0
ycm=0
zcm=0
#some of atomic masses
msum=0
for i in range(len(pdbdf['element_symbol'])):
    xcm=xcm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]))
    ycm=ycm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['y_coord'][i]))
    zcm=zcm+((MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['z_coord'][i]))
    # print(xcm)
    msum=msum+MM_of_Elements[pdbdf['element_symbol'][i]]
    # print(msum)
xcm=xcm/msum
ycm=ycm/msum
zcm=zcm/msum
# principal moments of inertia, Is
ixx=0
iyy=0
izz=0
ixy=0
ixz=0
iyz=0
for i in range(len(pdbdf['element_symbol'])):
    ixx=ixx+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['y_coord'][i]-ycm)**2+(pdbdf['z_coord'][i]-zcm)**2)
    iyy=iyy+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['z_coord'][i]-zcm)**2+(pdbdf['x_coord'][i]-xcm)**2)
    izz=izz+(MM_of_Elements[pdbdf['element_symbol'][i]])*((pdbdf['x_coord'][i]-xcm)**2+(pdbdf['y_coord'][i]-ycm)**2)
    ixy=ixy+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]-xcm)*(pdbdf['y_coord'][i]-ycm)
    ixz=ixz+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['x_coord'][i]-xcm)*(pdbdf['z_coord'][i]-zcm)
    iyz=iyz+(MM_of_Elements[pdbdf['element_symbol'][i]])*(pdbdf['y_coord'][i]-ycm)*(pdbdf['z_coord'][i]-zcm)
    
ixy=-ixy
ixz=-ixz
iyz=-iyz

I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

Ip = np.linalg.eigvals(I)
# Sort and convert principal moments of inertia to SI (kg.m2)
Ip.sort()
# print(Ip)  