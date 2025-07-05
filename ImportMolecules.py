# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:32:58 2022

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

#%%  adapted from http://rasbt.github.io/biopandas/tutorials/Working_with_PDB_Structures_in_DataFrames/
#fetch pdb. this command is to fetch PDB online but it will give error if not used for loading files from the local descktop 
ppdb = PandasPdb().fetch_pdb('3eiy')
#load file from the local desktop
p_1=ppdb.read_pdb('E:\Professional\TheProject\Codes\\aceticAcid.pdb')
print('PDB Code: %s' % ppdb.code)
print('PDB Header Line: %s' % ppdb.header)
print('\nRaw PDB file contents:\n\n%s\n...' % ppdb.pdb_text[:1000])
#%%
# reference_point = (9.362, 41.410, 10.542)
# distances = p_1.distance(xyz=('ATOM',), records=('ATOM',))

# distances.head()
#%% ATOM or HETATM names should correspond to the first column of the pdb file
pdbdf=ppdb.df['HETATM']
#[ppdb.df['HETATM']['element_symbol'] != 'OH'].head()
pdbdf.shape

#%%Functions
#diostance in 3d space
def dista(atom1, atom2):
    distancea=(((pdbdf['x_coord'][atom2]-pdbdf['x_coord'][atom1])**2)+((pdbdf['y_coord'][atom2]-pdbdf['y_coord'][atom1])**2)+((pdbdf['z_coord'][atom2]-pdbdf['z_coord'][atom1])**2))**0.5
    return distancea

# angle between 3 atoms. atom1 is at the center
# https://pycrawfordprogproj.readthedocs.io/en/latest/Project_01/Project_01.html, https://stackoverflow.com/questions/18945705/how-to-calculate-bond-angle-in-protein-db-file
def angle(atom1,atom2,atom3):
    
    a12=np.subtract([pdbdf['x_coord'][atom2],pdbdf['y_coord'][atom2],pdbdf['z_coord'][atom2]],[pdbdf['x_coord'][atom1],pdbdf['y_coord'][atom1],pdbdf['z_coord'][atom1]])
    a13=np.subtract([pdbdf['x_coord'][atom3],pdbdf['y_coord'][atom3],pdbdf['z_coord'][atom3]],[pdbdf['x_coord'][atom1],pdbdf['y_coord'][atom1],pdbdf['z_coord'][atom1]])
    norm12 = a12 / np.linalg.norm(a12)
    norm13 = a13 / np.linalg.norm(a13)
    dotprud=np.dot(norm12,norm13)
    angle=np.degrees(np.arccos(dotprud))
    return angle
#intersection (overlap) between two llists
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

#%%
for col in pdbdf.columns:
    print(col)
    
# xx=pdbdf['atom_name']
# print(xx)
#%%
#calculate molecular weight from smiles
# http://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html?highlight=pdb

# m = Chem.MolFromSmiles('c1ccccc1C(=O)O')
# mol_weight = Descriptors.MolWt(m)
# m=rdkit.Chem.rdmolfiles.MolFromPDBFile('./water.pdb')
# mol_weight = Descriptors.MolWt(m)

#%%

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

# a function to calculate molecular mass from the formula
def molecular_mass(compound: str, decimal_places=None) -> float:
    is_polyatomic = end = multiply = False
    polyatomic_mass, m_m, multiplier = 0, 0, 1
    element = ''

    for e in compound:
        if is_polyatomic:
            if end:
                is_polyatomic = False
                m_m += int(e) * polyatomic_mass if e.isdigit() else polyatomic_mass + MM_of_Elements[e]
            elif e.isdigit():
                multiplier = int(str(multiplier) + e) if multiply else int(e)
                multiply = True
            elif e.islower():
                element += e
            elif e.isupper():
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier, multiply = e, 1, False
            elif e == ')':
                polyatomic_mass += multiplier * MM_of_Elements[element]
                element, multiplier = '', 1
                end, multiply = True, False
        elif e == '(':
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier = '', 1
            is_polyatomic, multiply = True, False
        elif e.isdigit():
            multiplier = int(str(multiplier) + e) if multiply else int(e)
            multiply = True
        elif e.islower():
            element += e
        elif e.isupper():
            m_m += multiplier * MM_of_Elements[element]
            element, multiplier, multiply = e, 1, False
    m_m += multiplier * MM_of_Elements[element]
    if decimal_places is not None:
        return round(m_m, decimal_places)
    return m_m

#%%
# xx=molecular_mass('OH2')

#%% Molecular formulaa: make molecular formula from pdb file. Later the formula will be used to calculare MW.
# elsymb=pdbdf['element_symbol']

#count number of each element
countEl = pdbdf['element_symbol'].value_counts()
# print(countEl)
# count.index[0]
formula=''

for i in range(len(countEl)):
    # print(countEl[i])
    formula=formula+countEl.index[i]+str(countEl[i])
print(molecular_mass(formula))
#%% Use SMILES to Calculate MlogP and TPSA (topological polar surface area) for lipophilicity assessment. Look at https://www.rdkit.org/docs/index.html for more parameters
m = Chem.MolFromSmiles('c1ccccc1C(=O)O')
mlogp=Descriptors.MolLogP(m)
tpsa=Descriptors.TPSA(m)
#%% find a functional group or an structural scaffold in the PDB file based on the distances between the key atoms in the functional group
#return coordinates of atoms
# for col in pdbdf.columns:
#     print(col)
# xx=pdbdf['y_coord']
# print(xx)
# print(pdbdf['atom_name'])
#atom names
####### carboxylic acid
#he two carbon‑oxygen bonds in the delocalized carboxylate anion are identical (both 1.27 Å). However, in the structure of a carboxylic acid the  C−O bond (1.20 Å) is shorter than the  C−OH bond (1.34 Å). https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Organic_Chemistry_(Morsch_et_al.)/20%3A_Carboxylic_Acids_and_Nitriles/20.02%3A_Structure_and_Properties_of_Carboxylic_Acids

# carboxylic acid
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.45 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):
                    
                    if pdbdf['atom_name'][k]=='O' and dista(k,j)<1.45 and dista(k,j)>1.19 and k!=i:
                        # print(k,i)
                        for l in range(len(pdbdf['atom_name'])):
                            if (pdbdf['atom_name'][l]=='H' and dista(l,k)<1.1) or(pdbdf['atom_name'][l]=='H' and dista(l,i)<1.1):
                                ii=ii+1
                                print('carboxylic acid for atoms:',j+1,i+1,k+1)
print('number of carboxylic acid groups:',ii/2)
#%%
# carboxylate
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.45 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):
                    if pdbdf['atom_name'][k]=='O' and dista(k,j)<1.45 and dista(k,j)>1.19 and k!=i and dista(i,j)/dista(j,k) <1.1 and dista(i,j)/dista(j,k) >0.95 and dista(j,k)<1.42:
                        ii=ii+1
                        print('carboxylate found for atoms:',j+1,i+1,k+1)
                        
                        
                        
                                
print('number of carboxylate groups:',ii/2)
                                
#%%
#ester
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.35 and dista(i,j)>1.0 :
                for k in range(len(pdbdf['atom_name'])):
                    if pdbdf['atom_name'][k]=='O' and dista(k,j)<1.6 and dista(k,j)>1.1 and k!=i:
                        for l in range(len(pdbdf['atom_name'])):
                            if (pdbdf['atom_name'][l]=='C' and dista(l,k)<1.6 and l!=j) or(pdbdf['atom_name'][l]=='C' and dista(l,i)<1.6 and l!=j):
                                ii=ii+1
                                print('ester for atoms:',j+1,i+1,k+1)
print('number of ketone groups:', ii)
#%% alcohol
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.45 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,j)<1.6 and dista(k,j)>1.1:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='C' and dista(l,j)<1.6 and dista(l,j)>1.1 and l!=k:
                                for m in range(len(pdbdf['atom_name'])):
                                    if pdbdf['atom_name'][m]=='H' and dista(m,i)<1.2 and dista(m,i)>0.5:
                                        ii=ii+1
                                        print('alcohol for atoms:',i+1,j+1,k+1,l+1,m+1)
print('number of alcohol groups:',ii/2)
                                        
#%% primary amine
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='N':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.6 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,j)<1.6 and dista(k,j)>1.1:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='C' and dista(l,j)<1.6 and dista(l,j)>1.1 and l!=k:
                                for m in range(len(pdbdf['atom_name'])):
                                    if pdbdf['atom_name'][m]=='H' and dista(m,i)<1.2 and dista(m,i)>0.5:
                                        for n in range(len(pdbdf['atom_name'])):
                                            if pdbdf['atom_name'][n]=='H' and dista(n,i)<1.2 and dista(n,i)>0.5 and n!=m:
                                                ii=ii+1
                                                print('primary amine for atoms:',i+1,j+1,k+1,l+1,m+1,n+1)
                                              
print('number of primary amines:', ii/4)                                                
                                                
                                                # print('primary amine for atoms:',i+1,j+1,k+1,l+1,m+1)
#%% secondary amine
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='N':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.6 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,i)<1.6 and dista(k,i)>1.1 and k!=j:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='H' and dista(l,i)<1.2 and dista(l,i)>0.5:
                                ii=ii+1
                                print('secondary amine for atoms:',i+1,j+1,k+1,l+1)
print('number of secondary amines:', ii/4)
#%% tertiary amine
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='N':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.6 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,i)<1.6 and dista(k,i)>1.1 and k!=j:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='C' and dista(l,i)<1.6 and dista(l,i)>1.0 and l!=k and l!=j:
                                ii=ii+1
                                print('tertiary  amine for atoms:',i+1,j+1,k+1,l+1)
print('number of tertiary amines:', ii/6)
                                               
                              
#%% keton
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.26 and dista(i,j)>1.0 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,j)<1.6 and dista(k,j)>1.1:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='C' and dista(l,j)<1.6 and dista(l,j)>1.1 and l!=k:
                                ii=ii+1
                                print('ketone for atoms:',i+1,j+1,k+1,l+1)
print('number of ketone groups:',ii/2)
#%% aldehyde
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.26 and dista(i,j)>1.0 :
                for k in range(len(pdbdf['atom_name'])):                  
                    if pdbdf['atom_name'][k]=='C' and dista(k,j)<1.6 and dista(k,j)>1.1:
                        for l in range(len(pdbdf['atom_name'])):
                            if pdbdf['atom_name'][l]=='H' and dista(l,j)<1.5 and dista(l,j)>0.8 and angle(j,l,i)>118:
                                print('aldehyde for atoms:',i+1,j+1,k+1,l+1)
                                ii=ii+1
print('number of aldehyde groups:',ii)
#%%                                                                               
#amide
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.30 and dista(i,j)>1.19 :
                for k in range(len(pdbdf['atom_name'])):
                    
                    if pdbdf['atom_name'][k]=='N' and dista(k,j)<1.6 and dista(k,j)>1.19:
                        # print(k,i)
                        for l in range(len(pdbdf['atom_name'])):
                            if (pdbdf['atom_name'][l]=='C' and dista(l,k)<1.6 and l!=j) or(pdbdf['atom_name'][l]=='C' and dista(l,i)<1.45 and l!=j):
                                print('amide for atoms:',j+1,i+1,k+1,l+1)   
                                ii=ii+1
print('number of amide groups:',ii)

#%%                                                                               
#ether
ii=0
duma=[]
oxyarr=[]
for m in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][m]=='O':
        oxyarr.append(m)
        oxyarr=list(set(oxyarr))                                    
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='O':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.6 and dista(i,j)>1.19 :
                duma=[]
                for k in range(len(pdbdf['atom_name'])):
                    if pdbdf['atom_name'][k]=='C' and dista(k,i)<1.6 and dista(k,i)>1.19 and k!=j:  
                        for l in range(len(pdbdf['atom_name'])):
                            if (dista(k,l)<1.6 and dista(k,l)>0.5 and l!=i) or (dista(j,l)<1.6 and dista(j,l)>0.5 and l!=i):
                                duma.append(l)
                                duma=list(set(duma))
                                # print(duma)
                                if intersection(duma, oxyarr)==[] and len(duma)==6:
                                    ii=ii+1
                                    print('ether found for atoms:',i+1,j+1,k+1,l+1)
print('number of ether groups:',ii/2)
                                    
#%%                                                                               
#cyanide
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='N':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.25 :
                ii=ii+1
                print('cyanide for atoms:',j+1,i+1)
print('number of cyanide groups:',ii)

#%%                                                                               
#cyanate
ii=0
for i in range(len(pdbdf['atom_name'])):
    if pdbdf['atom_name'][i]=='N':
        for j in range(len(pdbdf['atom_name'])):
            if pdbdf['atom_name'][j]=='C' and dista(i,j)<1.20:
                for k in range(len(pdbdf['atom_name'])):
                    if pdbdf['atom_name'][k]=='O' and dista(k,j)<1.5 and dista (i,k) <3 and angle(j,i,k) >150 and angle(j,i,k) <190:
                        ii=ii+1
                        print('cyanate for atoms:',j+1,i+1,k+1)
print('number of cyanide groups:',ii)
                
                                                                    
                                    
                                
                                
                                        
                                        
                                    
                                
                    