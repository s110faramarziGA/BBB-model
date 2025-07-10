# QSAR Model for Blood-Brain Barrier Permeability Prediction

This repository contains a collection of machine learning models and scripts to predict blood-brain barrier permeability using molecular fingerprints. The models include neural networks, support vector machines (SVMs), and k-nearest neighbors (k-NN). The implementation utilizes Python libraries such as TensorFlow, Scikit-learn, and Pandas.
The initial dataset was adapted from a published study (https://www.frontiersin.org/articles/10.3389/fphar.2022.1040838/full), in which two QSAR software packages—Leadscope Enterprise (LS) version 3.9 (Instem Inc., United States) and CASE Ultra (CU) version 1.8.0.1 (MultiCASE Inc., United States)—were used to construct binary QSAR models for predicting the blood-brain barrier (BBB) permeability of small molecules. 


---

## About the Project

The goal of this project is to develop neural networks that (1) utilize new and more descriptive molecular fingerprints, (2) predict blood-brain barrier (BBB) permeability with greater precision, and (3) seamlessly incorporate new input datasets without requiring model reconstruction.

2D molecular fingerprints are generated using the Protein Data Bank (PDB) structures of the molecules, along with physicochemical properties such as molecular mass, MLogP, topological polar surface area (TPSA), number of hydrogen bond acceptors (HAccept), hydrogen bond donors (HDon), and rotatable bonds (nRotB). This database compiles rodent-derived data, capturing either blood/plasma (B/P) or blood/brain (B/B) ratios following compound administration via intravenous, intraperitoneal, or oral routes. Typically, researchers measured chemical concentrations in brain tissue and in blood or plasma between 30 minutes and several hours after dosing. Each record was ultimately assigned a binary classification for model development: substances with log BB values of –1 or higher were labeled as permeable ('1'), while those below –1 were deemed non-permeable ('0').

---

## Technologies Used

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
- Pandas
- NumPy

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/s110faramarziGA/BBB-model
