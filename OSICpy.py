# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:17:06 2020

@author: mazal
"""

"""
=========================================
Main Python File
=========================================
Purpose: 

"""

print("=========================================")
print("Project commissioning | DICOM solution")
print("=========================================")

"""
=========================================
Phase 0: Initiating | Product Type Selection
=========================================

Purpose: To get a UID dataset for the DICOM files in the project

"""

# Select Product Type

print("=========================================")
print("Phase 0: Initiating | Product Type Selection")
print("=========================================")
productTypeSelection = input("Insert the number option: (1) Population | (2) Prototype | (3) Sampling | : ")
print("\n=========================================")

productTypeSelection = int(productTypeSelection)

if (productTypeSelection == 1): productType = 'population'

if (productTypeSelection == 2): productType = 'prototype'

if (productTypeSelection == 3):
    productType = 'sampling'
    print("=========================================")
    print("Sampling Size Determination for the training dataset")
    print("=========================================")
    samplingSize = input("  Insert the sampling size (0-100% of the dataset): ")
    print("\n=========================================")

#print("=========================================")

print("=========================================")
print("Phase 0| Report")
print("=========================================")
print("  Product Type: ", productType)
if(productTypeSelection == 3): print("  Sampling Size: ", samplingSize, "%")
print("=========================================")


"""
=========================================
Phase 1: Get Requirements for Preconditions | Get the dataset
=========================================

Purpose: To get the dataset of the DICOM files in the project according to the following options related to raw data ingest: (1) Complete ingestion; (2) Partial ingestion (aleatory sampling); (3) Undefined ingestion (prototyping purposes)

"""

print("=========================================")
print("Phase 1a: Conditioning | Get dataset - raw level")
print("=========================================")
print("  Dataset location (main root): ", 'Y:/Kaggle_OSIC/2-Data/')
print("=========================================")
print("Phase 1| Report")
print("=========================================")

if (productType == 'population'):
    productType_root = 'Y:/Kaggle_OSIC/2-Data/'
    print("  Dataset location (root): ", productType_root)
    print("  Dataset location (training dataset): ", productType_root+'train/')
    print("  Dataset location (testing dataset): ", productType_root+'test/')

if (productType == 'prototype'):
    productType_root = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    print("  Dataset location (root): ", productType_root)
    print("  Dataset location (training dataset): ", productType_root+'train/')
    print("  Dataset location (testing dataset): ", productType_root+'test/')

if (productType == 'sampling'):
    productType_root = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    import os
    path = 'Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/'
    os.chdir(path)
    import generic_pydicom_ns as gpn
    gpn.trainDatasetSampler(samplingSize,testMode=False,reportMode=False)
    print("  Dataset location (root): ", productType_root)
    print("  Dataset location (training dataset): ", productType_root+'train/')
    print("  Dataset location (testing dataset): ", productType_root+'test/')

print("=========================================")


print("=========================================")
print("Phase 1b: Conditioning | Get UID dataset")
print("=========================================")
print("  Dataset location (outcome root): ", productType_root+'outcome/')
print("=========================================")
print("Phase 1b| Report")
print("=========================================")
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
os.chdir(path)
import generic_pydicom as gp
resultPhase1a = gp.UID_dataset_generator(productType, splitType = 'test', testMode = False)
resultPhase1b = gp.UID_dataset_generator(productType, splitType = 'train', testMode = False)
print("  UID dataset - Test split (location): ",resultPhase1a[1])
print("  UID dataset - Test split (file): ",resultPhase1a[2])
print("  UID dataset - Train split (location): ",resultPhase1b[1])
print("  UID dataset - Train split (file):", resultPhase1b[2])
print("=========================================")


"""
=========================================
Phase 2: Preconditions (Identifier of packages to conduct Transfer Syntax UID)
=========================================

Purpose: To get the preconditions understood as the supported transfer syntaxes for the project

"""

print("=========================================")
print("Phase 2: Conditioning | Get UID dataset")
print("=========================================")
print("  Dataset location (outcome root): ", productType_root+'outcome/')
print("=========================================")
print("Phase 2| Report")
print("=========================================")
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
os.chdir(path)
import generic_pydicom as gp
resultPhase2 = gp.packageIdentifierSummary(productType, testMode = False)
SupportedTransferSyntaxes = []
for i in resultPhase2[0].index: SupportedTransferSyntaxes = SupportedTransferSyntaxes + [i]
resultPhase2_UID = resultPhase2[1].loc[SupportedTransferSyntaxes]   
print("  Supported Transfer Syntaxes are required for the following UIDs")
for i in resultPhase2_UID.UID: print("    ",i)
print("=========================================")


"""
=========================================
Phase 3: Get processed images of Dicom data | Level 1: jpg images
=========================================

Purpose: To get procssed images per each file given an ID

"""

# Set parameters for imageProcesserIDSeries
import os
os.chdir('Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/')
import generic_pydicom
from generic_pydicom import imageProcesserIDSeries

ID_List = os.listdir(productType_root)
splitType_List = ['train', 'test']

for i in splitType_List:
    ID_List = os.listdir(productType_root+i+'/')
    for j in ID_List:
        ProductType = productType
        splitType = i
        ID = j
        layoutOption = False
        interpolationMethod = None, ' '
        testMode = False
        plottingMode = False
        
        imageProcesserIDSeries(ProductType,splitType,ID,layoutOption,interpolationMethod,plottingMode,testMode)

print("=========================================")
print("Phase 3: Visualization | Get jpg files from DICOM data")
print("=========================================")
print("  Dataset location (outcome root): ", productType_root+'outcome/'+'jpgImagesInterpolation/')
print("=========================================")


"""
=========================================
Phase 4: Get processed images of Dicom data | Level 2: RGB DataFrames from jpg images
=========================================

Purpose: To get processed images per each file given an ID

"""

# Set parameters for RGB_DataFrame_Maker
import os
os.chdir('Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/')
import generic_imageio
from generic_imageio import RGB_DataFrame_Maker

ID_List = os.listdir(productType_root)
splitType_List = ['train', 'test']

for i in splitType_List:
    ID_List = os.listdir(productType_root+i+'/')
    for j in ID_List:
        ProductType = productType
        splitType = i
        ID = j
        interpolationMethod = None
        testMode = False
        
        RGB_DataFrame_Maker(ProductType,splitType,ID,interpolationMethod,testMode)

print("=========================================")
print("Phase 4: Visualization | Get jpg files from DICOM data")
print("=========================================")
print("  Dataset location (outcome root): ", productType_root+'outcome/'+'jpgImageProcessing/')
print("=========================================")


"""
=========================================
Phase 5: Get processed images of Dicom data | Level 3: Compute space index (Index Type 1) on Week 0
=========================================

Purpose: To compute type-1 space index per each file given an ID

"""

# Import library generic_pydicom_ns
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/'
os.chdir(path)
import generic_pydicom_ns
from generic_imageio_ns import GetIndexType1_ScalingMode


# Set function parameters
ProductType = productType
interpolationMethod = None
PydicomMode = False
reportMode = True
testMode = False

# Run function
GetIndexType1_ScalingMode(productType,interpolationMethod,testMode,reportMode)

"""
=========================================
Phase 6: Precondition Machine Learning assets | Level 0:  Build dataset (Stacking solution case) to process with ML models without Pydicom data
=========================================

Purpose: To build an input dataset to be processed with an stacking solution

"""

# Import library generic_pydicom_ns
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/'
os.chdir(path)
import generic_pydicom_ns
from generic_pydicom_ns import stacking_Dataset_Builder

# Set function parameters
ProductType = productType
PydicomMode = True
reportMode = True

# Run function
resultPhase6 = stacking_Dataset_Builder(ProductType, PydicomMode, reportMode, testMode = False)

"""
=========================================
Phase 7: Precondition Machine Learning assets | Level 1: Build dataset without Pydicom files
=========================================

Purpose: To create a dataset available to be processed using machine learning model(s)

"""

# Import library MachineLearningModel_PydicomFeatures
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/MLModel/'
os.chdir(path)
import MachineLearningModel_PydicomFeatures
from MachineLearningModel_PydicomFeatures import datasetBuilder_ML

# Set function parameters
datasetType = 'pydicom competition - Image processing'
ProductType = productType
PydicomMode = True
reportMode = True
    
# Run function
resultPhase7 = datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode = False)


"""
=========================================
Phase 8: Precondition Machine Learning assets | Level 2: Make Pipeline to preprocess the data
=========================================

Purpose: Preprocess a dataset available to be processed using machine learning model(s) by building a pipeline

"""

# Import library MachineLearningModel_PydicomFeatures
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/MLModel/'
os.chdir(path)
import MachineLearningModel_PydicomFeatures
from MachineLearningModel_PydicomFeatures import pipelineMaker_MLPreprocessor

# Set function parameters
X = resultPhase7[0][0]
y = resultPhase7[0][1]
    
# Run function
reportMode = True
testMode = False
resultPhase8 = pipelineMaker_MLPreprocessor(X,y,reportMode,testMode)


"""
=========================================
Phase 9: Precondition Machine Learning assets | Level 3: Stack of predictors on a single data set
=========================================

Purpose: Get stack of predictors on a single data set.

"""

# Import library MachineLearningModel_PydicomFeatures
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/MLModel/'
os.chdir(path)
import MachineLearningModel_PydicomFeatures
from MachineLearningModel_PydicomFeatures import stackingSolution

# Set function parameters
ProductType = productType
processor_lin = resultPhase8[0][0]
processor_nlin = resultPhase8[0][1]

## Run function
reportMode = True
resultPhase9 = stackingSolution(processor_lin,processor_nlin,reportMode,testMode=False)


"""
=========================================
Phase 10: Conduct Machine Learning solution | Level 4: Get measures and plotting results
=========================================

Purpose: Get measures and plotting result.

"""

# Import library MachineLearningModel_PydicomFeatures
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/MLModel/'
os.chdir(path)
import MachineLearningModel_PydicomFeatures
from MachineLearningModel_PydicomFeatures import plot_regression_results
from MachineLearningModel_PydicomFeatures import getResults

# Set function parameters

# Must Input | Dataset: 'Ames housing dataset' | 'pydicom competition - prototype' | 'pydicom competition'

# Set Product type
ProductType = productType
X = resultPhase7[0][0]
y = resultPhase7[0][1]
stacking_regressor = resultPhase9[0][0]
estimators = resultPhase9[0][1]

# Run  "GetResults" and "plot_regression_results"
reportMode = True
testMode = False
pydicomMode = True
resultPhase10 = getResults(X,y,estimators,stacking_regressor,ProductType,pydicomMode,reportMode,testMode=False)

ax = resultPhase10[0][0]
score = resultPhase10[0][1]
elapsed_time = resultPhase10[0][2]
y_pred = resultPhase10[0][3]
result_dataset = resultPhase10[0][4]


"""
=========================================
Phase 11: Conduct Machine Learning solution | Level 5: Dataset builder (Stacking solution case) after ML outcome 
=========================================

Purpose: To build a submission CSV file (Stacking solution case; includes confidence measure or metric computation)

About the Shape Parameter

Shape Parameter amounts to c = 0.12607421874999922 for every instance in the oject of concern. c value has been computed deeming the
following data fitting scope: (1) Data: FVC predictions; (2) Probability density function as follows (staistical function in scipy
renowend as scipy.stats.loglaplace): loglaplace.pdf(x, c, loc=0, scale=1)

"""

# Import library generic_pydicom_ns
import os
path = 'Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/'
os.chdir(path)
import generic_pydicom_ns
from generic_pydicom_ns import Stacking_Submission_Dataset_Builder

# Set function parameters
ProductType = productType
shapeParameter_DataFrame = []
testMode = False
pydicomMode = True

## Run function
Stacking_Submission_Dataset_Builder(ProductType,shapeParameter_DataFrame,pydicomMode,testMode)

"""
=========================================
Reference
=========================================

Mason, D. L., et al, pydicom: An open source DICOM library, https://github.com/pydicom/pydicom [Online; accessed YYYY-MM-DD].


"""