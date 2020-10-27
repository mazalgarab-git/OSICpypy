# -*- coding: utf-8 -*-

"""
Created on Sat Sep  5 13:06:18 2020

@author: mazal
"""

"""
=========================================
Support functions of pydicom
=========================================
Purpose: Create support functions for the pydicom project.

"""

"""
Test mode 1 | Basics
testMode = True
reportMode = False 

Test mode 2 | Function Report
testMode = False
reportMode = True

Commisionning mode
testMode = False
reportMode = False 
"""
testMode = False
reportMode = False
 

"""
=========================================
Function 1: File reader (DICOMDIR tags, DICOMDIR Meta Dataset)
=========================================

Purpose: Extract from a DICOM file the following attributes: (1) DICOMDIR tags; (2) DICOMDIR Meta Dataset
Code Source: -  
Raw code reference (see Tester.py): Test 0

"""

def fileReader(filename, testMode):

    import pydicom
    
    ds = pydicom.dcmread(filename)  # plan dataset
    tagsDICOMDIR = ds.dir()  # get a list of tags with "setup" somewhere in the name
    metaDatasetDICOMDIR = pydicom.filereader.read_file_meta_info(filename)

    return tagsDICOMDIR, metaDatasetDICOMDIR, testMode

# Function Test 1

if (testMode==True):

    filename = '1.dcm'

    import os
    path = 'Y:/Kaggle_OSIC/OSICpypy/Tester/'
    os.chdir(path)
    
    filename = path + '1.dcm'
    resultFunction1 = fileReader(filename, testMode)
    print("=========================================")
    print("DICOMDIR Tags: ")
    print("=========================================")
    print(resultFunction1[0])
    print("=========================================")
    print("DICOMDIR Meta Dataset: ")
    print("=========================================")
    print(resultFunction1[1])
    print("=========================================")
    print("Test result Function 1: Success")
    print("=========================================")


"""
=========================================
Function 2: File reader (DICOMDIR-tag data)
=========================================

Purpose: to extract the Meta Dataset given a DICOMDIR tag
Code Source: - 
Raw code reference (see Tester.py): Test 0

Must inputs: output of Function 1

"""

def fileReaderTag(filename, resultFunction1, tagPosition, testMode):

    import pydicom
    tag = str(resultFunction1[0][tagPosition])
    ds = pydicom.dcmread(filename)
    
    return ds[tag], ds[tag].value, testMode

# Function Test 2

if testMode == True:

    filename = '1.dcm'
    tagRange = len(resultFunction1[0])

    import random
    tagPosition = random.randint(0,tagRange)  
    resultFunction2 = fileReaderTag(filename, resultFunction1, tagPosition, testMode)

    print("=========================================")
    print("Meta dataset of the DICOMDIR tag:")
    print("=========================================")
    print(resultFunction2[0])
    print("=========================================")
    print("Value of the DICOMDIR tag")
    print("=========================================")
    print(resultFunction2[1])
    print("=========================================")
    print("Test result Function 2: Success")
    print("=========================================")


"""
=========================================
Function 3: TransferSyntax
=========================================

Purpose: To get image data from compressed DICOM images
Code Source: - 
Raw code reference (see Tester.py): Test 1

Must inputs: output of Function 1

"""

def TransferSyntax(filename, testMode):

    from pydicom import dcmread
    ds = dcmread(filename)
    
    return ds.file_meta.TransferSyntaxUID, testMode

if testMode == True:

    filename = 'Y:/Kaggle_OSIC/OSICpypy/Tester/1.dcm'
    resultFunction3 = TransferSyntax(filename,testMode)

    print("=========================================")
    print("Transfer Syntax UID:")
    print("=========================================")
    print(resultFunction3[0])
    print("=========================================")
    print("Test result Function 3: Success")
    print("=========================================")   


"""
=========================================
Function 4: Generate a UID dataset
=========================================

Purpose: To generate a UID dataset for identifying the packages to conduct Transfer Syntax UID
Code Source: - 
Raw code reference (see Tester.py): Test 2

Must inputs: dataset of DICOM files

"""

def UID_dataset_generator(productType, splitType, testMode):

    import os
    import numpy as np
    import pandas as pd
    
    path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
    os.chdir(path)
    from generic_pydicom import TransferSyntax
    
    
    # Set productType and splitType
    
    if productType == 'population':
        if splitType == 'test':
            path_IDs = 'Y:/Kaggle_OSIC/2-Data/test/'
        else:
            path_IDs = 'Y:/Kaggle_OSIC/2-Data/train/'
    
    if productType == 'prototype':
        if splitType == 'test':
            path_IDs = 'Y:/Kaggle_OSIC/3-Data (Prototype)/test/'
        else:
            path_IDs = 'Y:/Kaggle_OSIC/3-Data (Prototype)/train/'
    
    if productType == 'sampling':
        if splitType == 'test':
            path_IDs = 'Y:/Kaggle_OSIC/4-Data (Sampling)/test/'
        else:
            path_IDs = 'Y:/Kaggle_OSIC/4-Data (Sampling)/train/'
            
    # Get Feature 1: IDs List
    list_IDs = os.listdir(path_IDs)
    
    # Get Transfer Syntax UID | Phase 1: Keys and values to build a dictionary
    
    TransferSyntaxUID_Dataset_keys = []
    TransferSyntaxUID_Dataset_values = []
    TransferSyntaxUID_Dataset_subvalues = []
    
    list_IDs_range = len(list_IDs)
    
    for i in range(0,list_IDs_range):
        #print("ID under iteration: ",list_IDs[i])
        pathFiles = path_IDs + list_IDs[i]
        #print("PathFile ", pathFiles)
        list_IDs_files = os.listdir(pathFiles)
        #print("List of IDs files",list_IDs_files)
        TransferSyntaxUID_Dataset_keys = TransferSyntaxUID_Dataset_keys + [list_IDs[i]]
        #print("keys", TransferSyntaxUID_Dataset_keys)
        list_IDs_files_range = len(list_IDs_files)
        
        for j in range(0,list_IDs_files_range):
            filename = pathFiles + '/' + list_IDs_files[j]
            UID = TransferSyntax(filename, testMode=False)
            UID = UID[0]
            filename_UID = (list_IDs[i],list_IDs_files[j],UID)
            filename_UID = np.array(filename_UID) 
            if(testMode==True):print("(Filename,UID):", filename_UID)
            TransferSyntaxUID_Dataset_subvalues = TransferSyntaxUID_Dataset_subvalues + [filename_UID]
            #print("values", TransferSyntaxUID_Dataset_values)
        
        TransferSyntaxUID_Dataset_values = TransferSyntaxUID_Dataset_values + [TransferSyntaxUID_Dataset_subvalues]
    
    # Get Transfer Syntax UID | Phase 2: Dictionary
    
    UID_Dictionary = {}
    UID_range = len(TransferSyntaxUID_Dataset_keys)
    
    for k in range(0,UID_range):
        UID_key = TransferSyntaxUID_Dataset_keys[k]
        UID_key = str(UID_key)
        UID_Dictionary[UID_key] = TransferSyntaxUID_Dataset_values[k]
    
    # Get Transfer Syntax UID | Phase 3: Dataframe
    
    UID_list = []
    UID_Dictionary_keys = UID_Dictionary.keys()
    
    for l in UID_Dictionary_keys:
        UID_list = UID_list + UID_Dictionary[l] 
    
    UID_list_range = len(UID_list)
    UID_list_dictionary = {}
    
    for m in range(0,UID_list_range):
        UID_list2 = list(UID_list[m])
        
        if (m == 0):            
            UID_list_dictionary = {'ID':UID_list2[0],'FileName':UID_list2[1],'UID':UID_list2[2]}
            indexName = str(m)+'_'+splitType
            UID_DataFrame = pd.DataFrame(data = UID_list_dictionary, index=[indexName])
        else:
            UID_list_dictionary2 = {'ID':UID_list2[0],'FileName':UID_list2[1],'UID':UID_list2[2]}
            indexName = str(m)+'_'+splitType
            UID_DataFrame2 = pd.DataFrame(data = UID_list_dictionary2, index=[indexName])
            UID_DataFrame = pd.concat([UID_DataFrame, UID_DataFrame2])
    
    if(splitType == 'Test'):
        path = path_IDs[:-5] + '/outcome/'
    else:
        path = path_IDs[:-6] + '/outcome/'
        
    filename = 'UID_Dataset' + '_' + splitType + '.csv'
    UID_DataFrame.to_csv(path+filename)
    
    resultFunction = UID_DataFrame

    return resultFunction, path, filename, testMode


if testMode == True:

    productType = 'prototype'
    splitType = 'test'
    resultFunction4 = UID_dataset_generator(productType, splitType, testMode)

    print("=========================================")
    print("UID dataset:")
    print("=========================================")
    print(resultFunction4[0])
    print("=========================================")
    print("Test result Function 4: Success")
    print("=========================================")


"""
=========================================
Function 5: Identify packages to conduct Transfer Syntax UID (Summary, ID filtering, FileName filtering)
=========================================

Purpose: To generate a UID dataset where the supported Transfer Syntaxes are able to visualize including the project data
Code Source: - 
Raw code reference (see Tester.py): Test 3
Supported Transfer Syntaxes sourced from Supported Transfer Syntaxes (2020).

Must inputs: UID dataset that includes code name and instance frequency (i.e. result Function 4; UID_ProductType, UID_ProductType_Frequency)

"""

def packageIdentifierSummary(productType, testMode):
    
    import os
    import pandas as pd
    
    path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
    os.chdir(path)
    
    
    # Set ProductType
    path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/outcome/'
    
    # Set productType
    
    if productType == 'population':path_ProductType = 'Y:/Kaggle_OSIC/2-Data/outcome/'
    
    if productType == 'prototype':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/outcome/'
    
    if productType == 'sampling':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/outcome/'
    
    # Load UID datasets (train and test) | Phase 1: Read CSV files (train and test)
    UID_Dataset_test_filename = 'UID_Dataset_test.csv'
    UID_Dataset_test = pd.read_csv(path_ProductType + UID_Dataset_test_filename)
    UID_Dataset_train_filename = 'UID_Dataset_train.csv'
    UID_Dataset_train = pd.read_csv(path_ProductType + UID_Dataset_train_filename)        
    
    # Load UID datasets (train and test) | Phase 2: Concatenate train and test dataset
    UID_Dataset = pd.concat([UID_Dataset_train, UID_Dataset_test], ignore_index = True)
    
    # Load UID datasets (train and test) | Phase 3: DataFrame manipulation
    UID_Dataset = UID_Dataset.rename(columns={"Unnamed: 0":"Split"})
    
    # Load Transfer Syntax Table
    ## Sourced from https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html)
    TransferSyntax_path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
    TransferSyntax_csv = 'TransferSyntax_Dataset.csv'
    TransferSyntax_Dataset = pd.read_csv(TransferSyntax_path+TransferSyntax_csv)
    
    # Get Summary | Phase 1: Inputs - UID features (UID nameCodes, UID frequency)
    UID_count = UID_Dataset.set_index(["UID", "Split", "FileName"]).count(level="UID")
    
    UID_nameCodes = UID_count.index
    UID_frequency = UID_count.values
    
    # Get Summary | Phase 2: Inputs - Get UID features (index_TransferSyntax for ProductType)
    TransferSyntax_UID_Types = len(TransferSyntax_Dataset['UID'])
    index_TransferSyntax = []
    for i in UID_nameCodes:
        for j in range(0,TransferSyntax_UID_Types):
            indexAssignment = TransferSyntax_Dataset.index[j]
            UID = TransferSyntax_Dataset['UID'][j]
            if (i == UID): index_TransferSyntax = index_TransferSyntax + [indexAssignment]
    
    # Get Summary | Phase 3: Inputs - UID Dataset (ProductType)
    UID_frequency2 = []
    for k in range(0,len(UID_frequency)): UID_frequency2 = UID_frequency2+ [UID_frequency[k][0]]
    UID_Dictionary = {'UID_ProductType':list(UID_nameCodes),'UID_ProductType_Frequency':list(UID_frequency2)}
    UID_Dataset_Summary = pd.DataFrame(data = UID_Dictionary, index=index_TransferSyntax)
    resultFunction1 = UID_Dataset_Summary
    
    # Get Summary | Phase 4: Output - Transfer syntax table including instances per UID of the ProductType
    UID_Dataset_Summary2 = pd.concat([TransferSyntax_Dataset, UID_Dataset_Summary], axis=1, sort=False)
    
    resultFunction2 = UID_Dataset_Summary2

    return resultFunction1, resultFunction2, testMode


if testMode == True:

    productType = 'prototype'
    resultFunction5 = packageIdentifierSummary(productType, testMode)

    print("=========================================")
    print("UID dataset:")
    print("=========================================")
    print(resultFunction5[0])
    print("=========================================")
    print("Test result Function 5: Success")
    print("=========================================")


"""
=========================================
Function 6: Identify packages to conduct Transfer Syntax UID per ID and FileName (ID filtering, FileName filtering)
=========================================

Purpose: To get an instance of the supported Transfer Syntaxes given an ID and a FileName
Code Source: - 
Raw code reference (see Tester.py): Test 4
Supported Transfer Syntaxes sourced from Supported Transfer Syntaxes (2020).

Must inputs: UID dataset (result Function 4) and Supported Transfer Syntaxes dataset

"""


def packageIdentifierUnit(productType, Split, ID, FileName, testMode):

    import os
    import pandas as pd
    
    path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
    os.chdir(path)
        
    # Set ProductType
    
    if productType == 'population':path_ProductType = 'Y:/Kaggle_OSIC/2-Data/outcome/'
    
    if productType == 'prototype':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/outcome/'
    
    if productType == 'sampling':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/outcome/'
    
    # Set Split, ID and filename    
        
    Split_toProcess = Split
    ID_toProcess = ID
    filename_toProcess = FileName
    
    # Load UID datasets (train and test) | Phase 1: Read CSV files (train and test)
    UID_Dataset_test_filename = 'UID_Dataset_test.csv'
    UID_Dataset_test = pd.read_csv(path_ProductType + UID_Dataset_test_filename)
    UID_Dataset_train_filename = 'UID_Dataset_train.csv'
    UID_Dataset_train = pd.read_csv(path_ProductType + UID_Dataset_train_filename)        
    
    # Load UID datasets (train and test) | Phase 2: Concatenate train and test dataset
    UID_Dataset = pd.concat([UID_Dataset_train, UID_Dataset_test], ignore_index = True)
    
    # Load UID datasets (train and test) | Phase 3: DataFrame manipulation
    UID_Dataset = UID_Dataset.rename(columns={"Unnamed: 0":"Split"})
    
    # Load Transfer Syntax Table
    ## Sourced from https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html)
    TransferSyntax_path = 'Y:/Kaggle_OSIC/OSICpypy/Support-Sourced/'
    TransferSyntax_csv = 'TransferSyntax_Dataset.csv'
    TransferSyntax_Dataset = pd.read_csv(TransferSyntax_path+TransferSyntax_csv)
    
    # Get instance | Instance of supported transfer syntax 
    
    UID_Dataset_FileName_range = len(UID_Dataset.FileName)
    
    for i in range(0,UID_Dataset_FileName_range):
        if (UID_Dataset.Split[i][-1] == 't'):
            Split2 = 'test'
        else:
            Split2 = 'train'
        if (UID_Dataset.FileName[i] == filename_toProcess):
            if(UID_Dataset.ID[i] == ID_toProcess):
                if(Split2 == Split_toProcess):
                    UID_ToProcess = UID_Dataset.UID[i]
                    break
    
    TransferSyntax_Dataset_range = len(TransferSyntax_Dataset)
    
    for j in range(0,TransferSyntax_Dataset_range):
        if (TransferSyntax_Dataset.UID[j] == UID_ToProcess):
            TransferSyntax_Instance = TransferSyntax_Dataset.loc[j]
            break
    
    resultFunction = TransferSyntax_Instance
    
    return resultFunction, testMode


if testMode == True:

    productType = 'prototype'
    Split = 'test'
    ID = 'ID00419637202311204720264'
    FileName = '1.dcm'    
    resultFunction6 = packageIdentifierUnit(productType, Split, ID, FileName, testMode)

    print("=========================================")
    print("UID dataset:")
    print("=========================================")
    print(resultFunction6[0])
    print("=========================================")
    print("Test result Function 6: Success")
    print("=========================================")


"""
=========================================
Function 7: Get image visualization
=========================================

Purpose: To get an image visualization
Code Source: - 
Raw code reference (see Tester.py): Test 6

Must inputs: -

"""

def imageVisualizer(filename,testMode):

    import matplotlib.pyplot as plt
    import pydicom
    ds = pydicom.dcmread(filename)
    imageVisualization = plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
    resultFunction = imageVisualization
    
    return resultFunction, testMode

if testMode == True:

    path = 'Y:/Kaggle_OSIC/OSICpypy/Tester/'
    os.chdir(path)
    filename = '1.dcm'    
    resultFunction7 = imageVisualizer(filename,testMode)

    print("=========================================")
    print("DICOM Image visualization:")
    print("=========================================")
    print(resultFunction7[0])
    print("=========================================")
    print("Test result Function 6: Success")
    print("=========================================")

"""
=========================================
Function 8: Get processsed images (ID series)
=========================================

Purpose: To get image visualization per each file given an ID
Code Source: - 
Raw code reference (see Tester.py): Test 6, Test 7

Must inputs: ID, function 7

"""

def imageProcesserIDSeries(productType,splitType,ID,layoutOption,interpolationMethod,plottingMode,testMode):

    import os
    import matplotlib.pyplot as plt
    import matplotlib.image as img
    import pydicom
    import numpy as np
        
    # Conditioning | Phase 0: Set root based on: (1) ProductType; (2) splitType; (3) ID
    
    if productType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if productType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if productType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/'


    if splitType == 'test':
        path_ID = path_ProductType + 'test/' + ID + '/'
    else:
        path_ID = path_ProductType + 'train/' + ID + '/'    
    
    # Conditioning | Phase 1: Build root-output directory jpgImages
    
    import distutils.ccompiler
    path_ProductType_output = path_ProductType + 'outcome/' + 'jpgImagesInterpolation/' + splitType + '/' + str(ID) + '/'
    distutils.dir_util.mkpath(path_ProductType_output)
    
    # Conditioning | Phase 2: Set working directory
    
    os.chdir(path_ID)
    
    # Conditioning | Phase 3: Get list of files (duely ordered)
    
    listFiles = os.listdir(path_ID)
    
    listFilesAdjusted = []
    for i in listFiles:
        newItem = i[:-4]
        listFilesAdjusted = listFilesAdjusted + [newItem]
    
    listFiles = []
    listFilesAdjusted.sort()
    listFiles = listFilesAdjusted.copy()
    
    listFilesAdjusted = []
    for i in listFiles:
        newItem = i + '.dcm'
        listFilesAdjusted = listFilesAdjusted + [newItem]    
    
    # Conditioning | Phase 4: Get validation to start based on interpolationMethod
    
    interpolationMethod_copy = interpolationMethod
    
    interpolationMethod_List = os.listdir(path_ProductType_output)
    
    interpolationMethod_validated = []
    interpolationMethod_NotValidated = []
    
    for i in interpolationMethod:
        
        # ## Verify if a method to use is in output directory
        
        ##Case 0: Method included -> Verify if all the images have been processed
        if i in interpolationMethod_List:
            #if i == j:
            ## Case 0.1: Method to use is in output directory
            filesNumberToProcess = os.listdir(path_ProductType_output+str(i)+'/')
            filesNumberProcessed = os.listdir(path_ID)
            if len(filesNumberToProcess) != len(filesNumberProcessed):
                ## Case 0.1.1: Files amount | no match -> Validation
                if (i not in interpolationMethod_validated):interpolationMethod_validated = interpolationMethod_validated + [i]
            else:
                ## Case 0.1.2: Files amount | match -> No validation 
                #if (i not in interpolationMethod_NotValidated):interpolationMethod_NotValidated = interpolationMethod_NotValidated + [i]
                interpolationMethod_NotValidated = interpolationMethod_NotValidated + [i]
                
        else:
        ##Case 1: Method not included -> method gets validation
            interpolationMethod_validated = interpolationMethod_validated + [i]
    
    interpolationMethod = interpolationMethod_validated
    
    if interpolationMethod_List == []:
        interpolationMethod_validated = interpolationMethod_copy
        interpolationMethod = interpolationMethod_copy
        interpolationMethod_NotValidated = []
    
    # Visualization
    
    ## Visualization | Phase 0: Fixing random state for reproducibility
    np.random.seed(19680801)
    
    ## Visualization | Phase 1: Ploting and saving image files
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    
    for i in listFilesAdjusted:
        
        os.chdir(path_ID)
        ds = pydicom.dcmread(i)
        grid = ds.pixel_array
            
        if (plottingMode == True):
        
            fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        
            for ax, interp_method in zip(axs.flat, methods):
                ax.imshow(grid, interpolation=interp_method, cmap=plt.cm.bone)
                ax.set_title(str(interp_method))
                
                # Image Saving
                os.chdir(path_ProductType_output)
                distutils.dir_util.mkpath(str(interp_method))
                filename = str(i[:-4])+'_'+str(interp_method)+'.jpg'
                path_ID_image = path_ProductType_output + str(interp_method) + '/'
                os.chdir(path_ID_image)
                img.imsave(filename, grid, cmap=plt.cm.bone, format='jpg', dpi=100, metadata=None)
                os.chdir(path_ID)
                
            if(layoutOption == True):
                # Plot Layout
                plt.tight_layout()
                plt.show()
                # Image saving | Layout
                path_ID_imageLayout = path_ProductType_output + 'Layout' + '/'
                distutils.dir_util.mkpath(path_ID_imageLayout)
                os.chdir(path_ID_imageLayout)
                filename_layout = str(i[:-4])+'_layout'+'.jpg'
                fig.savefig(filename_layout,bbox_inches='tight')
        
        else:
            
            for interp_method in methods:
                
                # Image Saving
                
                if (str(interp_method) in interpolationMethod):
                    os.chdir(path_ProductType_output)
                    distutils.dir_util.mkpath(str(interp_method))
                    filename = str(i[:-4])+'_'+str(interp_method)+'.jpg'
                    path_ID_image = path_ProductType_output + str(interp_method) + '/'
                    os.chdir(path_ID_image)
                    img.imsave(filename, grid, cmap=plt.cm.bone, format='jpg', dpi=100, metadata=None)
                    os.chdir(path_ID)
                
                if (interp_method is None):
                    os.chdir(path_ProductType_output)
                    distutils.dir_util.mkpath(str(interp_method))
                    filename = str(i[:-4])+'_'+str(interp_method)+'.jpg'
                    path_ID_image = path_ProductType_output + str(interp_method) + '/'
                    os.chdir(path_ID_image)
                    img.imsave(filename, grid, cmap=plt.cm.bone, format='jpg', dpi=100, metadata=None)
                    os.chdir(path_ID)

    # Closing: Function Result
    functionResult = interpolationMethod_validated,interpolationMethod_NotValidated,listFilesAdjusted
                
    # Closing | Report
    if testMode == False:
        print("===========================================")
        print("Product Type: ",productType)
        print("Split Type: ",splitType)
        print("Patient ID: ",ID)
        print("Interpolation Method(s) - Validated (included): ", functionResult[0])
        print("Interpolation Method(s) - Not validated (not included): ", functionResult[1])
        print("layout included: ",layoutOption)
        print("Processed images per method: ", len(functionResult[2]))
        print("===========================================")
    
    return functionResult


if testMode == True:

    productType = 'prototype'
    splitType = 'train'
    #ID = 'ID00007637202177411956430' # Image type 1
    #ID = 'ID00011637202177653955184'
    #ID = 'ID00009637202177434476278' # Image type 2
    ID = 'ID00014637202177757139317' # Image type 3
    layoutOption = False
    interpolationMethod = None, ' '
    #interpolationMethod = None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16','spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric','catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    plottingMode = False
    
    resultFunction8 = imageProcesserIDSeries(productType,splitType,ID,layoutOption,interpolationMethod,plottingMode,testMode)

if reportMode == True:
    print("===========================================")
    print("Product Type: ",productType)
    print("Split Type: ",splitType)
    print("Patient ID: ",ID)
    print("Interpolation Method(s) - Validated (included): ", resultFunction8[0])
    print("Interpolation Method(s) - Not validated (not included): ", resultFunction8[1])
    print("layout included: ",layoutOption)
    print("Processed images per method: ", len(resultFunction8[2]))
    print("===========================================")



"""
=========================================
Reference
=========================================

Mason, D. L., et al, pydicom: An open source DICOM library, https://github.com/pydicom/pydicom [Online; accessed YYYY-MM-DD].

Supported Transfer Syntaxes.(2020). Available at https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html


"""