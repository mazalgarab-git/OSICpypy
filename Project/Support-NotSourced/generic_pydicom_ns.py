# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:48:59 2020

@author: mazal
"""

"""
=========================================
Support functions of pydicom (Not sourced)
=========================================
Purpose: Create support functions for the pydicom project

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
Function 1: Aleatory Sampling
=========================================

Purpose: Build an aleatory sample given a train dataset of Kaggle for competition and a sample size

Raw code reference (see Tester.py): Test 5
"""

def trainDatasetSampler(samplingSize,testMode,reportMode):

    # Set sampling size (% of the train population)
    samplingSize = 5
    
    # Build a Sampling dataset | Phase 1: Determine: (1) the source path of the train data; (2) the location path of the sampling
    
    import os
    import pandas as pd
    
    path_source = 'Y:/Kaggle_OSIC/2-Data/train/'
    path_source_test = 'Y:/Kaggle_OSIC/2-Data/test/'
    path_destination = 'Y:/Kaggle_OSIC/4-Data (Sampling)/train/'
    path_destination_test = 'Y:/Kaggle_OSIC/4-Data (Sampling)/test/'
    path_destination_outcome = 'Y:/Kaggle_OSIC/4-Data (Sampling)/outcome/'
    
    
    # Build a Sampling dataset | Phase 2: Build dataset using the following features from train data: (1) ID; (2) # of DICOM files per ID (including percentage).
    ## Improvement: (3) # of other registers (not related to DICOM files)
    os.chdir(path_source)
    
    ID_list = os.listdir(path_source)
    ID_list_range = len(ID_list)
    
    DICOMFile_list = []
    DICOMFileNumber_list = []
    
    for i in range(0,ID_list_range):
        path_ID = path_source + ID_list[i] + '/'
        DICOMFile_list_unitary = os.listdir(path_ID)
        DICOMFile_list = DICOMFile_list + [DICOMFile_list_unitary]
        DICOMFileNumber_list_unitary = len(DICOMFile_list_unitary)
        DICOMFileNumber_list = DICOMFileNumber_list + [DICOMFileNumber_list_unitary]
         
    Population_Dictionary = {'ID':ID_list,'NumberDicomFiles':DICOMFileNumber_list,'DicomFIles':DICOMFile_list}
    
    Population_DataFrame = pd.DataFrame(data = Population_Dictionary)
    
    DICOMFilePercentage_list = []
    TotalNumberDicomFiles = sum(Population_DataFrame.NumberDicomFiles)
    
    for j in range(0,ID_list_range):
        Percentage = Population_DataFrame['NumberDicomFiles'][j] / TotalNumberDicomFiles * 100
        Percentage = round(Percentage,6)
        DICOMFilePercentage_list = DICOMFilePercentage_list + [Percentage]
    
    Population_Percentage_Dictionary = {'Percentage':DICOMFilePercentage_list}
    Population_Percentage_DataFrame = pd.DataFrame(data=Population_Percentage_Dictionary)
    
    Population_DataFrame = pd.concat([Population_DataFrame, Population_Percentage_DataFrame],axis=1, sort=False)
    
    filename_population = 'populationDataset.csv'
    path_population = path_destination_outcome
    Population_DataFrame.to_csv(path_population+filename_population)
    
    # Build a Sampling dataset | Phase 3: Get an aleatory grouping of IDs (just tags)
    
    import random
    
    Population_DataFrame_IndexToSample=[]
    Population_DataFrame_IDToSample=[]
    Population_DataFrame_PercentageToSample=[]
    samplingSizeGoal = 0
    
    while (samplingSizeGoal <= samplingSize):
        randomNumberTermination = len(Population_DataFrame.ID)
        randomNumber = random.randrange(0,randomNumberTermination,1)
        if (randomNumber not in Population_DataFrame_IndexToSample):
            Population_DataFrame_IndexToSample = Population_DataFrame_IndexToSample + [randomNumber]
            ID_unitary = Population_DataFrame.ID[randomNumber]
            Population_DataFrame_IDToSample = Population_DataFrame_IDToSample + [ID_unitary]
            Percentage_unitary = Population_DataFrame.Percentage[randomNumber]
            Population_DataFrame_PercentageToSample = Population_DataFrame_PercentageToSample + [Percentage_unitary]
            samplingSize_unitary = Population_DataFrame.Percentage[randomNumber]
            samplingSizeGoal = samplingSizeGoal + samplingSize_unitary
    
    
    samplingDataset_Dictionary = {'Index':Population_DataFrame_IndexToSample,'ID':Population_DataFrame_IDToSample,'Percentage':Population_DataFrame_PercentageToSample}
    samplingDataset_DataFrame = pd.DataFrame(data=samplingDataset_Dictionary)
    
    filename_sampling = 'samplingDataset.csv'
    path_sampling = path_destination_outcome
    samplingDataset_DataFrame.to_csv(path_sampling+filename_sampling)
    
    # Build a Sampling dataset | Phase 3: Get train dataset (an aleatory grouping of IDs; tree-copy task)
    
    from distutils.dir_util import create_tree
    from distutils.dir_util import remove_tree
    from distutils.dir_util import copy_tree
    
    remove_tree(path_destination)
    create_tree(path_destination,[])
    
    if testMode == True:
        print("=========================================")
        print("Building the Sampling Dataset given the Train Dataset of Kaggle for competition")
        print("=========================================")
    
    for k in Population_DataFrame_IDToSample:
    
        path_source_unitary =  path_source + k + '/'
        path_destination_unitary = path_destination + k + '/'
        create_tree(path_destination_unitary,[])
        copy_tree(path_source_unitary,path_destination_unitary)
        if testMode == True: print("ID tree copied: ",k)
    
    # Build a Sampling dataset | Phase 4: Get test dataset (tree-copy task)
    ## Assumption: The complete test dataset is copied.
    
    from distutils.dir_util import create_tree
    from distutils.dir_util import remove_tree
    from distutils.dir_util import copy_tree
    
    remove_tree(path_destination_test)
    create_tree(path_destination_test,[])
    
    if testMode == True:
        print("=========================================")
        print("Building the Test Dataset given the Test Dataset of Kaggle for competition")
        print("=========================================")
    
    IDList_test = os.listdir(path_source_test)
    
    for l in IDList_test:
    
        path_source_unitary =  path_source + l + '/'
        path_destination_unitary = path_destination_test + l + '/'
        create_tree(path_destination_unitary,[])
        copy_tree(path_source_unitary,path_destination_unitary)
        if testMode == True: print("ID tree copied: ",l)
    
    if (testMode == False and reportMode == True):
        from datetime import date
        reportDate = date.today()
        print("=========================================")
        print("Function Report | Date:",reportDate.year,'/',reportDate.month,'/',reportDate.day,'/' )
        print("=========================================")
        print("Function: trainDatasetSampler(samplingSize,testMode)")
        print("=========================================")
        print("(1) Inputs")
        print("=========================================")
        print("-Sampling Size :", samplingSize, "%")
        print("-Test Mode : False")
        print("=========================================")
        print("(2) Outputs")
        print("=========================================")
        print("-Type of sample: Aleatory based on IDs")
        print("-Train dataset percentage to sample (base): ", round(abs(samplingSize),6),"%")
        print("-Train dataset percentage to sample (adjustment): ", round(abs(samplingSizeGoal-samplingSize),6),"%")
        print("-Train dataset percentage to sample (fitted): ", round(samplingSizeGoal,6),"%")
        print("-Population of Train dataset (just information) available in file: ", filename_population)
        print("-Sample of Train dataset (just information) available in file: ", filename_sampling)
        print("=========================================")
        print("(2) Outcomes:")
        print("=========================================")
        print("Being the outcome expressed under the variable result, outcomes are as follows:")
        print("result[0] -> Dataframe for Population")
        print("result[1] -> Dataframe for Sample")
        print("result[2] -> Test Mode")
        print("result[3] -> Rerport Mode")
        print("=========================================")
    
    return Population_DataFrame, samplingDataset_DataFrame, testMode, reportMode 


if testMode == True:

    samplingSize = 5    
    resultFunction1 = trainDatasetSampler(samplingSize,testMode,reportMode)

    print("=========================================")
    print("Population dataset:")
    print("=========================================")
    print(resultFunction1[0])
    print("=========================================")
    print("Population dataset:")
    print("=========================================")
    print(resultFunction1[1])
    print("=========================================")
    print("Test result Function 1: Success")
    print("=========================================")


"""
=========================================
Function 2: Submission Builder
=========================================

Purpose: Build a submission CSV file

Raw code reference (see Tester.py): Test 8
"""

def SubmissionBuilder(ProductType,filename,testMode):

    import os
    import pandas as pd
    
    # Set ProductType
    path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
        # Set productType and splitType
    
    if ProductType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if ProductType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if ProductType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    
    
    # Set outcome
    path_outcome = path_ProductType + 'outcome/'
    
   
    # Get raw data as a DataFrame
    os.chdir(path_outcome)
    rawFile_DataFrame = pd.read_csv('submissionRawFile_2020_09_19.csv')
    
    # Get submission file template as a DataFrame
    os.chdir(path_ProductType)
    submissionFile_DataFrame = pd.read_csv('sample_submission.csv')
    
    # Get submission data as required in submission file
    submissionNumber_range = len(rawFile_DataFrame.index)
    IDcases_List = submissionFile_DataFrame.Patient_Week.copy()
    IDcases_List = IDcases_List[0:5]
    
    IDcases_List_range = len(IDcases_List)
    
    for i in range (0,IDcases_List_range):
        IDcases_List[i] = IDcases_List[i][:-4]
    
    # Get submission data as required in submission file | FVC    
    FVCDataList = []
       
    for k in range(0,submissionNumber_range):
        
        for j in IDcases_List:
            
            # Get datum in raw data 
            IDlabel_rawFile = str(j)+str('_FVC')
            datum = rawFile_DataFrame[IDlabel_rawFile][k]
            datum = round(datum,0)
            # Set datum in submission file
            FVCDataList = FVCDataList + [datum]
        
    submissionFile_DataFrame['FVC'] = FVCDataList
    
    # Get submission data as required in submission file | Confidence    
    CONDataList = []
       
    for k in range(0,submissionNumber_range):
        
        for j in IDcases_List:
            
            # Get datum in raw data 
            IDlabel_rawFile = str(j)+str('_CON')
            datum = rawFile_DataFrame[IDlabel_rawFile][k]
            datum = round(datum,0)
            # Set datum in submission file
            CONDataList = CONDataList + [datum]
        
    submissionFile_DataFrame['Confidence'] = CONDataList
    
    # Save file | Get directory

    path_destination = path_outcome+'submissions/'

    try:
        os.chdir(path_destination)
        GetCreation = True
            
    except FileNotFoundError:
        GetCreation = False
        
    if GetCreation == False:
        from distutils.dir_util import mkpath
        mkpath(path_destination)
        os.chdir(path_destination)
        
    submissionList = os.listdir(path_destination)
    number = len(submissionList)
    
    filename = 'submission_'+str(number+1)+'.csv'
    submissionFile_DataFrame.to_csv(filename, index=False)
     
    return submissionFile_DataFrame, filename, testMode


if testMode == True:

    ProductType = 'population'
    filename = 'submissionRawFile_2020_09_19.csv'
    resultFunction2 = SubmissionBuilder(ProductType,filename,testMode)

    print("=========================================")
    print("Product Type:")
    print("=========================================")
    print(ProductType)
    print("=========================================")
    print("Submission File saved as:")
    print("=========================================")
    print(resultFunction2[1])
    print("=========================================")
    print("Test result Function 2: Success")
    print("=========================================")


"""
=========================================
Function 3: Dataset builder (Stacking solution case) to process with ML models
=========================================

Purpose: Build an input dataset to be processed with an stacking solution

Raw code reference (see Tester.py): Test 15
"""

def stacking_Dataset_Builder(ProductType, PydicomMode, reportMode, testMode):

    # Set Product Type and its corresponding path
    
    if ProductType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if ProductType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if ProductType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    
    
    # Set working directory
    import os
    os.chdir(path_ProductType)
    
    # Get train dataset and test dataset
    import pandas as pd
    filename_trainDataset = 'train.csv'
    train_dataset = pd.read_csv(path_ProductType+filename_trainDataset)
    filename_testDataset = 'test.csv'
    test_dataset = pd.read_csv(path_ProductType+filename_testDataset)
    
    # Get submission dataset (template)
    import numpy as np
    path_resources = 'Y:/Kaggle_OSIC/3-Data (Prototype)/resources/'
    if (PydicomMode == False):
        filename_submissionDataset = 'submissionInputDataset.csv'
    else: 
        filename_submissionDataset = 'submissionInputDataset_pydicom.csv'
        
    submission_dataset = pd.read_csv(path_resources+filename_submissionDataset)
    submission_dataset = submission_dataset.replace(np.nan,'iNaN')
    
    # Adjust train dataset | Phase 1: Get ID list of the test dataset
    IDList = list(test_dataset.Patient)
    
    # Adjust train dataset | Phase 2: Get submission instances from train dataset
    instancesPopulation = len(train_dataset.Patient)
    indexList = []
    for i in IDList:
        for j in range(0,instancesPopulation):
            if i == train_dataset.Patient[j]:
                indexToInclude = train_dataset.index[j]
                indexList = indexList + [indexToInclude]
    
    # Adjust train dataset | Phase 3: Create an adjusted train dataset | a. Remove test instances from train dataset and reset index
    train_dataset_adjusted = train_dataset.drop(indexList)
    train_dataset_adjusted.reset_index
    
    # Adjust train dataset | Phase 3: Create an adjusted train dataset | b. Get Transferring data from train dataset
    instanceToTrasferList_index = []
    for k in range(0,instancesPopulation):
        for l in IDList:
            if train_dataset.Patient[k] == l:
                instanceToTransfer_Index = train_dataset.index[k]
                instanceToTrasferList_index = instanceToTrasferList_index + [instanceToTransfer_Index]
    
    train_dataset_instancesToTransfer = train_dataset.take(instanceToTrasferList_index)
    train_dataset_instancesToTransfer.index
    train_dataset_instancesToTransfer = train_dataset_instancesToTransfer.reset_index()
    train_dataset_instancesToTransfer.drop(columns='index')
    
    # Adjust train dataset | Phase 3: Create an adjusted train dataset | c. Update the submission dataset with the transferring data in b.
    submission_dataset_range = len(submission_dataset.Patient)
    train_dataset_instancesToTransfer_range = len(train_dataset_instancesToTransfer.Patient)
    
    Patient_List = []
    Week_List = []
    FVC_List = []
    Percent_List = []
    Age_List = []
    Sex_List = []
    SmokingStatus_List = []
    
    for m in range (0,submission_dataset_range):
        timesCopy = 0
        if(submission_dataset.Patient[m] in IDList):
            referenceWeek = submission_dataset.Weeks[m]
            for n in range (0,train_dataset_instancesToTransfer_range):
                if(train_dataset_instancesToTransfer.Patient[n] == submission_dataset.Patient[m] and train_dataset_instancesToTransfer.Weeks[n] == referenceWeek):
                    if (timesCopy == 0):
                        submission_dataset.FVC[m] = train_dataset_instancesToTransfer.FVC[n]
                        submission_dataset.Percent[m] = train_dataset_instancesToTransfer.Percent[n]
                        submission_dataset.Age[m] = train_dataset_instancesToTransfer.Age[n]
                        submission_dataset.Sex[m] = train_dataset_instancesToTransfer.Sex[n]
                        submission_dataset.SmokingStatus[m] = train_dataset_instancesToTransfer.SmokingStatus[n]
                        timesCopy = timesCopy + 1
                
                    else:
                    # Additional instances to include
                        Patient_List = Patient_List + [train_dataset_instancesToTransfer.Patient[n]]
                        Week_List = Week_List + [train_dataset_instancesToTransfer.Weeks[n]]
                        FVC_List = FVC_List + [train_dataset_instancesToTransfer.FVC[n]]
                        Percent_List = Percent_List + [train_dataset_instancesToTransfer.Percent[n]]
                        Age_List = Age_List + [train_dataset_instancesToTransfer.Age[n]]
                        Sex_List = Sex_List + [train_dataset_instancesToTransfer.Sex[n]]
                        SmokingStatus_List = SmokingStatus_List + [train_dataset_instancesToTransfer.SmokingStatus[n]]
    
    
    # Adjust train dataset | Phase 3: Create an adjusted train dataset | d. Add common values to submission dataset given those from the test dataset (Features: Age, Sex, SmokingStatus)
    submission_dataset_range = len(submission_dataset.Patient)
    for o in range(0,submission_dataset_range):
        if(submission_dataset.Patient[o] in IDList):
            for p in range(0,train_dataset_instancesToTransfer_range):
                if(submission_dataset.Patient[o] == train_dataset_instancesToTransfer.Patient[p]):
                    submission_dataset.Age[o] = train_dataset_instancesToTransfer.Age[p]
                    submission_dataset.Sex[o] = train_dataset_instancesToTransfer.Sex[p]
                    submission_dataset.SmokingStatus[o] = train_dataset_instancesToTransfer.SmokingStatus[p]
                    # Scenario to replace NaN values: Average FVC for a given Patient
                    averageFVC = train_dataset_instancesToTransfer.FVC[train_dataset_instancesToTransfer.Patient == train_dataset_instancesToTransfer.Patient[p]].mean()
                    submission_dataset.FVC[o] = averageFVC
            
    
    # Adjust train dataset | Phase 4: Create an adjusted train dataset | e. Concatenate the submission dataset (and additional instance) and the adjusted train dataset
    additionalDictionary = {submission_dataset.columns[0]:Patient_List,
                            submission_dataset.columns[1]:Week_List,
                                submission_dataset.columns[2]:FVC_List,
                                submission_dataset.columns[3]:Percent_List,
                                submission_dataset.columns[4]:Age_List,
                                submission_dataset.columns[5]:Sex_List,
                                submission_dataset.columns[6]:SmokingStatus_List}
    
    additional_dataset = pd.DataFrame(data=additionalDictionary)
    
    frames = [train_dataset_adjusted,submission_dataset,additional_dataset]
    
    train_dataset_adjusted = pd.concat(frames)
    train_dataset_adjusted = train_dataset_adjusted.reset_index()
    train_dataset_adjusted = train_dataset_adjusted.drop(columns='index')
    
    
    # Adjust train dataset with pydicom train dataset) | Phase 1: Get pydicom train dataset
    if(PydicomMode == True):
        filename_pydicom = 'train_pydicom.csv'
        path_ProductType_pydicom = path_ProductType + 'outcome/'
        train_dataset_pydicom = pd.read_csv(path_ProductType_pydicom + filename_pydicom)
                
    
    # Adjust train dataset with pydicom train dataset) | Phase 2: Include values from train_adjusted_pydicom.py into adjusted train dataset
    if(PydicomMode == True):
        
        instancesToInclude_List =  list(train_dataset_pydicom.Patient)
    
        InstanceToInclude_Patient = i
        
        newIndex = len(train_dataset_adjusted.Patient)
    
        for i in instancesToInclude_List:
        
           # Get instance to transfer
            
           InstanceToInclude_Patient = i
           InstanceToInclude_Week = list(train_dataset_pydicom[train_dataset_pydicom.Patient == i].Weeks)[0]
           InstanceToInclude_indexType1_Exhalation = list(train_dataset_pydicom[train_dataset_pydicom.Patient == i].indexType1_Exhalation)[0]
           InstanceToInclude_indexType1_Inhalation = list(train_dataset_pydicom[train_dataset_pydicom.Patient == i].indexType1_Inhalation)[0]
           InstanceToInclude_ImageType = list(train_dataset_pydicom[train_dataset_pydicom.Patient == i].ImageType)[0]
           
           # Put instance into train_dataset_adjusted DataFrame
           
           if (0 in list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].Weeks)):
               
                # Get index
                indexToComplete = list(train_dataset_adjusted[train_dataset_adjusted.Weeks == 0].Patient[train_dataset_adjusted.Patient == i].index)
                
                # Complete instance
                train_dataset_adjusted.indexType1_Exhalation[indexToComplete] = InstanceToInclude_indexType1_Exhalation
                train_dataset_adjusted.indexType1_Inhalation[indexToComplete] = InstanceToInclude_indexType1_Inhalation
                train_dataset_adjusted.ImageType[indexToComplete] = str(InstanceToInclude_ImageType)
               
           else:
               
                # Add new instance
               
                ## Get repeatable instances
                repeatableInstance1 = list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].FVC)[0]
                repeatableInstance2 = list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].Percent)[0]
                repeatableInstance3 = list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].Age)[0]
                repeatableInstance4 = list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].Sex)[0]
                repeatableInstance5 = list(train_dataset_adjusted[train_dataset_adjusted.Patient == i].SmokingStatus)[0]
               
                ## Get Dictionary 
                DictionaryToInclude = {}
                DictionaryToInclude['Patient'] = InstanceToInclude_Patient
                DictionaryToInclude['Weeks'] = InstanceToInclude_Week
                DictionaryToInclude['FVC'] = repeatableInstance1
                DictionaryToInclude['Percent'] = repeatableInstance2
                DictionaryToInclude['Age'] = repeatableInstance3
                DictionaryToInclude['Sex'] = repeatableInstance4
                DictionaryToInclude['SmokingStatus'] = repeatableInstance5
                DictionaryToInclude['indexType1_Exhalation'] = InstanceToInclude_indexType1_Exhalation
                DictionaryToInclude['indexType1_Inhalation'] = InstanceToInclude_indexType1_Inhalation
                DictionaryToInclude['ImageType'] = str(InstanceToInclude_ImageType)
               
                ## Get DataFrame
                
                DataFrameToInclude = pd.DataFrame(data = DictionaryToInclude, index=[newIndex])
                newIndex = newIndex + 1
               
                ## Concatenate DataFrame
                train_dataset_adjusted = pd.concat([train_dataset_adjusted, DataFrameToInclude])
    
    # nan filling 
    train_dataset_adjusted = train_dataset_adjusted.replace('iNaN',np.nan)
    
    # Specifying dtype 
    train_dataset_adjusted.astype({'Patient': 'O'}).dtypes
    train_dataset_adjusted.astype({'Weeks': 'float64'}).dtypes
    train_dataset_adjusted.astype({'Percent': 'float64'}).dtypes
    train_dataset_adjusted.astype({'Age': 'float64'}).dtypes
    train_dataset_adjusted.astype({'Sex': 'O'}).dtypes
    train_dataset_adjusted.astype({'SmokingStatus': 'O'}).dtypes
    train_dataset_adjusted.astype({'FVC': 'float64'}).dtypes
    
    if(PydicomMode == True):
        train_dataset_adjusted.astype({'indexType1_Exhalation': 'float64'}).dtypes
        train_dataset_adjusted.astype({'indexType1_Inhalation': 'float64'}).dtypes
        train_dataset_adjusted.astype({'ImageType': 'O'}).dtypes
        
    
    # Get CSV file
    path_output = path_ProductType +'outcome/'
    if(PydicomMode == False):
        filename_output = 'train_adjusted.csv'
    else:
        filename_output = 'train_adjusted_pydicom.csv'
    train_dataset_adjusted.to_csv(path_output+filename_output)
    
    # Function Result
    resultFunction = train_dataset_adjusted,path_output,filename_output
    
    # Report Mode
    
    if reportMode == True:
        print("=========================================")
        print("Function Report")
        print("=========================================")
        print("DataFrame")
        print("=========================================")
        print(resultFunction[0])
        print("=========================================")
        print("Product Type: ", ProductType)
        print("=========================================")
        print("Pydicom Mode: ", PydicomMode)
        print("=========================================")
        print("Location of Input File:", resultFunction[1])
        print("=========================================")
        print("Input File saved as:", resultFunction[2])
        print("=========================================")
        print("Data type of the dataset")
        print("=========================================")
        print(resultFunction[0].dtypes)
        print("=========================================")
        print("Test result Function 3: Success")
        print("=========================================")
    
    return resultFunction


if testMode == True:

    ProductType = 'prototype'
    PydicomMode = True
    reportMode = False
    resultFunction3 = stacking_Dataset_Builder(ProductType, PydicomMode, reportMode, testMode)

    print("=========================================")
    print("Function Report")
    print("=========================================")
    print("DataFrame")
    print("=========================================")
    print(resultFunction3[0])
    print("=========================================")
    print("=========================================")
    print("Product Type: ", ProductType)
    print("=========================================")
    print("Pydicom Mode: ", PydicomMode)
    print("=========================================")
    print("Location of Input File:", resultFunction3[1])
    print("=========================================")
    print("Input File saved as:", resultFunction3[2])
    print("=========================================")
    print("Data type of the dataset")
    print("=========================================")
    print(resultFunction3[0].dtypes)
    print("=========================================")
    print("Test result Function 3: Success")
    print("=========================================")
    

"""
=========================================
Function 4: Submission dataset builder (Stacking solution case) after ML outcome
=========================================

Purpose: Build a submission CSV file (Stacking solution case)

Raw code reference (see Tester.py): Test 17

About the Shape Parameter: It amounts to c = 0.12607421874999922 for every instance in the oject of concern. c value has been computed
deeming the following data fitting scope: (1) Data: FVC predictions; (2) Probability density function as follows (staistical function
in scipy renowend as scipy.stats.loglaplace): loglaplace.pdf(x, c, loc=0, scale=1).
"""

def Stacking_Submission_Dataset_Builder(ProductType,shapeParameter_DataFrame,pydicomMode,testMode):
    
    # Set Product Type and its corresponding path
    if ProductType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if ProductType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if ProductType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    
    
    # Set working directory
    import os
    os.chdir(path_ProductType + 'outcome/')
    
    # Get result data and test dataset
    import pandas as pd
    if(pydicomMode == True):
        filename_resultDataset = 'result_pydicom.csv'
    else:
        filename_resultDataset = 'result.csv'
    
    result_dataset = pd.read_csv(path_ProductType+'outcome/'+filename_resultDataset)
    filename_testDataset = 'test.csv'
    test_dataset = pd.read_csv(path_ProductType+filename_testDataset)
    
    # Get submission instances | Phase 1: Index
    IDList = list(test_dataset.Patient)
    IDList_index_dictionary = {}
    for i in IDList:
        itemToInclude = result_dataset.Patient[result_dataset.Patient==i].index
        IDList_index_dictionary[i] = itemToInclude
    
    # Get submission instances | Phase 2: Extract submission instances from result dataset
    IDList_index = []
    IDList_columns = ['Patient', 'Weeks', 'Random Forest', 'Lasso', 'Gradient Boosting', 'Stacking Regressor']
    for j in IDList: IDList_index = IDList_index + list(IDList_index_dictionary[j])
    
    submission_dataset = result_dataset.loc[IDList_index]
    
    # Get submission instances | Phase 3: Extract duplicated instances
    submission_dataset = submission_dataset.drop_duplicates(subset=['Patient','Weeks'])
    
    # Get submission instances | Phase 4: Sort submission instances by Weeks (ascending) and reset index
    submission_dataset = submission_dataset.sort_values(by=['Weeks','Patient'])
    
    submission_dataset = submission_dataset.reset_index()
    submission_dataset = submission_dataset.drop(columns=['Unnamed: 0','index'])
    
    # Get confidence measure | Phase 1: Get shape Parameter DataFrame by default
    ## When shapeParameter_DataFrame==[], parameter c = 0.126074 is assigned by default per model and ID
    if (shapeParameter_DataFrame == []):
        shapeParameter_dictionary = {}
        shapeParameter = 0.126074
        MLModelList = IDList_columns[2:]
        
        for l in MLModelList:
            keyShapeParameter = 'c Parameter_'+l
            shapeParameter_dictionary[keyShapeParameter] = [shapeParameter,shapeParameter,shapeParameter,shapeParameter,shapeParameter]
        
        shapeParameter_DataFrame = pd.DataFrame(data = shapeParameter_dictionary, index = IDList)
        
    # Get confidence measure | Phase 2: Get standard-deviation-clipped per instance
    ## Metric - Part 1: standard_deviation_clipped = max(standard_deviation, 70)
    ## Build a DataFrame with standard-deviation-clipped values given an ID and a ML Model: standardDeviationClipped_DataFrame
    standardDeviationClipped_DataFrame = shapeParameter_DataFrame.copy()
    columnLabels = list(standardDeviationClipped_DataFrame.columns)
    columnLabels_SDC_dictionary = {}
    
    for i in columnLabels:
        columnLabels_item ='SD_Clipped'+i[11:]
        columnLabels_SDC_dictionary[i]=columnLabels_item
    
    standardDeviationClipped_DataFrame = standardDeviationClipped_DataFrame.rename(columns=columnLabels_SDC_dictionary)
    import numpy as np
    standardDeviationClipped_DataFrame = standardDeviationClipped_DataFrame.replace(3,np.nan)
    
    ID_List = list(standardDeviationClipped_DataFrame.index)
    SDModel_List = list(standardDeviationClipped_DataFrame.columns)
    CParameter_List = list(shapeParameter_DataFrame.columns)
    numy = 0
        
    from scipy.stats import loglaplace
    
    for j in ID_List:
        
        for k in SDModel_List:
            itemToInclude = CParameter_List[numy]
            c = shapeParameter_DataFrame[itemToInclude][j]
            sd_LL = loglaplace.std(c, loc=0, scale=100)
            standardDeviationClipped_DataFrame[k][j] = max(70,sd_LL) # j: index is ID | k: SD_Clipped_(ML Model)
            numy = numy + 1
        
        numy = 0
           
    # Get confidence measure | Phase 3: Get metric axe per model: |FVC_true - FVC_predicted|
    ## Metric - Part 1: |FVC_true - FVC_pred|
    
    if(pydicomMode == True):
        variableNumber = 10
    else:
        variableNumber = 7
        
    MLModelList = list(submission_dataset.columns[variableNumber:])
    metric_dictionary = {}
    
    for j in MLModelList:
        metric_differential = abs(submission_dataset.FVC - submission_dataset[j])
        metric_differential = list(metric_differential)
        keyToInclude = 'metric_'+j
        metric_dictionary[keyToInclude] = metric_differential
        metric_DataFrame = pd.DataFrame(data=metric_dictionary)
    
    # Get confidence measure | Phase 4: Get metric axe per model: min(|FVC_true - FVC_predicted|, 1000)
    ## metric per instance
    ## Metric - Part 2: min(|FVC_true - FVC_pred|,1000)
    
    metricLabels = list(metric_DataFrame.columns)
    instancesNumber = len(submission_dataset.index)
    
    for i in metricLabels:
        j = 0
        while (j<instancesNumber):
            metric_DataFrame[i][j] = min(metric_DataFrame[i][j],1000)
            j = j+1

    submission_dataset = submission_dataset.join(metric_DataFrame)
        
    # Get confidence measure | Phase 5: Get metric axe per model: (-1 * differential * 2^0.5 / SDC ) - ln(2^0.5 * SCD)
    ## metric per instance
    ## differential = min(|FVC_true - FVC_predicted|, 1000)
    ## SDC: Standard Deviation Clipped
    ## Metric - Part 2: min(|FVC_true - FVC_pred|,1000)
    
    IDList = list(test_dataset.Patient)
    
    SDModel_List = list(standardDeviationClipped_DataFrame.columns)
    SDModel_index_List = list(standardDeviationClipped_DataFrame.index)
        
    metric_lists = list(metric_DataFrame.columns)
    metric_index_lists = list(metric_DataFrame.index)
    
    submission_dataset_index_List = list(submission_dataset.index)
    
    instancesNumber = len(submission_dataset_index_List)
    
    indexPerID_dictionary = {}
    
    ### Step 1: Get index per ID to compute
        
    for i in IDList:
        listToInclude = list(submission_dataset.Patient[submission_dataset.Patient == i].index)
        indexPerID_dictionary[i] = listToInclude
        
    indexPerID_DataFrame = pd.DataFrame(data=indexPerID_dictionary)
    
    ### Step 3: Compute metric
    import math
    from math import log1p  
    
    for k in IDList:
        for i in metric_lists:
            for j in list(indexPerID_DataFrame[k]):
                
                differential = submission_dataset[i][j]
                
                SDC_Label = 'SD_Clipped_' + i[7:]
                SDC = standardDeviationClipped_DataFrame[SDC_Label][k]
                
                metric_part1 = -1* 2**0.5 * differential / SDC
                metric_part2 = -1 * math.log1p(2**0.5 * SDC)
                metric = metric_part1 + metric_part2                
                
                submission_dataset[i][j] = metric
            
    # Result function specification
    resultFunction = submission_dataset,shapeParameter_DataFrame,standardDeviationClipped_DataFrame
    
    
    # Get submission files | Phase 1: Get submission file template
    filename = 'sample_submission.csv'
    submissionFile = pd.read_csv(path_ProductType+filename)
    
    ## Get submission files | Phase 2: Create directory
    try:
        path_output = path_ProductType + 'submission/'
        os.chdir(path_output)
    except FileNotFoundError:
        import distutils.ccompiler
        path_output = path_ProductType + 'submission/'
        distutils.dir_util.mkpath(path_output)
        
    ## Get submission files | Phase 3: Get correlative
    files_list = os.listdir(path_output)
    try: 
        maxNumber = max(files_list)
        maxNumber = maxNumber[:-4]
        maxNumber = int(maxNumber)
        nextNumber = maxNumber+1
    except ValueError:
        nextNumber = 0
        
    ## Get submission files | Phase 4: Get models to include and their corresponding metrics
    ModelToInclude = IDList_columns[2:]
        
    ## Get submission files | Phase 5: Build Files
    for i in ModelToInclude:
        filename = 'sample_submission.csv'
        submissionFile = pd.read_csv(path_ProductType+filename)
        submissionFile_columns = list(submissionFile.columns)
        
        fvc_array = np.array(submission_dataset[i])
        confidence_array = np.array(submission_dataset['metric_'+i])
        
        submissionFile['FVC'] = fvc_array
        submissionFile['Confidence'] = confidence_array
        
        filename_output = str(nextNumber)+'.csv'
        path_output = path_ProductType +'submission/'
        submissionFile.to_csv(path_output+filename_output,columns=submissionFile_columns,index=False)
        nextNumber = nextNumber + 1
        
    return resultFunction

if testMode == True:

    # Set Product type 
    ProductType = 'prototype'
    
    # ShapeParameter_Dataframe
    example = False
    if (example == True):
        import pandas as pd
        shapeParameter_IDList = ['ID00419637202311204720264','ID00421637202311550012437','ID00422637202311677017371','ID00423637202312137826377','ID00426637202313170790466']
        c_List1 = [3,3,3,3,3]
        c_List2 = [3,3,3,3,3]
        c_List3 = [3,3,3,3,3]
        c_List4 = [3,3,3,3,3]
        shapeParameter_dictionary = {'Random Forest':c_List1, 'Lasso':c_List2, 'Gradient Boosting':c_List3, 'Stacking Regressor':c_List4}
        shapeParameter_DataFrame = pd.DataFrame(data = shapeParameter_dictionary, index = shapeParameter_IDList)
    else:
        shapeParameter_DataFrame = []
    
    # Set Pydicom mode
    pydicomMode = True
    
    resultFunction4 = Stacking_Submission_Dataset_Builder(ProductType,shapeParameter_DataFrame,pydicomMode,testMode)

    print("=========================================")
    print("Shape Parameter -  Laplace Log Likelihood:")
    print("=========================================")
    print(resultFunction4[1])
    print("Standard Deviation Clipped -  Laplace Log Likelihood:")
    print("=========================================")
    print(resultFunction4[2])
    print("=========================================")
    print("Test result Function 4: Success")
    print("=========================================")


"""
=========================================
Function 5: Get parameters given a must-usage of a log-laplace distribution (i.e. Laplace Log Likelihood)
=========================================

Purpose: Get shape parameter visualization for loglaplace

Raw code reference (see Tester.py): Test 17

"""

def shapeParameter_visualizer(ProductType,testMode):
    
    import numpy as np
    from scipy.stats import loglaplace
    import matplotlib.pyplot as plt
    fig, ax =  plt.subplots(4, 5, sharex=False, sharey=False, figsize=(32, 24))
    
    ## Get IDs to test
    import os
    import pandas as pd
    
    ## Set Product Type and its corresponding path
    if ProductType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if ProductType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if ProductType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
        
    ## Get probabilities from predicted values grouping by ID and Model 
    path = path_ProductType + 'outcome/'
    filename = 'result.csv'
    y_pred = pd.read_csv(path+filename)

    ## Get IDs to test
    path = path_ProductType
    filename = 'test.csv'
    test_dataset = pd.read_csv(path+filename)
    ID_List = list(test_dataset.Patient)

    ## Get models
    model_List = ['Random Forest', 'Lasso', 'Gradient Boosting', 'Stacking Regressor']
    
    ## Grouping task
    
    k = 0
    l = 0
    
    for i in ID_List:
        
        k = 0
        
        for j in model_List:
            
            # Data Fit task
            #r = y_pred[y_pred.Patient==i][j]/sum(y_pred[y_pred.Patient==i][j])
            r = y_pred[y_pred.Patient==i][j]
            r = np.array(r)
            
            c1, loc1, scale1 = loglaplace.fit(r,floc=0,fscale=1)
            c = c1
            
            # # Calculate a few first moments                       
            # mean, var, skew, kurt = loglaplace.stats(c, moments='mvsk')
            
            # Display the probability density function (pdf):
            x = np.linspace(loglaplace.ppf(0.01, c), loglaplace.ppf(0.99, c), num=100)
            ax[k,l].plot(x, loglaplace.pdf(x, c),'r-', lw=5, alpha=0.6, label='loglaplace pdf')
    
            # Freeze the distribution and display the frozen pdf:
            rv = loglaplace(c)
            ax[k,l].plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
                        
            # Generate random numbers:
            r = loglaplace.rvs(c1, loc=0, scale=1, size=1000)

            # And compare the histogram:
            #ax[k,l].hist(r, density=True, histtype='stepfilled', alpha=0.2)
            ax[k,l].legend(loc='best', frameon=False)
            
            # Set limits
            #ax[k,l].set_xlim(0,0.1)
            #ax[k,l].set_ylim(0,4)
            ax[k,l].set_xlabel('x')
            ax[k,l].set_ylabel('f(x,c)')

            # Check Accuracy
            vals = loglaplace.ppf([0.001, 0.5, 0.999], c)
            accuracy = np.allclose([0.001, 0.5, 0.999], loglaplace.cdf(vals, c))
            # Returns True if two arrays are element-wise equal within a tolerance.
            if(accuracy == True):
                accuracy = 'Equal case'
            else:
                accuracy = 'Unequal case'
            
            # Set title               
            title = str('Probability density function for loglaplace'+'\n'+i + '\n' + j + ' | Accuracy:'+accuracy)
            ax[k,l].set_title(title)

            k = k + 1
        
        l = l + 1
    
    plt.tight_layout()
    plt.show()
    
    resultFunction = c
    
    return resultFunction

if testMode == True:

    # Set Product type 
    ProductType = 'prototype'
    
    # ShapeParameter_Dataframe
    resultFunction5 = shapeParameter_visualizer(ProductType, testMode = True)

    print("=========================================")
    print("Shape Parameter -  Laplace Log Likelihood:")
    print("=========================================")
    print(resultFunction5)
    print("=========================================")
    print("Test result Function 4: Success")
    print("=========================================")

    
# """
# =========================================
# Function : Dataset builder 2 (Stacking solution case) to process with ML models 
# =========================================

# Purpose: Build an input dataset to be processed with an stacking solution but including Pydicom image-processing solution

# Raw code reference (see Tester.py): 15
# """

# def stacking_Dataset_Builder_PydicomSolution(productType, testMode):

#     # Set Product Type and its corresponding path
    
#     if ProductType == 'population':
#         path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
#     if ProductType == 'prototype':
#         path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
#     if ProductType == 'sampling':
#         path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    
    