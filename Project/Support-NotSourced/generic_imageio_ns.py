# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:31:02 2020

@author: mazal
"""

"""
=========================================
Support functions of imageio
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
testMode = True

"""
=========================================
Function 1: Compute space index that depicts FVC status | Index type 1
=========================================
Purpose: Get space index

Code Source: The index prototype is available at generic_imageio.py

Raw code reference (see Tester.py): Test 18

"""

def IndexType1(productType,splitType,ID,interpolationMethod,testMode,reportMode):
    
    # Conditionning | Phase -1: Set root based on: (1) ProductType; (2) splitType; (3) ID; (4) Interpoltion method
    
    import os
    import pandas as pd
    import numpy as np
    
    if productType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if productType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if productType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/'


    if splitType == 'test':
        if(interpolationMethod == None):interpolationMethod = 'None'
        imageioDataFrame_path_input = path_ProductType + 'outcome/' + 'jpgImageProcessing/' + 'test/' + ID + '/'+ interpolationMethod + '/'
    else:
        if(interpolationMethod == None):interpolationMethod = 'None'
        imageioDataFrame_path_input = path_ProductType + 'outcome/' + 'jpgImageProcessing/' + 'train/' + ID + '/' + interpolationMethod + '/'
    
    
    # Conditionning | Phase -1: Set input data: (1) path input ;(2) filename list. 
    path_input = imageioDataFrame_path_input
    os.chdir(path_input)
    fileNameList = os.listdir(path_input)    
    
    # Computing | Index Type 1 

    indexTriad_List = []

    for i in fileNameList:
        
        filename = i

        # Get CSV file
        rawFile_DataFrame = pd.read_csv(filename)
        rawFile_DataFrame = rawFile_DataFrame.drop(columns=['Unnamed: 0'])
        
        # Get triad list and DataFrame
        triadNumberElements_index = list(rawFile_DataFrame.index)
        triadNumberElements = list(rawFile_DataFrame.columns)
        triadNumberElements_range = int(len(list(triadNumberElements))/3)
        
        triad_list = []
        
        for i in triadNumberElements_index:
            
            constant_growth = 3
            variable_growth = 0 #start value
        
            for j in range(0,triadNumberElements_range):
        
                # Column-j series
                
                ## cg:constant growth (+3)
                ## cv:variable growth (+1 iteration)
                
                ## j:0 ## j=0 + cg=3 * cv=0 -> 0 
                ## j:0 ## j=1 + cg=3 * cv=0 -> 1
                ## j:0 ## j=2 + cg=3 * cv=0 -> 2
                
                ## j:1 ## j=0 + cg=3 * cv=1 -> 3 
                ## j:1 ## j=1 + cg=3 * cv=1 -> 4
                ## j:1 ## j=2 + cg=3 * cv=1 -> 5
        
                ## j:2 ## j=0 + cg=3 * cv=2 -> 6 
                ## j:2 ## j=1 + cg=3 * cv=2 -> 7
                ## j:2 ## j=2 + cg=3 * cv=2 -> 8
                
                ## ...
                ## ...
                ## ...
                
                ## j:99 ## j=0 + cg=3 * cv=99 -> 297 
                ## j:99 ## j=1 + cg=3 * cv=99 -> 298
                ## j:99 ## j=2 + cg=3 * cv=99 -> 299
                
                variable_growth = j
                
                x_value = 0 + constant_growth * variable_growth
                y_value = 1 + constant_growth * variable_growth
                z_value = 2 + constant_growth * variable_growth
                
                x_col = str(x_value)
                y_col = str(y_value)
                z_col = str(z_value)
                
                triadIndex = int(i)
                
                x_triad = rawFile_DataFrame[x_col][triadIndex]
                y_triad = rawFile_DataFrame[y_col][triadIndex]
                z_triad = rawFile_DataFrame[z_col][triadIndex]
                
                triad = np.array([x_triad,y_triad,z_triad])
                #print(triad)
                triad_list = triad_list + [triad]        
        
        ####triad_dictionary = {'triad':triad_list}
        ####triad_DataFrame = pd.DataFrame(data=triad_dictionary)
        
        # Define RGB Scale boundaries based on a grey scale
        ## RGB | Grey Scale |
        ### (1) 255,255,255
        ### (2) 224,224,224
        ### (3) 192,192,192
        ### (4) 160,160,160
        ### (5) 128,128,128
        ### (6) 96,96,96
        ### (7) 64,64,64
        ### (8) 32,32,32
        ### (9) 0,0,0
        
        boundary9 = np.array([255,255,255])
        boundary8 = np.array([224,224,224])
        boundary7 = np.array([192,192,192])
        boundary6 = np.array([160,160,160])
        boundary5 = np.array([128,128,128])
        boundary4 = np.array([96,96,96])
        boundary3 = np.array([64,64,64])
        boundary2 = np.array([32,32,32])
        boundary1 = np.array([0,0,0])
        
        GreyScaleBoundaries = [boundary1,boundary2,boundary3,boundary4,boundary5,boundary6,boundary7,boundary8,boundary9]
        
        # Get new frequency group by RGB scale boundaries
        frequencyRGB_Dictionary = {}
        
        for i in triad_list:
            
            for j in GreyScaleBoundaries:
            
            # print("Iteration: ")
            # print("Triads to evaluate")    
            # print("Triads of concern: ",i)
            # print("Triad Boundary: ",j)
            
                if max(i) >= max(j):
                    
                    None
                
                else:
                    
                    key_frequencyRGB_Dictionary_list = list(frequencyRGB_Dictionary.keys())
                    keyToValidate = str(list(j))
                    keyOfConcern = str(list(j))
                    
                    
                    if(keyToValidate in key_frequencyRGB_Dictionary_list):
                        
                        frequencyRGB_Dictionary[keyOfConcern] = [frequencyRGB_Dictionary[keyOfConcern][0] + 1]
                        break
                    
                    else:
        
                        frequencyRGB_Dictionary[keyOfConcern] = [1]
                        break
        
        frequencyRGB_DataFrame = pd.DataFrame(data = frequencyRGB_Dictionary)
        
        # Image typing | Phase (1): Get frequency DataFrame group by triad
        frequency_dictionary = {}
        #### indexList_frequency_DataFrame = []
        #### indexNumber = 0
        
        for i in triad_list:
            
            keyToValidate  = str(list(i))
            keyList = list(frequency_dictionary.keys())
            
            if(keyToValidate not in keyList):
                array_frequency_dictionary = i
                key_frequency_dictionary = str(list(array_frequency_dictionary))
                frequency_dictionary[key_frequency_dictionary] = [1]
            else:
                keyToInclude = keyToValidate
                array_frequency_dictionary = i
                key_frequency_dictionary = str(list(array_frequency_dictionary))
                newFrequency = frequency_dictionary[keyToInclude][0] + 1
                frequency_dictionary[keyToInclude] = [newFrequency]
        
        #### frequency_DataFrame = pd.DataFrame(data = frequency_dictionary)
        
        # Image typing | Phase (2): Image typing identifier
        ## Types 2: 'ID00007637202177411956430' | blackTriadFrequency = 1896 -> 2000
        ## Types 3: 'ID00009637202177434476278'| 5075 -> 5500
        ## Types 1: 'ID00014637202177757139317', 'ID00419637202311204720264' | 457 -> 1000
        ## Criterion: black color spacing - RGB triad: frequency below RGB triad (32,32,32)
        
        try:
            triadFrequencyToEvaluate = frequencyRGB_DataFrame['[32, 32, 32]'][0]
        except KeyError:
            triadFrequencyToEvaluate = 0
        
        imageType1 = 1000
        imageType2 = 2000
        imageType3 = 5500
        
        imageTypeBoundaries = [imageType1, imageType2, imageType3]
        imageTypeBoundaries_label = ['1','2','3']
        
        comparisonResults = []
        
        for i in imageTypeBoundaries:
            
            if(triadFrequencyToEvaluate >= i):
                comparisonResults = comparisonResults + [True]
            else:
                comparisonResults = comparisonResults + [False]
        
        iteration = 0
        for i in comparisonResults:
            
            if i is False:
                imageType =  imageTypeBoundaries_label[iteration]
                break
            else:
                iteration = iteration + 1
        
        if comparisonResults == [True, True, True]: imageType =  imageTypeBoundaries_label[2]
        
        # Image typing | Phase (3): Differ state Inhalation / Exhalation and get index
        
        ## Step 0: Set column Label list
        #### labelList = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]','[128, 128, 128]','[96, 96, 96]','[64, 64, 64]','[32, 32, 32]']
        
        ## Step 1: Get the state given the type of image (i.e. inhalation or exhalation)
        
        if (imageType == '1'):    
            # Set RGB triads to differ 'inhalation' from 'exhalation'
            ## RGB triads to build a pattern: ['[224, 224, 224]','[192, 192, 192]']
            ## Boundary | Exhalation: Summation(RGB triads) <= 7000 (6715)
            ## Boundary | Inhalation: 8000 (7840) <= Summation(RGB triads)
            labelList_imageState = ['[224, 224, 224]','[192, 192, 192]']
            labelList_exhalation_dark = ['[160, 160, 160]','[128, 128, 128]','[96, 96, 96]','[64, 64, 64]','[32, 32, 32]']
            labelList_exhalation_tenous = ['[224, 224, 224]','[192, 192, 192]']
            labelList_inhalation_dark = ['[64, 64, 64]','[32, 32, 32]']
            labelList_inhalation_tenous = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]','[128, 128, 128]','[96, 96, 96]']
            stateBoundaryExhalation = 7000
        
        if (imageType == '2'):    
            # Set RGB triads to differ 'inhalation' from 'exhalation'
            ## RGB triads to build a pattern: ['[224, 224, 224]','[192, 192, 192]']
            ## Boundary | Exhalation: Summation(RGB triads) <= 3500 (3215)
            ## Boundary | Inhalation: 5000 (5195) <= Summation(RGB triads)
            labelList_imageState = ['[224, 224, 224]','[192, 192, 192]']
            labelList_exhalation_dark = ['[128, 128, 128]','[96, 96, 96]','[64, 64, 64]','[32, 32, 32]']
            labelList_exhalation_tenous = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]']
            labelList_inhalation_dark = ['[64, 64, 64]','[32, 32, 32]']
            labelList_inhalation_tenous = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]','[128, 128, 128]','[96, 96, 96]']
            stateBoundaryExhalation = 3500
        
        if (imageType == '3'):
            # Set RGB triads to differ 'inhalation' from 'exhalation'
            ## RGB triads to build a pattern: ['[128, 128, 128]','[96, 96, 96]']
            ## Boundary | Exhalation: Summation(RGB triads) <= 3500 (3426)
            ## Boundary | Inhalation: 6500 (6720) <= Summation(RGB triads) 
            labelList_imageState = ['[128, 128, 128]','[96, 96, 96]']
            labelList_exhalation_dark = ['[32, 32, 32]']
            labelList_exhalation_tenous = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]','[128, 128, 128]','[96, 96, 96]','[64, 64, 64]']
            labelList_inhalation_dark = ['[32, 32, 32]']
            labelList_inhalation_tenous = ['[224, 224, 224]','[192, 192, 192]','[160, 160, 160]','[128, 128, 128]','[96, 96, 96]','[64, 64, 64]']
            stateBoundaryExhalation = 3500
        
        ## Step 2: Get ciphers to compute the index
        
        stateTriads_List = []
        for i in labelList_imageState:
            try:
                elementToInclude = frequencyRGB_DataFrame[i][0]
                stateTriads_List = stateTriads_List + [elementToInclude]
            except KeyError:
                stateTriads_List = stateTriads_List + [0]
        
        if sum(stateTriads_List) <= stateBoundaryExhalation:
            imageState = 'Exhalation'
            DarkGreyCipherLabel = labelList_exhalation_dark
            TenousGreyCipherLabel = labelList_exhalation_tenous
        else:
            imageState = 'Inhalation'
            DarkGreyCipherLabel = labelList_inhalation_dark
            TenousGreyCipherLabel = labelList_inhalation_tenous
        
        DarkGreyCipher_list = []
        for i in DarkGreyCipherLabel:
            try:
                elementToInclude = frequencyRGB_DataFrame[i][0]
                DarkGreyCipher_list = DarkGreyCipher_list + [elementToInclude]
            except KeyError:
                DarkGreyCipher_list = DarkGreyCipher_list + [0]
                
        TenousGreyCipher_list = []
        for i in TenousGreyCipherLabel:
            try:
                elementToInclude = frequencyRGB_DataFrame[i][0]
                TenousGreyCipher_list = TenousGreyCipher_list + [elementToInclude] 
            except KeyError:
                DarkGreyCipher_list = DarkGreyCipher_list + [0]
        
        # Image typing | Phase (4):  Get index value
        numerator = sum(TenousGreyCipher_list)
        denominator = sum(DarkGreyCipher_list)
        index = round(numerator / denominator,6)
        
        ## Step 7: Get index triad
        indexTriad_List = indexTriad_List + [[ID, index, imageState, imageType]]
        
    # Computing | Index Type 1 | Build DataFarme using indexTriad_List
    indexTriad_Array = np.array(indexTriad_List)
    indexTriad_DataFrame = pd.DataFrame(indexTriad_Array,columns=['Patient','indexType1','ImageState','ImageType'])

    # Computing | Index Type 1 | Get average values for indexType1 regarding the ImageState
    exhalationValues = []
    inhalationValues = []
    for i in indexTriad_DataFrame.index:
        itemToInclude = float(indexTriad_DataFrame.indexType1[i])
        if(indexTriad_DataFrame.ImageState[i] == 'Exhalation'): exhalationValues = exhalationValues + [itemToInclude]
        if(indexTriad_DataFrame.ImageState[i] == 'Inhalation'): inhalationValues = inhalationValues + [itemToInclude]
    
    exhalationAverage = round(np.mean(exhalationValues),6)
    inhalationAverage = round(np.mean(inhalationValues),6)
    
    exhalationTriad = np.array([ID,exhalationAverage,'Exhalation',indexTriad_DataFrame.ImageType[0]])
    inhalationTriad = np.array([ID,inhalationAverage,'Inhalation',indexTriad_DataFrame.ImageType[0]])
    
    # Closing | Get Function Result
    import time
    processTime = time.process_time() 
    FunctionResult = indexTriad_DataFrame,exhalationTriad,inhalationTriad,processTime

    if reportMode == True:
    
        print("=========================================")
        print("Function Report")
        print("=========================================")
        print("Inputs")
        print("  Product type: ",ProductType)
        print("  Split type: ",splitType)
        print("  Patient ID: ",ID)
        print("  Interpolation Method: ",interpolationMethod)
        print("=========================================")
        print("Process")
        print("  Time: ", FunctionResult[3])
        print("=========================================")
        print("Outputs")
        print("  Number of DataFrames: ", 1)
        print("  Number of processed images", len(FunctionResult[0]))
        print("  Exhalation triad: ",FunctionResult[1][1:])
        print("  Inhalation triad: ",FunctionResult[2][1:])
        print("=========================================")
        print("Test result Function 1: Success")
        print("=========================================")

    return FunctionResult


if testMode == True:

    #ProductType = 'prototype'
    ProductType = 'population'
    splitType = 'train'
    #ID = 'ID00007637202177411956430' # Image type 1
    #ID = 'ID00011637202177653955184'
    #ID = 'ID00009637202177434476278' # Image type 2
    #ID = 'ID00014637202177757139317' # Image type 3
    #ID = 'ID00009637202177434476278'
    #ID = 'ID00134637202223873059688'
    ID ='ID00135637202224630271439'
    interpolationMethod = None
    reportMode = True
    
    FunctionResult1 = IndexType1(ProductType,splitType,ID,interpolationMethod,testMode,reportMode)
    

"""
=========================================
Function 2: Compute space index that depicts FVC status | Index type 1 (Scaling mode)
=========================================
Purpose: Get space index under scaling mode

Code Source: -

Raw code reference (see Tester.py): -

"""

def GetIndexType1_ScalingMode(productType,interpolationMethod,testMode,reportMode):
    
    import os
    import pandas as pd
    import numpy as np
    import distutils.ccompiler
    
    # Create datasets | Phase -1: Get output path
    
    if productType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if productType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if productType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/'
    
    path_output_train = path_ProductType + 'outcome/' + 'indexType1/train/' 
    path_output_test = path_ProductType + 'outcome/' + 'indexType1/test/'
    path_output = path_ProductType + 'outcome/'
    
    try:
        os.chdir(path_output_train)
    except FileNotFoundError:
        distutils.dir_util.mkpath(path_output_train)
        
    try:
        os.chdir(path_output_test)
    except FileNotFoundError:
        distutils.dir_util.mkpath(path_output_test)
    
    # Conditionning | Phase 0: Get ID lists
    
    ## Train and test paths
    train_path = 'Y:/Kaggle_OSIC/2-Data/train/'
    test_path = 'Y:/Kaggle_OSIC/2-Data/test/'
    
    ## Train and test lists
    trainList = os.listdir(train_path)
    testList = os.listdir(test_path)
    
    ## Set parameters
    
    testMode = False
    reportMode = True
    
    # Create datasets | Phase 1: Test dataset
    tetrad_test_List = []
    
    try:
        tetrad_test_List_output = os.listdir(path_output_test)
    except FileNotFoundError:
        tetrad_test_List_output = []
    
    splitType = 'test'
    
    for i in testList:
        
        try:
            # Verify if process is done
            filenameToVerify = i + '.csv'
            if (filenameToVerify in tetrad_test_List_output): 
                print("=========================================")
                print(splitType," ",filenameToVerify, " -> Done")
                print("=========================================")
            
            else:
                # Process
                ID = i
                FunctionResult = IndexType1(ProductType,splitType,ID,interpolationMethod,testMode,reportMode)
                exhalationTetrad = list(FunctionResult[1])
                inhalationTetrad = list(FunctionResult[2])
                tetrad_test_List = [exhalationTetrad] + [inhalationTetrad]
                tetrad_test_array = np.array(tetrad_test_List)
                
                # Get unitary DataFrame
                indexType1_test_DataFrame_unitary = pd.DataFrame(tetrad_test_array, columns=['Patient','indexType1','ImageState','ImageType']) 
                # Create CSV File
                filename_test_output_unitary = i + '.csv'
                indexType1_test_DataFrame_unitary.to_csv(path_output_test+filename_test_output_unitary)
        
        except FileNotFoundError:
            break
    
    # Create datasets | Phase 2: Train dataset
    tetrad_train_List = []
    
    try:
        tetrad_train_List_output = os.listdir(path_output_train)
    except FileNotFoundError:
        tetrad_train_List_output = []
    
    splitType = 'train'
    
    for j in trainList:

        try:
            # Verify if process is done
            filenameToVerify = j + '.csv'
            if (filenameToVerify in tetrad_train_List_output):
                print("=========================================")
                print(splitType," ",filenameToVerify, " -> Done")
                print("=========================================")

            else:            
                # Process
                ID = j
                FunctionResult = IndexType1(ProductType,splitType,ID,interpolationMethod,testMode,reportMode)
                exhalationTetrad = list(FunctionResult[1])
                inhalationTetrad = list(FunctionResult[2])
                tetrad_train_List = [exhalationTetrad] + [inhalationTetrad]
                tetrad_train_array = np.array(tetrad_train_List)
                
                # Get unitary DataFrame
                indexType1_train_DataFrame_unitary = pd.DataFrame(tetrad_train_array, columns=['Patient','indexType1','ImageState','ImageType']) 
                # Create CSV File
                filename_train_output_unitary = j + '.csv'
                indexType1_train_DataFrame_unitary.to_csv(path_output_train+filename_train_output_unitary)
            
        except FileNotFoundError:
            break
    
    # Create datasets | Phase 2: Create DataFarme and CSV file
    
    ## Train dataset case
    
    unitary_Dataframe = []
    trainList_output = os.listdir(path_output_train)
   
    if (len(trainList) == len(trainList_output)):
        unitary_Dataframe = []
    
        for i in trainList_output:
            filename = i
            if(unitary_Dataframe != []):
                unitary_DataframeToInclude = pd.read_csv(path_output_train+filename)
                frames = [unitary_Dataframe,unitary_DataframeToInclude]
                unitary_Dataframe = pd.concat(frames)
            else:
                unitary_Dataframe = pd.read_csv(path_output_train+filename)
        
        indexType1_train_DataFrame = unitary_Dataframe
        
    filename_train_output = 'indexType1_train.csv'
    indexType1_train_DataFrame.to_csv(path_output+filename_train_output)

    ## Test dataset case

    unitary_Dataframe = []
    testList_output = os.listdir(path_output_test)
   
    if (len(testList) == len(testList_output)):
        for i in testList_output:
            filename = i
            if(unitary_Dataframe != []):
                unitary_DataframeToInclude = pd.read_csv(path_output_test+filename)
                frames = [unitary_Dataframe,unitary_DataframeToInclude]
                unitary_Dataframe = pd.concat(frames)
            else:
                unitary_Dataframe = pd.read_csv(path_output_test+filename)
        
        indexType1_test_DataFrame = unitary_Dataframe
        
    filename_test_output = 'indexType1_test.csv'
    indexType1_test_DataFrame.to_csv(path_output+filename_test_output)
    
    # Closing | Time
    import time
    processTime = time.process_time() 

    # Closing | Function Result
    FunctionResult = indexType1_test_DataFrame, indexType1_train_DataFrame, processTime
    
    # Closing | Report
    
    if reportMode == True:
    
        print("=========================================")
        print("Function Report")
        print("=========================================")
        print("Inputs")
        print("  Product type: ",ProductType)
        print("  Split type: ",splitType)
        print("  Patient ID list - train: ",trainList)
        print("  Patient ID list - test: ",testList)
        print("  Interpolation Method: ",interpolationMethod)
        print("=========================================")
        print("Process")
        print("  Time: ", FunctionResult[2])
        print("=========================================")
        print("Outputs")
        print("  Output path: ", path_output)
        print("  output file - train: ", filename_train_output)
        print("  output file - test: ", filename_test_output)
        print("=========================================")
        print("Test result Function 2: Success")
        print("=========================================")
    
    return FunctionResult
    

if testMode == True:

    #ProductType = 'prototype'
    ProductType = 'population'
    interpolationMethod = None
    reportMode = True
    
    FunctionResult2 = GetIndexType1_ScalingMode(ProductType,interpolationMethod,testMode,reportMode)    
