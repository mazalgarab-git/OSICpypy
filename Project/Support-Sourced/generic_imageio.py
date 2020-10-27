# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:23:29 2020

@author: mazal
"""

"""
=========================================
Image Processing
=========================================
Purpose: To process images using imageio

Process: 
    Phase 1: Get images into RGB dataframes
    Phase 2 : Get space index that depicts FVC status
    Phase 3: Put index into Kaggle dataset to process through a Machine Learning solution
       
Space index prototype guidelines

(1) Image concept

    space = Summation of grey spacing due to a given scale (grey scales based on RGB)
        
(2) Assumptions
    
    Assumption 1: Fibrotic change is depicted by opacity when visualizing using HRCT images in the view of Nakamura and Aoshiba (2016).
       
        
        max FVC -> maximum spacing (grey - tenuous case) | minimum spacing (grey - dark case)
        FVC -> spacing 
        min FVC -> minimum spacing (grey - tenuous case) | maximum spacing (grey - dark case)
       
    Assumption 2: Sampling type of the images is deemed a baseline at time Week=0 (OSIC Pulmonary Fibrosis Progression, 2020)
    
(3) Index prototypes

index protoype 1 | Balance assumption
- Basics: Ratio is expected to remain the same (or at least, to trend) in regular conditions).
- Quick Definition: Average ratio = tenousGrey / darkGrey
- Hyphotehesis 1: the index is related to the feature percent given the technology purpose; percent is defined by Kaggle (OSIC Pulmonary Fibrosis Progression, 2020).
- Assumption: The certitude of the hyphotesis 1 is taken for granted.
    
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
Function 1: Get images into RGB dataframes
=========================================
Purpose: Get jgp images into RGB dataframes

Raw code reference: -

Sources: References within the code

"""

def RGB_DataFrame_Maker(productType,splitType,ID,interpolationMethod,testMode):

    
    # Conditionning | Phase -1: Set root based on: (1) ProductType; (2) splitType; (3) ID; (4) Interpoltion method
    
    if productType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if productType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if productType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Sampling)/'


    if splitType == 'test':
        if(interpolationMethod == None):interpolationMethod = 'None'
        image_path_input = path_ProductType + 'outcome/' + 'jpgImagesInterpolation/' + 'test/' + ID + '/'+ interpolationMethod + '/'
        image_path_output = path_ProductType + 'outcome/' + 'jpgImageProcessing/' + 'test/' + ID + '/' + interpolationMethod + '/'
    else:
        if(interpolationMethod == None):interpolationMethod = 'None'
        image_path_input = path_ProductType + 'outcome/' + 'jpgImagesInterpolation/' + 'train/' + ID + '/' + interpolationMethod + '/'
        image_path_output = path_ProductType + 'outcome/' + 'jpgImageProcessing/' + 'train/' + ID + '/' + interpolationMethod + '/'  
    

    # Conditionning | Phase 0: Import common libraries
    import os
    import pandas as pd
    import numpy as np
    import imageio
    import cv2
    from distutils.dir_util import mkpath
    
    # Conditionning | Phase 1: Get list of images to process given the ID
    imageList = os.listdir(image_path_input)
    
    # Conditionning | Phase 2: Verification of completion.
    ## Processing required: input files number different from output files
    ## Processing not required: input files number equal to output files
    
    try:
        filesNumberToProcess = os.listdir(image_path_input)
        filesNumberProcessed = os.listdir(image_path_output)
        if (filesNumberToProcess != filesNumberProcessed):
            for i in filesNumberProcessed:
                filename = i
                os.remove(image_path_output + filename)
            processing = 'required'
        else:
            processing = 'not required'
    
    except FileNotFoundError:
        mkpath(image_path_output) # Conditionning | Phase 3: Create output directory
        processing = 'required'
            
    # RGB DataFrame Maker
    
    if (processing == 'required'):
    
        for j in imageList:
        
        ## RGB DataFrame Maker | Phase 1: Read Image JPG
    
            ## Reading an image file | (Rashka & Mirjalili, 2019, pp.532-536)
            ### Change1: Insert variable 'filename' for the filename in the form 'example-image.png'.
            ### Change2: Built into Function reading_image_raw(filename)       
            filename = j
            imageInView = imageio.imread(image_path_input+filename)
            
            if(testMode == True):
                print('Image shape:', imageInView.shape)
                print('Number of channels:', imageInView.shape[2])
                print('Image data type:', imageInView.dtype)
                print(imageInView[100:102, 100:102, :])
            
        ## RGB DataFrame Maker | Phase 2: Normalize
            
            ## Normalizing image | Convert numpy.ndarray into imageio.core.util.Image | (Rothman et al., 2018, pp. 470-479)
            ## OpenCV | Smoothing Images | Image Blurring | (OpenCV, n.d.)
            ## OpenCV | Geometric Transformations of Images | Scaling (OpenCV, n.d.)
            ## OpenCV | Geometric Image Transformations | Image procecssing | resize()  | (OpenCV, n.d.)
            ## From 512x512 pixels to 100x100 pixels | Adobe criteria to size (Adobe,n.d.) 
            imageInView_resize = cv2.resize(imageInView, (100, 100), interpolation = cv2.INTER_AREA) 
            imageInView_array = cv2.GaussianBlur(imageInView_resize, (5, 5), 0)
            
        ## RGB DataFrame Maker | Phase 3: Get DataFrame
        
            si,sj,sk = np.shape(imageInView_array)
            numy = 0
            
            for i in range(0,si):
                
                if (i == 0):
                    colNames = [str(numy)] + [str(numy+1)] + [str(numy+2)]
                    numy = numy + 3
                    df_left = pd.DataFrame(imageInView_array[i], columns = colNames)
                
                else:
                    colNames = [str(numy)] + [str(numy+1)] + [str(numy+2)]
                    numy = numy + 3
                    df_right = pd.DataFrame(imageInView_array[i], columns = colNames) 
                    df_left = pd.merge(df_left,df_right,left_index=True, right_index=True)
        
        ## RGB DataFrame Maker | Phase 4: Build CSV file
            path_output = image_path_output
            filename_output = j[:-4] + '.csv'
            imageInView_DataFrame = df_left
            imageInView_DataFrame.to_csv(path_output+filename_output)

    else:
        None
    
    if (processing == 'not required'):None

    ## Closing | Set Function Result
    functionResult = imageList
    
    ## Closing | Report
    if(testMode == False):
        print("=========================================")
        print("Function Report")
        print("=========================================")
        print("Inputs")
        print("  Product type: ",productType)
        print("  Split type: ",splitType)
        print("  Patient ID: ",ID)
        print("  Interpolation Method: ",interpolationMethod)
        print("=========================================")
        print("Outputs")
        print("  Number of processed images", len(filesNumberToProcess))
        print("=========================================")
        print("Test result Function 1: Success")
        print("=========================================")  
        
    return functionResult


if testMode == True:

    ProductType = 'prototype'
    #ProductType = 'population'
    splitType = 'train'
    #ID = 'ID00007637202177411956430' # Image type 1
    #ID = 'ID00011637202177653955184'
    ID = 'ID00009637202177434476278' # Image type 2
    #ID = 'ID00014637202177757139317' # Image type 3
    #ID = 'ID00009637202177434476278'
    interpolationMethod = None
    
    FunctionResult1 = RGB_DataFrame_Maker(ProductType,splitType,ID,interpolationMethod,testMode)
    
    
## Print results if reportMode = True
    
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
    print("Outputs")
    print("  Number of processed images", len(FunctionResult1))
    print("=========================================")
    print("Test result Function 1: Success")
    print("=========================================")    


"""
=========================================
Function 2: Compute space index that depicts FVC status | Index type 1
=========================================
Purpose: Get space index

Raw code reference: 

Sources: 

"""

# Define Scale (Quartiles)

# Measure Spacing

# Compute Index



"""
=========================================
Reference
=========================================

Adobe. n.d. Resize and Crop images in Photoshop and Photoshop Elements. Retrieved from
    https://www.adobe.com/support/techdocs/331327.html  

Nakamura, H & Aoshiba, K. (e.d.). (2016). Idiophatic Pulmonary Fibrosis - 
    Advances in Diagnostic Tools and Disease Management. Chapter 6
    High-Resolution Computed Tomography of Honeycombing and IPF/UIP (pp. 82-84).
    Tokyo, Japan: Springer Japan.
    
OpenCV. n.d. Geometric Image Transformations. Retrieved from
    https://docs.opencv.org/trunk/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d

OSIC Pulmonary Fibrosis Progression. (2020). Available at https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data

Raschka, S., & Mirjalili, V. (2019). Python Machine Learning: Machine Learning
    and Deep Learning with Python, scikit-learn, and TensorFlow 2 (3rd ed.).
    Birmingham, United Kingdom: Packt Publishing Ltd.
    
Rothman, D., Lamons, M., Kumar, R., Nagaraja, A., Ziai, A., & Dixit, A. (2018).
    Python: Beginner's Guide to Artificial Intelligence: Build applications to
    intelligently interact with the world around you using Python (1st ed.).
    Birmingham, United Kingdom: Packt Publishing Ltd.
"""
