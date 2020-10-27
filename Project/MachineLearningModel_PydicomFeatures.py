# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:38:10 2020

@author: mazal
"""

"""
=========================================
Machine learning Models
=========================================
Purpose: Manage machine learning models
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
Function 1: Dataset Builder
=========================================
Purpose: Create a dataset available to be processed using machine learning model(s)

Raw code reference: Test 11

Sourced from Combine predictors using stacking. n.d.
"""

def datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode):
    
    # Dataset type: 'Ames housing dataset'
    ## Sourced from Combine predictors using stacking. n.d.
    
    if(datasetType == 'Ames housing dataset'):
    
        import numpy as np
        from sklearn.datasets import fetch_openml
        from sklearn.utils import shuffle
        
        df = fetch_openml(name="house_prices", as_frame=True)
        X = df.data
        y = df.target
    
        features = ['YrSold', 'HeatingQC', 'Street', 'YearRemodAdd', 'Heating',
                    'MasVnrType', 'BsmtUnfSF', 'Foundation', 'MasVnrArea',
                    'MSSubClass', 'ExterQual', 'Condition2', 'GarageCars',
                    'GarageType', 'OverallQual', 'TotalBsmtSF', 'BsmtFinSF1',
                    'HouseStyle', 'MiscFeature', 'MoSold']
    
        X = X[features]
        X, y = shuffle(X, y, random_state=0)
    
        X = X[:600]
        y = y[:600]
        
        resultFunction = X, np.log(y)
    
    
    if(datasetType == 'pydicom competition - prototype'):
       
       # Set working directory
       import os
       train_path = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
       os.chdir(train_path)
       
       # Set train file name
       import pandas as pd
       train_file = 'train.csv'
              
       # Get train data and target
       train_data = pd.read_csv(train_path+train_file,usecols=['Patient','Weeks','Percent','Age','Sex','SmokingStatus'],na_values=None)
       train_target = pd.read_csv(train_path+train_file,usecols=['FVC'],na_values=None)
       X = train_data
       y = train_target
       
       # Get results
       import numpy as np
       #resultFunction = X, np.log(y)
       resultFunction = X, y
    
    
    if(datasetType == 'pydicom competition'):
       
       # Set Product Type path
       if ProductType == 'population':path_ProductType = 'Y:/Kaggle_OSIC/2-Data/outcome/'
       if ProductType == 'prototype':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/outcome/'
       if ProductType == 'sampling':path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/outcome/'
       
       # Set working directory
       import os
       train_path = path_ProductType
       os.chdir(train_path)
       
       # Set train file name
       import pandas as pd
       
       try:
           train_file = 'train_adjusted.csv'
       except FileNotFoundError:
        os.chdir('Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/')
        from generic_pydicom_ns import stacking_Dataset_Builder
        #stacking_Dataset_Builder(ProductType, testMode=False)
        stacking_Dataset_Builder(ProductType, PydicomMode, reportMode, testMode)
        
        train_path = path_ProductType
        os.chdir(train_path)
        train_file = 'train_adjusted.csv'
           
       # Get train data and target
       train_data = pd.read_csv(train_path+train_file,usecols=['Patient','Weeks','Percent','Age','Sex','SmokingStatus'])
       train_target = pd.read_csv(train_path+train_file,usecols=['FVC'])
       X = train_data
       y = train_target
       
       # dtype specifying
       X.astype({'Patient': 'O'}).dtypes
       X.astype({'Weeks': 'float64'}).dtypes
       X.astype({'Percent': 'float64'}).dtypes
       X.astype({'Age': 'float64'}).dtypes
       X.astype({'Sex': 'O'}).dtypes
       X.astype({'SmokingStatus': 'O'}).dtypes
       y.astype({'FVC': 'float64'}).dtypes
       
       # Get results
       import numpy as np
       #resultFunction = X, np.log(y)
       resultFunction = X, y
    
    if(datasetType == 'pydicom competition - Image processing'):
       
       # Set Product Type path
       if ProductType == 'population':path_ProductType = 'Y:/Kaggle_OSIC/2-Data/outcome/'
       if ProductType == 'prototype':path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/outcome/'
       if ProductType == 'sampling':path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/outcome/'
       
       # Set working directory
       import os
       train_path = path_ProductType
       os.chdir(train_path)
       
       # Set train file name
       import pandas as pd
       
       try:
           train_file = 'train_adjusted_pydicom.csv'
       except FileNotFoundError:
        os.chdir('Y:/Kaggle_OSIC/OSICpypy/Support-NotSourced/')
        from generic_pydicom_ns import stacking_Dataset_Builder
        stacking_Dataset_Builder(ProductType, PydicomMode, reportMode, testMode)
        
        train_path = path_ProductType
        os.chdir(train_path)
        train_file = 'train_adjusted_pydicom.csv'
           
       # Get train data and target
       train_data = pd.read_csv(train_path+train_file,usecols = ['Patient','Weeks','Percent','Age','Sex','SmokingStatus', 'indexType1_Exhalation','indexType1_Inhalation','ImageType'])
       train_target = pd.read_csv(train_path+train_file,usecols = ['FVC'])
       X = train_data
       y = train_target
       
       # dtype specifying
       X.astype({'Patient': 'O'}).dtypes
       X.astype({'Weeks': 'float64'}).dtypes
       X.astype({'Percent': 'float64'}).dtypes
       X.astype({'Age': 'float64'}).dtypes
       X.astype({'Sex': 'O'}).dtypes
       X.astype({'SmokingStatus': 'O'}).dtypes
       X.astype({'indexType1_Exhalation': 'float64'}).dtypes
       X.astype({'indexType1_Inhalation': 'float64'}).dtypes
       X.astype({'ImageType': 'O'}).dtypes
       y.astype({'FVC': 'float64'}).dtypes
       
       # Get results
       import numpy as np
       #resultFunction = X, np.log(y)
       resultFunction = X, y
    
    if(reportMode == True):

        print("=========================================")
        print("Report | Function 2: Dataset Builder")
        print("=========================================")
        print("Dataset Type: ",datasetType)
        print("=========================================")
        print("Dataset | Part 1: Features (X)")
        print("=========================================")
        print(resultFunction[0])
        print("=========================================")
        print("Dataset | Part 2: Target variable (y)")
        print("=========================================")
        print(resultFunction[1])
        print("=========================================")
        print("Data type of the dataset | Features")
        print(resultFunction[0].dtypes)
        print("=========================================")
        print("Data type of the dataset | Target")
        print(resultFunction[1].dtypes)
        print("=========================================")
        print("Test result Function 2: Success")
        print("=========================================")
    
    return resultFunction, testMode


if(testMode == True):

    # Set dataset type
    #datasetType = 'Ames housing dataset'
    #datasetType = 'pydicom competition - prototype'
    
    ProductType = 'population'
    datasetType = 'pydicom competition - Image processing'
    #datasetType = 'pydicom competition'
    PydicomMode = True
    reportMode = True
    
    # Run function
    resultFunction2 = datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode)
    

"""
=========================================
Function 3: Pipeline maker to preprocess the data
=========================================
Purpose: Preprocess a dataset available to be processed using machine learning model(s) by building a pipeline

Raw code reference: Test 12

Sourced from Combine predictors using stacking. n.d.

Input: Output of result function 2 (i.e. Pair X, y )
"""

def pipelineMaker_MLPreprocessor(X,y,reportMode,testMode):
    
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import StandardScaler
    
    
    cat_cols = X.columns[X.dtypes == 'O']
    num_cols = X.columns[X.dtypes == 'float64']
    
    categories = [
        X[column].unique() for column in X[cat_cols]]
    
    for cat in categories:
        cat[cat == None] = 'missing'  # noqa
    
    cat_proc_nlin = make_pipeline(
        SimpleImputer(missing_values=None, strategy='constant',
                      fill_value='missing'),
        OrdinalEncoder(categories=categories)
        )
    
    num_proc_nlin = make_pipeline(SimpleImputer(strategy='mean'))
    
    cat_proc_lin = make_pipeline(
        SimpleImputer(missing_values=None,
                      strategy='constant',
                      fill_value='missing'),
        OneHotEncoder(categories=categories)
    )
    
    num_proc_lin = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )
    
    # transformation to use for non-linear estimators
    processor_nlin = make_column_transformer(
        (cat_proc_nlin, cat_cols),
        (num_proc_nlin, num_cols),
        remainder='passthrough')
    
    # transformation to use for linear estimators
    processor_lin = make_column_transformer(
        (cat_proc_lin, cat_cols),
        (num_proc_lin, num_cols),
        remainder='passthrough')
    
    resultFunction = [processor_nlin] + [processor_lin]
    
    if(reportMode == True):

        print("=========================================")
        print("Report | Function 3: Pipeline maker to preprocess the data")
        print("=========================================")
        print("=========================================")
        print("(1) Pipeline for non-linear estimators: processor_nlin")
        print("=========================================")
        print(resultFunction[0])
        print("=========================================")
        print("(2) Pipeline for linear estimators: processor_lin")
        print("=========================================")
        print(resultFunction[1])
        print("=========================================")
        print("Test result Function 3: Success")
        print("=========================================")
    
    return resultFunction, testMode


if(testMode == True):

    # Must Input | Dataset: 'Ames housing dataset' | 'pydicom competition - prototype' | 'pydicom competition'
    
    # Set Product type
    #ProductType = 'prototype'
    ProductType = 'population'
    
    ## Set dataset type
    #datasetType = 'Ames housing dataset'
    #datasetType = 'pydicom competition - prototype'
    #datasetType = 'pydicom competition'
    datasetType = 'pydicom competition - Image processing'
    PydicomMode = True
    reportMode = True

    ## Run Function 2 "Dataset builder"
    resultFunction2 = datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode = False)
    
    ## Set input pair X,y
    X = resultFunction2[0][0]
    y = resultFunction2[0][1]
    
    # Run Function 3
    reportMode = True
    resultFunction3 = pipelineMaker_MLPreprocessor(X,y,reportMode,testMode)


"""
=========================================
Function 4: Stack of predictors on a single data set
=========================================
Purpose: Get stack of predictors on a single data set

Raw code reference: Test 13

Sourced from Combine predictors using stacking. n.d.

Input: 
    
    Output of result function 2 (i.e. Pair X, y )
    Output of result function 3 (i.e. Pair processor_lin, processor_nlin)

"""

def stackingSolution(processor_lin,processor_nlin,reportMode,testMode):

    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import RidgeCV
    
    
    lasso_pipeline = make_pipeline(processor_lin,
                                    LassoCV())
    
    rf_pipeline = make_pipeline(processor_nlin,
                                RandomForestRegressor(random_state=42))
    
    gradient_pipeline = make_pipeline(
        processor_lin, #modify from processor_nlin to processor_lin
        HistGradientBoostingRegressor(random_state=0))
    
    estimators = [('Random Forest', rf_pipeline),
                  ('Lasso', lasso_pipeline),
                  ('Gradient Boosting', gradient_pipeline)]
    
    stacking_regressor = StackingRegressor(estimators=estimators,
                                            final_estimator=RidgeCV())
    
    resultFunction = stacking_regressor,estimators
    
    if(reportMode == True):

        print("=========================================")
        print("Report | Function 4: Stack of predictors on a single data set")
        print("=========================================")
        print("=========================================")
        print("Final Estimator (stacking_regressor.finale_estimator)")
        print("=========================================")
        print(resultFunction[0].final_estimator)
        print("=========================================")
        print("Estimators: pipelines")
        print("=========================================")
        i = 0
        while (i<100):
            try:
                print(resultFunction[1][i][0])
                i = i+1
            except IndexError:
                break
        print("=========================================")
        print("Test result Function 4: Success")
        print("=========================================")
    
    return resultFunction, testMode


if testMode == True:

    # Must Input | Dataset: 'Ames housing dataset' | 'pydicom competition - prototype' | 'pydicom competition'

    ## Set Product Type
    ProductType = 'prototype'

    ## Set dataset type
    #datasetType = 'Ames housing dataset'
    #datasetType = 'pydicom competition - prototype'
    #datasetType = 'pydicom competition'
    datasetType = 'pydicom competition - Image processing'

    ## Run Function 2 "Dataset builder"
    PydicomMode = False
    reportMode = False
    resultFunction2 = datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode=False)
    
    ## Set input pair X,y
    X = resultFunction2[0][0]
    y = resultFunction2[0][1]
    
    # Must Input 2 | Get processors (output Function 3 pipelineMaker_MLPreprocessor)
    reportMode = False
    resultFunction3 = pipelineMaker_MLPreprocessor(X,y,reportMode,testMode=False)
    processor_lin = resultFunction3[0][0]
    processor_nlin = resultFunction3[0][1]
    
    # Run Function 4 stackingSolution
    reportMode = True
    resultFunction4 = stackingSolution(processor_lin,processor_nlin,reportMode,testMode=False)

    

"""
=========================================
Function 5: Measure and Plot the results
=========================================
Purpose: Get measures and plotting result.

Raw code reference: Test 14

Sourced from Combine predictors using stacking. n.d.

Input: 
    
    Output of result function 2 (i.e. Pair X, y )
    Output of result function 3 (i.e. Pair processor_lin, processor_nlin)
    Output of result function 4 (i.e. Pair stacking regressor object. estimators)

"""
    
def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):

    print("=========================================")
    #print(ax)
    #print(y_true)
    #print(y_pred)
    print(title)
    print(scores)
    print(elapsed_time)
    print("=========================================")

    import matplotlib.pyplot as plt

    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],[y_true.min(), y_true.max()],'--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)
    
    return
            

def getResults(X,y,estimators,stacking_regressor,ProductType,pydicomMode,reportMode,testMode):
    
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_validate, cross_val_predict
    from sklearn.model_selection import RepeatedKFold

    plt.close('all')
    fig, axs = plt.subplots(2, 2, figsize=(18, 14)) #fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)
    
    functionResult_ax = []
    functionResult_score = []
    functionResult_elapsed_time = []
    functionResult_y_pred = []
    
    for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', stacking_regressor)]):
        start_time = time.time()
        
        score = cross_validate(est, X, y,
                                scoring=['r2', 'neg_mean_absolute_error'],
                                n_jobs=-1, verbose=0)
        
        elapsed_time = time.time() - start_time
    
        try:
            y_pred = cross_val_predict(est, X, y, cv=None, n_jobs=-1, verbose=0)
        except TypeError:
            try:
                #y_pred = cross_val_predict(est, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=5), return_estimator=True,n_jobs=-1, verbose=0)
                #y_pred = cross_val_predict(est, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=5),return_estimator=False,n_jobs=-1, verbose=0)
                #y_pred = cross_val_predict(est, X, y, cv=3, return_estimator=True,n_jobs=-1, verbose=0)
                y_pred = cross_val_predict(est, X, y, cv=3, n_jobs=-1, verbose=0)
                #y_pred = cross_val_predict(est, X, y, cv=15, return_estimator=True,n_jobs=-1, verbose=0)
            except TypeError: 
                y_pred = []
        
        functionResult_ax = functionResult_ax + [ax]
        functionResult_score = functionResult_score + [score]
        functionResult_elapsed_time = functionResult_elapsed_time + [elapsed_time]
        functionResult_y_pred = functionResult_y_pred + [y_pred]
    
        # Run Function plot_regression_results

        if(y_pred!=[]):

            try:
                plot_regression_results(
        ax, y, y_pred,
        name,
        (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
        .format(np.mean(score['test_r2']),
                np.std(score['test_r2']),
                -np.mean(score['test_neg_mean_absolute_error']),
                np.std(score['test_neg_mean_absolute_error'])),
        elapsed_time)
                
            except ValueError:
                print()
                #print("ValueError",ax)
            
    if(y_pred!=[]):
        plt.suptitle('Single predictors versus stacked predictors')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    #Get result dataset | Phase 1: Build y_pred Dataset
    i = 0
    y_pred_dictionary = {} 
    
    while(i>=0):
        try:
            keyToInclude = estimators[i][0]
            y_pred_dictionary[keyToInclude] = list(functionResult_y_pred[i])
        except IndexError:
            keyToInclude = 'Stacking Regressor'
            y_pred_dictionary[keyToInclude] = list(functionResult_y_pred[i])
            break
        
        i = i + 1
    
    import pandas as pd
    y_pred_Dataset = pd.DataFrame(data = y_pred_dictionary)
    
    #Get result dataset | Phase 2: Build output file y_pred
    result_dataset = X.merge(y,left_index=True,right_index=True)
    result_dataset = result_dataset.merge(y_pred_Dataset,left_index=True,right_index=True)
    print(result_dataset)
    
    # Get result dataset | Phase 3: Set Product type
    if ProductType == 'population':
        path_ProductType = 'Y:/Kaggle_OSIC/2-Data/'
    
    if ProductType == 'prototype':
        path_ProductType = 'Y:/Kaggle_OSIC/3-Data (Prototype)/'
    
    if ProductType == 'sampling':
        path_ProductType = 'Y:/Kaggle_OSIC/4-Data (Sampling)/'
    
    # Get result dataset | Phase 4: Get CSV file
    path_output = path_ProductType +'outcome/'
    if(pydicomMode == True):
        filename_output = 'result_pydicom.csv'
    else:
        filename_output = 'result.csv'
        
    result_dataset.to_csv(path_output+filename_output)
    
    # Function result
    
    functionResult = functionResult_ax,functionResult_score,functionResult_elapsed_time,functionResult_y_pred,result_dataset
    
    if (reportMode == True):

        # Run Function 5.1 "GetResults" and Function 5.2 plot_regression_results
        ax = functionResult[0]
        score = functionResult[1]
        elapsed_time = functionResult[2]
        y_pred = functionResult[3]
        result_dataset = functionResult[4]        

        print("=========================================")
        print("Report | Function 5: Measure and Plot the results")
        print("=========================================")
        print("=========================================")
        print("Number of Prediciton sets: ", len(y_pred))
        print("=========================================")
        print("Number of instances per prediciton set: ", len(y_pred[0]))
        print("=========================================")
        scorerList = list(score[0].keys())
        print("Scorers: ", len(scorerList))
        for i in scorerList:print("  ",i)
        print("=========================================")
    
    return functionResult, testMode


if testMode == True:

    # Must Input | Dataset: 'Ames housing dataset' | 'pydicom competition - prototype' | 'pydicom competition'
    
    # Set Product type
    ProductType = 'prototype'

    ## Set dataset type
    #datasetType = 'Ames housing dataset'
    #datasetType = 'pydicom competition - prototype'
    #datasetType = 'pydicom competition'
    datasetType = 'pydicom competition - Image processing'

    ## Run Function 2 "Dataset builder"
    PydicomMode = False
    reportMode = False
    resultFunction2 = datasetBuilder_ML(datasetType, ProductType, PydicomMode, reportMode, testMode=False)
    
    ## Set input pair X,y
    X = resultFunction2[0][0]
    y = resultFunction2[0][1]
    
    # Must Input 2 | Get processors (output Function 3 pipelineMaker_MLPreprocessor)
    reportMode = False
    resultFunction3 = pipelineMaker_MLPreprocessor(X,y,reportMode,testMode=False)
    processor_lin = resultFunction3[0][0]
    processor_nlin = resultFunction3[0][1]
    
    # Must Input 3 | Get stacking regressor object and estimators
    reportMode = False
    resultFunction4 = stackingSolution(processor_lin,processor_nlin,reportMode,testMode=False)
    stacking_regressor = resultFunction4[0][0]
    estimators = resultFunction4[0][1]
    
    # Run Function 5.1 "GetResults" and Function 5.2 plot_regression_results
    reportMode = True
    pydicomMode = True
    functionResult5 = getResults(X,y,estimators,stacking_regressor,ProductType,pydicomMode,reportMode,testMode=False)

"""
=========================================
Reference
=========================================

Classifier comparison. n.d. Available at https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

Combine predictors using stacking. n.d. Available at https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py

"""
