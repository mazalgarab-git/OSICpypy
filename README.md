# OSICpypy
Python project as a proposal for the Kaggle competition OSIC Pulmonary Fibrosis Progression available at https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

## Core ideas to get the proposed solution

1. Image processing: Dicom images are processed using library imageio and build index based on train data; see the file `Project/Support-Sourced/generic_imageio.py`
2. Predicting: Prediction is based on a stacking solution fostered by scikit-learn known as Combine predictors using stacking (see https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py), that scikit-learn solution has been adequated to predict; see the file `Project/MLModel/MachineLearningModel_PydicomFeatures.py`

### Structure of the project

[Diagram powered by Lucidchart](https://lucid.app/publicSegments/view/e2e5459f-ec5b-402d-b229-996608c53945/image.png)

### Activity diagram

[Diagram powered by Lucidchart](https://lucid.app/publicSegments/view/c03a61ce-5d1c-4e7f-9fcd-f62e444996c1/image.png)

## Conditionning and Installation guide

1. Conditionning | Phase 1: Get the repository by commiting `git clone https://github.com/mazalgarab-git/OSICpypy.git`
2. Conditionning | Phase 2: Get libraries by commiting `pip install -r requirements.txt`
3. Conditionning | Phase 3: Build a root `Y:/Kaggle_OSIC/`
4. Conditionning | Phase 4: Get basic structure (i.e. tree-directory structure to start based on Data/TreeStructure_To_start.txt) by unzipping `Data/TreeStructure_To_start.rar` into `Y:/`
5. Conditionning | Phase 5: Get Kaggle data (i.e. pydicom files on test and train directories) from `https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data`
6. Conditionning | Phase 6: Put Pydicom data as follows: (1) `train dataset` to `Y:/Kaggle_OSIC//2-Data/train/`; (2) `test dataset` to `Y:/Kaggle_OSIC/2-Data/test/`
7. Installation | Phase 1: Put files on `Project/` into `Y:/Kaggle_OSIC/OSICpypy/`
