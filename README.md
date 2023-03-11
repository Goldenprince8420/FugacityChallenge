# **Fugacity Challenge**
- Author: Rahul Golder
- Roll No: 19CH30042
- Department: Chemical Engineering:
- Year: 4th
- Problem Statement Page: [Link](https://unstop.com/hackathons/chemical-intelligence-ml-challenge-fugacity23-iit-kharagpur-634871)

<img src = "https://d8it4huxumps7.cloudfront.net/uploads/images/festival/logo/150x150/61efda27cab0d_fugacity_logo.jpeg?d=250x250" width="200" height="100" style="horizontal-align:middle" >


*Several Instruction Before Running this Notebook:*

1.   This Notebook should be run in [Colab](https://colab.research.google.com/drive/18O4RjkXr0iF9LjhfxGUYZf-jzIByHkvO?usp=sharing) or Jupyter Notebook
2.   The requirements file has been created considering these basic modules are installed:
   - numpy
   - pandas
   - matplotlib
   - sklearn
   - torch
   - scipy
   - tensorflow
   - seaborn
   - ipython
3. At the end of running the notebook the submission file should be present in the `./data` folder.
4. Internet should be present(For seeing results of explanation module). Absence doesn't hamper training process

For each section instructions has been provided here:

_________________________________

## **Setup**

- Importing the Working Repository for the code. Here's the [link](https://github.com/Goldenprince8420/FugacityChallenge.git)
- installing secondary but necessary depecdencies, library and modules

_____________________
## **Imports**
- Importing all necessary modules and libraries
- Importing the codes from the repository for further analysis
- Importing the data

*Note*:

1. The repository has the folowing files:
    - main.py: for just definition
    - data.py: Contains functions for importing various forms of data, preprocessing of data, test dataset.
    - models.py: Contains code for running different models on the training data.
    - explainability.py: Contains code for explainable models for understanding the reasons of predictions
    - utils.py: Contains various utility functions like visualization and preprocessing
    - preprocessing.py: Contains function for feature engineering and standardization.
    - explainability.py: Contains functions for explainable models like LIME and SHAP to interpret the predictions
    - optimization.py: Contains function for bayesian optimization

____________________
## **Exploration**

- Preprocessing of the imported data has been done here
- The training and validation set has been created for additional analysis
- Statistical analysis has been performed
- Distribution of all features has been provided


### **Preprocessing and Feature Engineering**
- Some new coefficients has been defined to increase richness of the data. These coefficients has been defined after detailed literature survey to find approximate relation between Relative humidity and other features like Temperature and Average Humidity
- Temperature coeff: `100 * exp((7.5 * T)) / (237.7 + T)`
- Average Humidity Coeff: `AH * 461.5 * T / 100`

________________________
## **Model Application**

- Various Machine Learning Algorithms has been experimented to perform comparative  study and choosing the most effective model for prediction.
- Models used are:
    - Logistic Regression
    - Decision Tree
    - Support Vector Machine
    - Random Forest
    - XGBoost
    - CatBoost
    - LightGBM
    - Ensemble Model
    - AutoML Model
    - TabNet Classifier


_________________________________________
## **Explanation**

- Explainable Models are used to perform causal inference on black box machine learning models.
- Here two model-agnostic explainable techniques has been used to perform explaination.
- The best performing model from the above comparative analysis has been chosen to perform explaination study on.
- The two explainable ai models are:
    - LIME(Local Interpretable Model-Agnostic Explanations)
    - SHAP(SHapley Additive exPlanations)

_________________________________
## **Inference**
- After all the comparative analysis is completed and the results are explained by explainable ai techniques, inference has been done in this section

______________________________
## **Run all in one code**
```
from main import main_pipeline

data_path = "data"
print("LightGBM Model...")
main_pipeline(data_path)
print("Done!!")
```
