## **Predictive Models**
__________________________
### **Logistic Regression**

Logistic Regression is a statistical method used to analyze a dataset in which there are one or more independent variables (predictors) that determine an outcome or a dependent variable (response). In particular, it is used to model the probability of a certain event or outcome occurring based on the values of the independent variables.

The dependent variable in logistic regression is binary, meaning it can take only one of two possible values (e.g., 0 or 1). Logistic regression models the probability of the dependent variable taking on the value of 1 given the values of the independent variables, using a mathematical function called the logistic function.

The logistic function (also known as the sigmoid function) maps any real-valued input to a value between 0 and 1, which can be interpreted as a probability. The logistic regression model estimates the coefficients of the independent variables, which indicate the strength and direction of their relationship with the dependent variable.

*For more reading go [here](https://www.ibm.com/in-en/topics/logistic-regression#:~:text=Resources-,What%20is%20logistic%20regression%3F,given%20dataset%20of%20independent%20variables.)*

*For sklearn documentation for [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)*

___________________________
### **Decision Tree**

Decision Tree is a popular machine learning algorithm used for solving classification and regression problems. It works by constructing a tree-like structure where each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label or a numerical value (in case of regression).

The algorithm begins with the entire dataset as the root node, and then recursively splits the data into subsets based on the values of the attributes until all data in a subset belong to the same class (in case of classification) or until a stopping criterion is met (in case of regression).

The splitting process is done by selecting the attribute that provides the most information gain or the best split. Information gain is a measure of the amount of uncertainty that is reduced by the split. The attribute with the highest information gain is chosen as the splitting attribute.

*For more reading go [here](https://www.ibm.com/in-en/topics/decision-trees#:~:text=A%20decision%20tree%20is%20a,internal%20nodes%20and%20leaf%20nodes.)*

*For sklearn documentation for [here](http://scikit-learn.org/stable/modules/tree.html)*
______________________________
### **Support Vector Machine**

Support Vector Classifier (SVC) is a popular machine learning algorithm used for solving binary and multi-class classification problems. It works by finding the hyperplane that best separates the two classes in a high-dimensional feature space.

The basic idea behind SVC is to find the hyperplane that maximizes the margin between the two classes. The margin is the distance between the hyperplane and the closest data points from each class. The closest data points are called support vectors, hence the name Support Vector Classifier.

SVC can handle both linearly separable and non-linearly separable datasets. In the case of non-linearly separable data, SVC uses a kernel function to map the data to a higher-dimensional space where it becomes linearly separable. The most commonly used kernel functions are linear, polynomial, and radial basis function (RBF).

The optimal hyperplane is found by solving a quadratic optimization problem. This optimization problem can be solved using various techniques such as the Sequential Minimal Optimization (SMO) algorithm or the Quadratic Programming (QP) method.

*For more reading go [here](https://en.wikipedia.org/wiki/Support_vector_machine)*

*For sklearn documentation for [here](http://scikit-learn.org/stable/modules/svm.html)*
_______________________________
### **Random Forest**

Random Forest is a popular machine learning algorithm used for solving classification and regression problems. It is an ensemble method that combines multiple decision trees to create a more accurate and stable model.

The basic idea behind Random Forest is to build a large number of decision trees on different subsets of the training data and different subsets of the features. Each tree in the forest is trained on a random subset of the data and a random subset of the features. This randomization helps to reduce overfitting and improve the generalization performance of the model.

The final prediction of the Random Forest is made by aggregating the predictions of all the individual trees. For classification problems, the final prediction is usually the class that receives the most votes from the trees. For regression problems, the final prediction is usually the mean or median of the predictions of all the trees.

*For more reading go [here](https://www.ibm.com/in-en/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,both%20classification%20and%20regression%20problems.)*

*For sklearn documentation for [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)*
______________________________
### **XGBoost**

XGBoost (eXtreme Gradient Boosting) is a popular machine learning algorithm used for solving regression, classification, and ranking problems. It is an optimized implementation of the Gradient Boosting Trees algorithm, which is an ensemble method that combines multiple decision trees to create a more accurate and robust model.

The basic idea behind XGBoost is to iteratively add decision trees to the model, each of which corrects the errors of the previous trees. The trees are added in a greedy manner, such that each tree minimizes the loss function of the entire model. The loss function is a measure of the difference between the predicted and actual values.

XGBoost uses a gradient-based optimization method to train the model, which makes it faster and more accurate than traditional Gradient Boosting Trees. It also includes several advanced features such as regularization, tree pruning, and handling missing values, which help to reduce overfitting and improve the generalization performance of the model.

*For more reading go [here](https://en.wikipedia.org/wiki/XGBoost)*

*For XGBoost documentation for [here](https://xgboost.readthedocs.io/)*
_______________________________
### **CatBoost**

CatBoost is a popular machine learning algorithm used for solving classification and regression problems. It is an optimized implementation of gradient boosting trees that is specifically designed to handle categorical features and improve generalization performance.

The basic idea behind CatBoost is similar to that of XGBoost and other gradient boosting algorithms, which is to iteratively add decision trees to the model and optimize the loss function. However, CatBoost has several advanced features that make it unique and more effective.

One of the key features of CatBoost is its handling of categorical features. CatBoost uses an algorithm called ordered boosting, which exploits the order of the categorical features to create a more accurate model. It also uses a novel algorithm for encoding categorical features that can handle high cardinality features without the need for one-hot encoding.

CatBoost also includes several advanced regularization techniques to reduce overfitting, such as random feature selection, feature combinations, and bagging. It can handle missing values by using a combination of mean imputation and gradient-based methods.

*For more reading go [here](https://en.wikipedia.org/wiki/CatBoost)*

*For CatBoost documentation for [here](https://catboost.ai/)*
_______________________________
### **LightGBM**

LightGBM (Light Gradient Boosting Machine) is a popular machine learning algorithm used for solving classification and regression problems. It is an optimized implementation of gradient boosting trees that is specifically designed to handle large-scale datasets and high-dimensional features.

The basic idea behind LightGBM is similar to that of other gradient boosting algorithms, which is to iteratively add decision trees to the model and optimize the loss function. However, LightGBM has several advanced features that make it unique and more effective.

One of the key features of LightGBM is its handling of large-scale datasets. LightGBM uses a technique called Gradient-based One-Side Sampling (GOSS) to speed up the training process and reduce memory usage. GOSS samples the instances in the dataset based on their gradient information, which reduces the number of instances needed for training while retaining the informative instances.

LightGBM also uses a novel algorithm for splitting the data into leaves, called Exclusive Feature Bundling (EFB). EFB groups similar features together, which reduces the number of features needed for training while retaining the informative features.

LightGBM includes several advanced regularization techniques to reduce overfitting, such as feature fraction, bagging, and histogram-based gradient boosting. It can handle missing values by using a combination of mean imputation and gradient-based methods.

LightGBM has several advantages over other gradient boosting algorithms, including faster training time, lower memory usage, and better handling of large-scale datasets and high-dimensional features. It is also easy to use and does not require much parameter tuning.

*For more reading go [here](https://en.wikipedia.org/wiki/LightGBM)*

*For LightGBM documentation for [here](https://lightgbm.readthedocs.io/)*
________________________________
### **Ensemble Model**

Ensemble machine learning models are techniques that combine multiple models to improve the accuracy and robustness of the overall model. The idea behind ensemble models is that by combining multiple models that may have different strengths and weaknesses, the ensemble can achieve better performance than any single model alone.

There are several types of ensemble models, including:

1. Bagging: Bagging, or bootstrap aggregating, involves training multiple models on different subsets of the training data and combining their predictions. The goal of bagging is to reduce the variance of the model by averaging the predictions of multiple models.

2. Boosting: Boosting involves training multiple models sequentially, where each model is trained to correct the errors of the previous model. The goal of boosting is to reduce the bias of the model by combining multiple weak models into a strong model.

3. Stacking: Stacking involves training multiple models on the same data and combining their predictions using a meta-model. The goal of stacking is to reduce the bias and variance of the model by combining the strengths of multiple models.

4. Random Forest: Random forest is a specific type of ensemble model that uses bagging and decision trees to create a more accurate and robust model. Random forest combines multiple decision trees that are trained on different subsets of the data and different subsets of features.

*For more reading go [here](https://www.sciencedirect.com/topics/computer-science/ensemble-modeling#:~:text=Ensemble%20modeling%20is%20a%20process,prediction%20for%20the%20unseen%20data.)*

*For sklearn documentation for [here](http://scikit-learn.org/stable/modules/ensemble.html)*
__________________________________
### **AutoML Model**

AutoML (Automated Machine Learning) is a process of automating the tasks involved in the development and deployment of machine learning models. The goal of AutoML is to simplify the machine learning process, reduce the time and resources required to build and deploy models, and make machine learning more accessible to non-experts.

AutoML can automate many tasks involved in the machine learning process, including:

1. Data pre-processing: AutoML can automatically clean and transform data, handle missing values, and convert categorical variables to numerical values.

2. Feature engineering: AutoML can automatically generate new features from existing data, such as polynomial features, interaction terms, and feature selection.

3. Model selection: AutoML can automatically select the best model for a given task, including decision trees, random forests, support vector machines, and neural networks.

4. Hyperparameter tuning: AutoML can automatically tune the hyperparameters of the selected model, such as learning rate, regularization, and number of layers.

5. Model evaluation: AutoML can automatically evaluate the performance of the selected model on a validation set and optimize it for the desired performance metric.

*For more reading go [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)*

*For autoML documentation for [here](https://cloud.google.com/automl)*
______________________________________
### **TabNet Classifier**

TabNet (Tabular Attention-Based Neural Network) is a deep learning model that was introduced in 2019 by researchers at Google Brain. TabNet is designed for tabular data, which is data organized in a table format with rows and columns.

TabNet is an attention-based neural network that combines both deep learning and decision trees. The model can handle both categorical and continuous data and is capable of automatically selecting the most relevant features for a given task.

The key feature of TabNet is the use of a novel attention mechanism called "sparse self-attention". Unlike traditional attention mechanisms, which operate on all the inputs, sparse self-attention only attends to a small subset of the inputs. This makes TabNet more efficient and scalable, particularly for large and high-dimensional datasets.

TabNet also uses a novel decision tree-like structure called the "transformer encoder". This structure is used to encode the input features and produce the final output of the model. The transformer encoder is composed of multiple decision steps, each of which involves a decision tree that selects the most important features based on the sparse self-attention mechanism.

*For more reading go [here](https://towardsdatascience.com/tabnet-e1b979907694)*

*For sklearn documentation for [here](https://github.com/dreamquark-ai/tabnet)*

______________________
## **Explainable Models**

_____________________________
### **LIME Explainer**

LIME (Local Interpretable Model-Agnostic Explanations) is an explainable machine learning method used to understand the predictions made by complex models. LIME is designed to provide local explanations, meaning it focuses on explaining the predictions of the model for individual instances, rather than the model as a whole.

LIME works by creating a simpler, interpretable model that approximates the behavior of the original, complex model. The simpler model is trained on a subset of the original data that is similar to the instance being explained. LIME generates a set of interpretable rules that explain the behavior of the simpler model and, by extension, the behavior of the original model for that instance.

One of the key advantages of LIME is that it is model-agnostic, meaning it can be used to explain the predictions of any machine learning model, regardless of its complexity or underlying algorithm. This makes LIME a versatile and widely applicable method for model explanation.

*For more reading go [here](https://homes.cs.washington.edu/~marcotcr/blog/lime/)*

*For sklearn documentation for [here](https://github.com/marcotcr/lime)*
______________________________
### **SHAP Explainer**

SHAP (SHapley Additive exPlanations) is a method for explaining the predictions made by machine learning models. It was introduced in 2017 by Lundberg and Lee.

SHAP is based on the concept of Shapley values from cooperative game theory, which provides a way to fairly allocate the contribution of each player to a group outcome. In the context of machine learning, the "players" are the input features, and the "group outcome" is the model prediction.

The SHAP values provide a measure of the contribution of each feature to the model prediction for a given instance. The SHAP values are additive, meaning the sum of the SHAP values for each feature equals the difference between the model prediction for the instance and the expected value of the model prediction for all instances.

The SHAP values can be used to provide both global and local explanations for machine learning models. Global explanations provide an overview of the importance of each feature across all instances, while local explanations provide an explanation for a specific instance.

One of the advantages of SHAP is that it can be used with any machine learning model, including black-box models like neural networks and support vector machines. SHAP also provides a unified framework for interpreting the predictions of different machine learning models.

*For more reading go [here](https://towardsdatascience.com/understand-the-working-of-shap-based-on-shapley-values-used-in-xai-in-the-most-simple-way-d61e4947aa4e#:~:text=SHapley%20Additive%20exPlanation%20(SHAP)%2C,popularly%20used%20in%20Game%20Theory.)*

*For sklearn documentation for [here](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)*

__________________________________
## **Optimization Models**

Optimization models are used to optimize performance of complex statistical and neural network models

here the optimization technique used is Bayesian Optimization

___________________________
### **Bayesian Optimization**

Bayesian optimization is a technique for optimizing expensive black-box functions. It involves constructing a probabilistic model of the objective function, called a surrogate model, and using it to predict the next best set of hyperparameters to evaluate. The surrogate model is updated with each evaluation, and the process continues until the optimum is found or a predefined budget is exhausted.

Bayesian optimization is particularly useful when the objective function is expensive to evaluate, as it can reduce the number of function evaluations required compared to traditional optimization techniques such as grid search or random search. It has been successfully applied in various fields, including machine learning, computer vision, and material science, among others.

*For more reading go [here](https://en.wikipedia.org/wiki/Bayesian_optimization#:~:text=Bayesian%20optimization%20is%20a%20sequential,expensive%2Dto%2Devaluate%20functions.)*

*For sklearn documentation for [here](https://github.com/fmfn/BayesianOptimization)*

