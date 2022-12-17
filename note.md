# Credit Card Fraud Detection

In the following project we are going to analyse a **credit card fraud detection** dataset, trying to understand its main characteristic and its data patterns. We'll follow some predefined steps of data mining:

- Set the goal: set the goal to achieve which can be, for instance, identify most of the frauds or reduce the number of false positives
- Data analysis: understand the data we obtaind, their statistical properties, if there are null values or missing values to replace, if the number of features should be decreased, the presence of outliers...
- Data preprocessing: based on the above observations we'll process the dataset to obtain a different view of the data, cleaned, without duplicates, with only the most relevant features...
- Classification: select a list of models to evaluate, use techniques like cross-validation, holdout, grid approach or similar technique to reach our goal and to obtain the optimal results
- Evaluation: the models executed in the previous steps will be evaluated on different performance metrics and based on different criteria like execution time, precision, accuracy etc. we are gonna pick one model which will be, in a real scenario, deployed in the system

These are the main steps of data mining, in a real scenario the evaluation could lead to additional data analysis and processing, hence additional evaluation with different models or different parameters to obtain the optimal result.

The dataset [1]() we pick is the [European Cardholders Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/versions/2?resource=download), contains transactions made by credit cards in *September 2013* by european cardholders. This dataset presents transactions that occurred in **two days**, where we have **492 frauds out of 284,807 transactions.** We'll discuss later on the imbalanced nature of it.

## Why credit card fraud?
Find paper on credit card fraud and similar, even using the paper of the dataset and looking in the references or introduction.

## Why this particular dataset? 
As already said I wanted to focus on fraud detection on transaction, if possible related to banks or similar, browsing through all the main dataset available on the web, there were few possibilities apart from the one choosen. The first one was an old dataset on credit data, [German Credit Data](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29), the problem here is the low number of instances (nearby a thousand) and additionally it is not strictly related to fraud detection. The second choice was the use of **synthetic data**, in particular it was possible to generate an ... number of transactions with the original features which can be later modified and transformed. This last solution was a good tradeoff however it still represents a fake scenario which even with multiple random generation of data it still is easy to model and it's very far from being compared with a real dataset.









## Set the goal
Given the fact that in credit card fraud detection the FN has an higher weight than FP, our goal will be to detect most of the positive elements, so increasing the recall (aka the ratio of true positive), the other metrics doesn't give a useful insights in the performance of our model (in case of credit card fraud detection). We'll discuss later about other metrics used in assessing the performance of a particular model. In general our goal is to reach an high recall and a good accuracy.


## Data Analysis
 The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.



If we display a full description of the choosen dataset we can figure out many useful things:
- V1, ..., V28 are 28 features results of the PCA process
- Time, Amount and Class are the only "unchanged" features

As the name says, ***Time*** contains the seconds elapsed between each transaction and the first transaction in the dataset. ***Amount*** describes the amount of cash transferred with the transaction and ***Class*** informs us if the transaction has been labeled as fraudulent (1) or not (0). All the features are **numerical.**


Il dataset scelto contiene 




Imbalanced
Outlier??





## Split of the data
In a fraud detection context, the transactions of the test set occur chronologically after the transactions used for training the model 
 (evaluate distribution label timestamp before doing this) (Andrea Dal Pozzolo, Giacomo Boracchi, Olivier Caelen, Cesare Alippi, and Gianluca Bontempi. Credit card fraud detection: a realistic modeling and a novel learning strategy. IEEE transactions on neural networks and learning systems, 29(8):3784–3797, 2017)
 
Qui mi ricollego alla domanda: il test set deve iniziare a tempo zero o no??


## Metrics
The **recall, specificity, precision, and F1 score metrics,** also known as threshold-based metrics, have well-known limitations due to their dependence on a decision threshold which is difficult to determine in practice, and strongly depends on the business-specific constraints. They are often complemented with the **AUC ROC**, and more recently, the **Average Precision (AP)** metrics. The AUC ROC and AP metrics aim at assessing, with a single number, the performance for all possible decision thresholds, and are referred to as threshold-free metrics.

The AUC ROC is currently the *de-facto* metric for assessing fraud detection accuracies [Cha09, DP15]. Recent research has however shown that this metric is also misleading for assessing highly imbalanced problems such as fraud detection [Mus19], and recommended using the *Precision-Recall curve* and *AP metric* instead [BEP13, SR15].

Moreover we should fix an additional thing related to positive and negative classes. A fraud detection problem is in nature a **cost-sensitive problem:** missing a fraudulent transaction is usually considered more costly than raising a false alert on a legitimate transaction, indeed blocking the legitimate transaction of a user. For this reason the TP,FP,TN and FN should be weighted, e.g. a FN has more weigth than a FP.

We will not fix this. However:
- Display ROC curve (see github code) (Chapter3 5.)
- Plot or print average precision (AP)
- Precision-recall curve

Of course the more complex fraud detection system make more assumptions, possible also by the availablity of retrieving data: time inspection of the transaction, time of reporting, time evaluation (the labeled data aka test data will be obtained after one week of evaluation??).... More advanced performance metrics, in the literature, are used which considers different weight for positive classes etc., but we'll not use it.


# Model selection
Pros and cons of the model selected and why them



Model selection: Model selection consists in selecting the prediction model that is expected to provide the best
performances on future data. Steps: list of models, train on training data, take the best using cross-validation techniques,
select the one with the highest performances.

Here we use cross validation to compare the results of different models on different sampled data, at the end we'll pick the best result and we'll try to apply a grid approach to additionally improve the performance of the optimal model. Or if we don't use grid approach we'll give an explaination of possible adjustments which can help with generalization or similar even if we apply the cross validation.

For every results of every model we'll give also the execution time because it can be very important in big dataset and many times we have to handle a tradeoff between performance metrics and execution time.


The model that we are going to test are:
- For unbalanced data, aka original dataset and NCL: Weighted Random Forest. The grid approach we'll be exploited to obtain the optimal weight (considering the ROC AUC?? See how grid search works and if this metric can be implemented)
- For balanced data, aka RandomOverSampling, RandomUnderSampling, SMOTE, borderlineSMOTE and SMOTEENN:
   - LogisticRegression
   - LinearSVM
   - DecisionTree
   - RandomForest
   
We use these models because we already saw them in class and it was very suitable for numerical data, as in our case, the DecisionTree was also picked to compare it with RandomForest approach even if is well documented that the DecisionTree performs worst than RandomForest. The KNN was not used because it was largely used in the sampling method and I was afraid that I could lead to a sort of overfitting result. Si utilizza ?????

## Cross-validation
The performance of a model on training data is often a bad indicator of the performance on future data [Bis06, Bon21, FHT01]. The former is referred to as training performance, and the latter as test performance (that is, the performance of the model when tested on unseen data). In particular, increasing the degree of freedom of a model (such as the depth for a decision tree) always allows to increase the training performance, but usually leads to lower test performances (overfitting phenomenon). Validation procedures aim at solving this issue by estimating, on past data, the test performance of a prediction model by setting aside a part of them [CTMozetivc20, GvZliobaiteB+14]


We should handle the KFold to be coherent with the Time





To read last chapter (6)


How many data, missing, null etc. on the description of the dataset
Tip: for good slides think that you design the slide in a way that you are able
to discuss them even after one month or more


[1] Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
