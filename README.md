# Predicting Fetal Risk Using Cardiotocographic Data
![header](https://www.babycenter.in/ims/2015/08/ThinkstockPhotos-467038537_wide.jpg)

**Authors**: Ismael Araujo and Jacob Heyman

***Note: This is a revised version of the original project created by Ismael Araujo and Jacob Heyman. You can find the original version [here](https://github.com/Jacobheyman702/Predicting-Fetal-Risk-Using-Cardiotocographic-Data "here").***

## Overview
As medical technology advances, the rate of preventable child mortality decreases. Lowering child mortality rates has become a key goal in advancing any society and a key in human progress as a whole. While many advancements have improved the mortality rate, not all of these practices are globally available. To reduce mortality rates in cost-effective and readily available solutions need to be applied and perfected.

One such cost-effective and relatively simple method would be the use of Cardiotocograms (CTG). The CTG is a non-invasive fetal monitor which is used to assess fetal health. The CTG is used to detect fetal heart rate (FHR), uterine contractions, fetal movement, and sudden heart rate changes. Extreme changes to the FHR result in surgical intervention through the use of cesarian delivery(C-section). Currently, doctors rely on a visual analysis of the CTG, leading to erroneous exam interpretations. Minute changes which may be extremely detrimental to fetal health may also not be visible to the naked eye.

Conversely, some observable changes in fetal heart rate may appear to be fetal distress but are just a response to other factors like uterine contractions. In case doctors opt to perform C-section increases the chances of mortality for the child and the mother. For these reasons’ visual analysis of a CTG exam is limited by human error and cannot definitively detect fetal distress.

Machine learning algorithms could become a viable option to improve the diagnosis of fetal distress with a CTG and decrease the mortality rate. A well-trained model will identify which variable changes to the FHR have the most significant effect on fetal health. To lower the overall risk of child mortality, an ML model must accurately classify between normal and distressed fetal health and be precise enough in its predictions to prevent any unnecessary interventive surgical procedures.
We used a CTG exam dataset found on Kaggle from The Journal of Maternal-Fetal Medicine for our analysis. The dataset consisted of 2126 CTG exam records classified into three classes (normal, suspect, pathological). In this initial analysis, we decided to create a normal and distressed binary class, combining the suspect and pathological classes into one class.

To address the issue of eliminating human error and erroneous surgical intervention, our goal of this analysis was to identify which key features played a role in determining the healthy or distressed classes and to produce a highly precise model to eliminate unnecessary C-sections. To achieve these goals, we performed an extensive exploratory data analysis, created baseline models, and performed hyperparameter tuning of models for both an experimental feature and unaltered CTG data set. Our final model had evident feature importance and the best evaluation metrics.



## Bussiness Problem 
Preventing child mortality is a crucial indicator of medical progress. A large percentage of deaths are the result of low access to resources that monitor fetal health. One cost-effective solution to monitor fetal health is the use of Cardiotocograms(CTGs). CTGs monitor fetal heart rate, fetal movement, uterine contractions, sudden heart rate changes, and many other health metrics. CTG scans are currently interpreted via visual analysis by the physician, and erroneous errors may increase fetal health risk. Using a Kaggle database of collected CTG exams, we aim to identify which features have the most impact on fetal health, being either normal or distressed. Using these features, we also aim to tune a model that provides the best fetal health class predictions. To identify the key features and create a precise model, we considered the following research questions

**What feature has the most significant influence on the model**

**What, if any, new features affect the model**

**Which model makes the best predictions of fetal health class and has the most precise predictions.**



## Data
Our data came from Kaggle.com (https://www.kaggle.com/andrewmvd/fetal-health-classification).  The data comes from 2126 cardiotocogram measurments described in Ayres de Campos et al. (2000) SisPorto 2.0 A Program for Automated Analysis of Cardiotocograms. J Matern Fetal Med 5:311-318.  The CTG measurments where used to create a classification model to classify fetal health being either normal or distressed.  The DataFrame included the following columns:

|   |   |
| ------------ | ------------ |
|**baseline_value **|Baseline Fetal Heart Rate**|
|**accelerations **|Number of accelerations per second|
|**fetal_movement **|Number of fetal movements per second|
|**uterine_contractions **|Number of uterine contractions per second|
|**light_decelerations **|Number of LDs per second|
|**severe_deceleration **|Number of Sds per second|
|**prolongued_deceleration **|Number of PDs per second|
|**abnormal_short_term_variatability **|Percentage of time with abnormal short tearm variatability|
|**mean_value_of_short_term_variability **|Mean value of short term variability|
|**percentage_of_time_with_abnormal_long_term_variabilit**|Percentage of time with abnormal long term variability|
|**mean_value_of_long_term_variability **|average value of long tearm variability|
|**histogram_width **|Width of histogram using all the values from the record|
|**histogram_min **|minimum value of histogram|
|**histogram_max **|maximum value of hitogram|
|**histogram_number_of_peaks **|number of peaks in the exam histogram|
|**histogram_number_of_zeroes **|number of zeros in the exam histogram|
|**histogram_mode **|histogram mode|
|**histogram_mean **|histogram mean|
|**histogram_median **|histogram median|
|**histogram_variance **|histogram variance|
|**histogram_tendency **|histogram tendancy|
|**fetal_health **|0: normal 1:distressed|




## Methods
To determine the key features in fetal health classification and produce the best possible model, we implemented the following methods in our analysis
- Initial Data Analysis and Data cleaning
    - changed target class label into binary fetal health class (normal: 1, distressed: 2)
- EDA
    - In depth analysis of features and relations with other features and the target class
    ![histogram_mean](./images/histogram_class_by_hist_mean.png)
- Baseline models
    - default models(KNN, Logistic Regression, Decision Tree, and Random Forest)
- Feature Engineering
- Created two modeling process notebooks Vanilla and Experimental
    - to compare unaltered dataset model tuning to experimental feature dataset
- modeling and hyperparameter tuning for both databases
    - KNN
    - Logistic Regression
        - Bagging classifer 
    - Decision Tree
    - Random Forest
        - GridSearch
        - Model with best parameteres
    - XGBoost
        - Gridsearch
        - Model with best parameters
    
- checked feature importance of best models





## Results
In our exploritory analysis we preformed several baseline models to see how the dataset classifed fetal health.  Each baseline model was surprisingly accurate at predicting the classes.  We decided to preform two seperate modeling processes focusing on tuning the hyperparameters on the untouched dataset(vanilla) and an experimental dataset with new features.  For the experimental dataset we created several new features, using binning, dummy variables and a combination of several existing features.
![sqrt_total_change](./images/sqrt_total_change_by_class.png)

The best model overall for **vanilla** was a grid search best features random forest with the following hyperparameters and evaluation metrics:
    
   - **Best Parameters:** {'criterion': 'entropy', 'max_depth': 10, 'max_features': 10, 'n_estimators': 50, 'oob_score': False}
   - **Evaluation Metrics** 
       - **Accuracy:** 0.9605263157894737
       - **Recall:** 0.9755501222493888
       - **F1 Score:** 0.9743589743589743
       - **Precision:** 0.973170731707317
   - **Top 10 Features**
       
       
![rfc_top_10](./images/vanilla_rfc_top_10.png)



The best model overall for **Experimental** was a grid search best features XGBboost with the following hyperparemeters and evaluation metrics:

   - **Best Parameters:** {'colsample_bytree': 0.45,'learning_rate': 0.07,'max_depth': 12,'min_child_weight': 1,
                           'n_estimators': 500}
   - **Evaluation Metrics**
       - **Accuracy:** 0.9661654135338346
       - **Recall:** 0.9902200488997555
       - **F1 Score:** 0.9782608695652174
       - **Precision:** 0.9665871121718377
   - **Top 10 Features:**


![ex_xgboost_top_10](./images/ex_xgboost_top_10.png)






## Conclusions
Cardiotocographs are a cost effective method to moniter fetal health, and use are used as a tool to  lower the child mortality rate.  One of the major issues with CTG monitereing was errors made through visual analysis.  Any invterventive surgery weather necessary or not increase risk.  In this study, we utalized multiple modeling processes to predict fetal health class, and attempted to create the highest precision in our prediction to reduce false positive classification instances.  With the CTG exam dataset from kaggle, we preformed two simultaneous medeling processes with a vanilla and experimental data set.  While the random forest model of our vanilla set had the highest precision, which was our target metric, the xgboost model of the experimental had the best overall metrics with a very high precision.  The key features in our best model where abnormal short term varitability, histogram mode, histogram min, histogram mean and histogram width.  All of these features are measurements of the fetal heart rates change over time.  This lines up with what is visually obsereved for fetal distress in the O.R.  Our model has a high precision and other metrics and can predict on these features to a greater effect than the standared visual analysis.  We believe that our model will help in the precise classification of fetal distress and improve the overall child mortality rate by reducing  risk through human error.  




## Next Steps
- make a multiclass for fetal health, multiple classes of fetal health to better classify the health state
- find more CTG exam data to better train model.  This data set did not have a lot of instances for poor fetal health
- consider maternal health and other diagnostic metrics into the model(heart rate, oxygen level, what anestetics are used)

 

## For More Information
- kaggle dataset [kaggle_data](https://www.kaggle.com/andrewmvd/fetal-health-classification)
- ncbi papers [Hoodbhoy, Noman](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6822315/)
- Notebooks
    - [Analysis](./fetal_heatlh_data_analysis.ipynb)
    - [Vanilla modeling process](./vanilla_model_cleaned.ipynb)
    - [experimental modeling process](.experimental_feature_Modeling.ipynb)






## Repository Structure

```
├── README.md                           <- Overview of the project
├── fetal_heatlh_data_analysis          <- Analysis and EDA of Fetal health dataset
├── Vanilla_model_cleaned               <- Modeling process with vanilla dataset
├── experimental_feature_Modeling       <- Modeling process with experimental dataset
├── experiment_features.csv             <- experimental dataset       
├─  data.csv                            <- folder with fetal health dataset
└── Phase_3_Project_Pres                <- Presentation slidedeck
```

