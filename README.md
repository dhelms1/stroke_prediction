# Stroke Prediction

<img src="/headers/stroke.jpg" width="600">

---

### Data & Project Overview
Stroke is the third leading cause of death in the United States, with over 140,000 people dying annually. Each year approximately 795,000 people suffer from a stroke with nearly 75% of these occuring in people over the age of 65. High blood pressure is the most important risk factor for stroke [(Stroke Center)](http://www.strokecenter.org/patients/about-stroke/stroke-statistics/#:~:text=More%20than%20140%2C000%20people%20die,and%20185%2C000%20are%20recurrent%20attacks.). The data originated from the [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) repository for stroke prediction. There are 11 features that were recorded for 5110 observations. The goal for this project will be to explore the data and find any correlations between features and the response variable stroke that will allow us to engineer new features for the data. After doing this, we will make a comparison between Statistical Modeling and Ensemble Modeling to see which we are able to achieve better results with. Note that these models will be evaluated by an F-Beta and Recall score since avoiding a missed diagnosis is the main focus.

#### Extra Libraries:
- [StatsModels](https://www.statsmodels.org/stable/index.html) - used for statistical modeling (GLMs).
- [ImBalanced-Learn](https://imbalanced-learn.org/stable/) - used for sampling methods and ensemble classifiers.

#### Data Format:
For the **statistical modeling** section, the data was reformatted in two ways to accommodate the large class imbalance (around 20x more observations of "No Stroke" compared to "Stroke"):
- Training data was balanced by using SMOTE (Synthetic Minority Oversampling Technique) to increased of minority "stroke" class to a 3:4 ratio with the majority class "no stroke". This resulted in around 3400 majority observations (0) and 2500 minority observations (1).
- Testing data was balanced using the NearMiss algorithm, which undersampled the majority class to a 4:3 ratio with the minority class. This resulted in around 120 majority and 90 minority observations to be used for evaluation. *Note: when evaluating based on oversampled data, I did not feel the results were as accurate since repeated observations were increasing the scores. I want the model to be prepared for real world data rather than higher metrics on repeated data.*

For the **ensemble modeling** section, the data was reformatted in the following ways to accommodate the class imbalance:
- Training data was left untouched since the ensemble algorithms we used are able to handle the imbalance within the model itself.
- Testing data was resampled so that we would have a "Stroke" to "No Stroke" ratio of 2:3, resulting in around 50 minority and 75 majority observations (slightly smaller than the statistical modeling data).
- An important note is that the extra observations from the majority class (after being undersampled) in the testing data were added back into the training data so that we had more data to train on. This was due to the algorithms being able to handle class imbalance (so more majority observations would not have a negative effect).

---

### Findings
#### Statistical Modeling:
For the statistical modeling section, we first fit an initial model using all features that resulted in the following output:

<img src="/rm_img/glm_base.jpg" width="300"> <img src="/rm_img/glm_base_met.jpg" width="425">

Following this, 3 models were fit using a feature subset selection process (Best, Forwards, and Backwards). Each of the 3 models selected the same subset of 4 features, all of which were statistically significant (p-value < 0.05) and included: *age, bmi, age_over_45, & never_smoked*. From the base model (fit on all features), we have the following improvements:  
- True Negatives decreased by 2, while False Positives increased by 2 (*more people classified as stroke that did not have a stroke*).
- False Negative decreased by 21, while True Positives increased by 21 (*more people classified as stroke that actually had a stroke*).
- Precision increased from 57% to 66%.
- Recall increased from 58% to 82% (*this was the most important evaluation metric to improve*).
- Accuracy increased from 63% to 74%.

<img src="/rm_img/glm_final.jpg" width="300"> <img src="/rm_img/glm_met.jpg" width="425">

*Note: Backwards Feature Selection is shown, but all 3 methods had same confusion matrix and evaluation metrics.*

#### Ensemble Modeling:
For the ensemble modeling section, we used the ImBalanced-Learn API (linked above in the extra libraries section) which has similar ensemble methods to Sklearn, but better suited to handle imbalanced classes. To model chosen for this section is the BalancedRandomForestClassifier, which had the highest recall of the 3 initial models that were fit (see ensemble modeling section in notebook for more details). The initial BRFC model had the following results:

<img src="/rm_img/ens_base.jpg" width="300"> <img src="/rm_img/ens_met.jpg" width="425">

After fitting the above model, we proceeded with hyperparameter tuning in order to see if we could improve our results. However, the tuned model (for the most part) was equivalent to the initial model except for one change. The number of estimators was changed from 100 to 25, which had no negative impact on the predicted results and doubled the computational speed of the model. The confusion matrix and evaluation metrics for the tuned model were the same as above (so they will not be shown), but the following feature importance were graphed for the tuned model:

<img src="/rm_img/ens_feat_importance.jpg" width="300">

An important note is that the *age, bmi, and avg_glucose_levels* were not normalized and all other features had discrete values in the range [0,1], but normalizing the inputs resulted in the same feature importance but with worse performance.

*Note: The tuned model was also run through the [Exhaustive Feature Selector](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/) for mlxtend to find the best combination of features (ranging from 2 to 7), but the subset model had slightly worse performance. The original tuned model was kept due to this (see end of notebook for more info).*

---

### Conclusion
*To be filled in...*

