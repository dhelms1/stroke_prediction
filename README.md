# Stroke Prediction

<img src="/headers/stroke.jpg" width="600">

---

### Introduction
Stroke is the third leading cause of death in the United States, with over 140,000 people dying annually. Each year approximately 795,000 people suffer from a stroke with nearly 75% of these occuring in people over the age of 65. High blood pressure is the most important risk factor for stroke [(Stroke Center)](http://www.strokecenter.org/patients/about-stroke/stroke-statistics/#:~:text=More%20than%20140%2C000%20people%20die,and%20185%2C000%20are%20recurrent%20attacks.).

--- 

### Data & Project Overview

The data originated from the [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) repository for stroke prediction. There are 11 features that were recorded for 5110 observations. The goal for this project will be to explore the data and find any correlations between features and the response variable stroke that will allow us to engineer new features for the data. After doing this, we will make a comparison between Statistical Modeling and Ensemble Modeling to see which we are able to achieve better results with. Note that these models will be evaluated by an F-Beta and Recall score since avoiding a missed diagnosis is the main focus.

#### Extra Libraries:
- [StatsModels](https://www.statsmodels.org/stable/index.html) - used for statistical modeling (GLMs).
- [ImBalanced-Learn](https://imbalanced-learn.org/stable/) - used for sampling methods and ensemble classifiers.

#### Data Format:
For the statistical modeling section, the data was reformatted in two ways to accommodate the large class imbalance (around 20x more observations of "No Stroke" compared to "Stroke"):
- Training data was balanced by using SMOTE (Synthetic Minority Oversampling Technique) to increased of minority "stroke" class to a 3:4 ratio with the majority class "no stroke". This resulted in around 3400 majority observations (0) and 2500 minority observations (1).
- Testing data was balanced using the NearMiss algorithm, which undersampled the majority class to a 4:3 ratio with the minority class. This resulted in around 120 majority and 90 minority observations to be used for evaluation. *Note: when evaluating based on oversampled data, I did not feel the results were as accurate since repeated observations were increasing the scores. I want the model to be prepared for real world data rather than higher metrics on repeated data.*

For the ensemble modeling section, the data was reformatted... *To be filled in*

---

### Findings
#### Statistical Modeling:
For the statistical modeling section, each of the 3 final models were selected using a feature subset selection process (Best, Forwards, and Backwards). Each of the 3 models selected the same subset of 4 features, all of which were statistically significant (p-value < 0.05) and included: *age, bmi, age_over_45, & never_smoked*. From the base model (fit on all features), we have the following improvements:  
- True Negatives decreased by 2, while False Positives increased by 2 (*more people classified as stroke that did not have a stroke*).
- False Negative decreased by 21, while True Positives increased by 21 (*more people classified as stroke that actually had a stroke*).
- Precision increased from 57% to 66%.
- Recall increased from 58% to 82% (*this was the most important evaluation metric to improve*).
- Accuracy increased from 63% to 84%.

#### Ensemble Modeling:
*To be filled in...*
