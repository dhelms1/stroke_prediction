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
