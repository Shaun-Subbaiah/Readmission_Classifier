# Diabetes 30-Day Readmission Classifier

## Introduction

This project is a machine learning classifier designed to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge. I built this primarily as a hands-on way to work with real healthcare data and to understand how machine learning models behave in clinically realistic settings, especially when the problem is messy, imbalanced, and full of tradeoffs.

One of the biggest challenges here is class imbalance as only about 11% of patients in the dataset are readmitted. Rather than treating this as a bit of a problem, I leaned into it and used this project to explore how different modeling choices like feature engineering, class weighting, validation strategy, directly affect clinical usefulness. The goal wasn’t just to maximize a single metric, but to understand when a model is actually helpful in practice.

This is version 2 (V2) of the project. Compared to an earlier baseline, V2 has 9 new features it looks at, applies more aggressive and adjustable class weighting to cater to the needs of the people using it, and introduces a difficulty based validation strategy that specifically tests the model on borderline cases where clinical decisions are hardest. 

Overall, working on this project taught me a lot about the practical side of healthcare ML, dealing with messy real world data, understanding clinical tradeoffs, and balancing model's complexity with how user friendly it can be. It's been a great learning experience in moving beyond just the general common problems to something that could actually be useful in a hospital setting!


## Dataset & Credit

This project uses the Diabetes 130-US Hospitals dataset from the UCI Machine Learning Repository, which contains 10 years (1999–2008) of inpatient data from 130 U.S. hospitals.

**Citation:**
Beata Strack et al., *Impact of HbA1c Measurement on Hospital Readmission Rates*, BioMed Research International, 2014.

**Dataset Link:**
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008



## How to Use

### Run the classifier directly from the command line:

```
python readmission_classifier.py diabetic_data.csv
```
where diabetic_data.csv is the desired dataset you wish to use.

Make sure the dataset CSV file is in the same directory as the script.

**NOTE:** Make sure the dataset used follows the same format as the UCI Daibetes dataset as thats what the model was built for!

You’ll be prompted to choose which model(s) to train:

* **lr** — Logistic Regression only
* **rf** — Random Forest only
* **both** — Train and compare both models

### Adjusting the model's weight:
The classifier allows adjustment of the *readmit_weight* parameter to control the precision–recall trade-off:

It is located on line 16 of the code, alsong with some recommendations and metrics with which I tested on:
```
def __init__(self, csv_file, readmit_weight=12, random_state=42)
```

Higher weights prioritize catching more readmissions (higher recall) but result in more false alarms.



## How It Works

1| **Data Loading -** Reads the CSV file and handles missing values in the dataset.

2| **Feature Preparation -** Assigns values to categorical variables, engineers clinical risk indicators, and groups diagnosis codes into meaningful categories.

3| **Difficulty-Based Validation -** Separates “clear-cut” cases from “ambiguous” cases to test performance where clinical decisions are hardest.

4| **Model Training -** Trains the selected model(s) using adjustable class weights to handle the class imbalance.

5| **Evaluation -** Reports AUC, precision, recall, and F1 score.

6| **Visualization -** Shows confusion matrices, ROC curves, and feature importance (Random Forest only).



## What It Does

Builds a machine learning system to predict 30-day hospital readmission risk for diabetic patients using real clinical data.

Trains and compares two models:

* **Logistic Regression -** as a simple starting point by determineing whether linear relationships like with prior admission or length of stay ahev any predictive value, so not very complex.
* **Random Forest -** to capture more complex non-linear patterns between patient features. It performed slightly better in testing, showing that some complexity helps.

Adds a few clinically useful features that help with the prediction, including:

* High healthcare utilization (prior admissions or frequent ER visits)
* Polypharmacy (10+ medications)
* Long hospital stays (≥7 days)
* Grouped diagnosis codes into broad clinical categories to reduce noise.

Uses a difficulty-based validation strategy that:

* Trains on clearly low risk and high-risk cases (either very short or very long stays)
* Tests on medium length stays, where clinical decisions are more ambiguous
* Achieves performance above random prediction on these harder cases.

Reports standard classification metrics like AUC, precision, recall and F1 and generates graphs, including:

* Confusion matrices
* ROC curves
* Feature importance plots (Random Forest only)



## Understanding the Metrics

* **AUC (Area Under the ROC Curve) -** Measures how well the model separates readmitted vs non-readmitted patients. 

* **Precision -** Of the patients flagged as high-risk, how many actually readmit? Lower precision means more false alarms.

* **Recall -** Of all patients who actually readmit, how many did the model catch? Higher recall means fewer missed cases.

* **F1 Score -** (Only when testing individual models) A balance between precision and recall. Useful when both false alarms and missed cases matter.

Here, recall is was slightly prioritized because missing a readmission can have higher clinical and financial costs than a false positive.



## Limitations & Future Improvements

* **Simplified diagnosis encoding** — Diagnosis codes are grouped broadly, more granular modeling could maybe capture better clinical patterns.
* **Cross validation** — The model just splits the data 80/20, adding a more uniform cross validation could get rid of the randomness of the split.
* **Cost-benefit analysis** — An function could be added to determine the economic value/impact of the model.
* **Combining models** — Combining the functions of models could help improve the functioning of the classifier.


