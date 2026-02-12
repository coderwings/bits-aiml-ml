# bits-aiml-ml
Repo for ML assignment.

Problem statement: Early Stage Diabetes Risk Prediction using following 6 classification models.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

Dataset description : https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
This dataset contains the sign and symptpom data of newly diabetic or would be diabetic patient.
Dataset Shape: (520, 17)



|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|
|Logistic Regression|0\.9327|0\.9797|0\.9332|0\.9327|0\.9328|0\.8587|
|Decision Tree|0\.9904|0\.9922|0\.9906|0\.9904|0\.9904|0\.9800|
|KNN|0\.9442|0\.9925|0\.9470|0\.9442|0\.9446|0\.8861|
|Naive Bayes|0\.9000|0\.9592|0\.9005|0\.9000|0\.9002|0\.7896|
|Random Forest (Ensemble)|0\.9981|1\.0|0\.9981|0\.9981|0\.9981|0\.9959|
|XGBoost (Ensemble)|0\.9942|1\.0|0\.9943|0\.9942|0\.9942|0\.9879|
