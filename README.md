# bits-aiml-ml
Repo for ML assignment.

Problem statement: Early Stage Diabetes Risk Prediction using following 6 classification models.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

Dataset : https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
This dataset contains the sign and symptpom data of newly diabetic or would be diabetic patient.
Dataset Shape: (520, 17)



|index|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|---|
|0|Logistic Regression|0\.9230769230769231|0\.9773794280836535|0\.9315068493150684|0\.9577464788732394|0\.9444444444444444|0\.8203582248310268|
|1|Decision Tree|0\.9519230769230769|0\.9647887323943662|1\.0|0\.9295774647887324|0\.9635036496350365|0\.8984790706935947|
|2|KNN|0\.8942307692307693|0\.9773794280836534|0\.9545454545454546|0\.8873239436619719|0\.9197080291970803|0\.7697713250294985|
|3|Naive Bayes|0\.9134615384615384|0\.960734101579172|0\.9305555555555556|0\.9436619718309859|0\.9370629370629371|0\.7988230541997952|
|4|Random Forest|0\.9903846153846154|1\.0|1\.0|0\.9859154929577465|0\.9929078014184397|0\.9782218452166099|
|5|XGBoost|0\.9711538461538461|1\.0|1\.0|0\.9577464788732394|0\.9784172661870504|0\.9369814684936246|
