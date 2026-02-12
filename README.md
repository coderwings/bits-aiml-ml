# Repo for ML assignment.

**Problem statement**: Early Stage Diabetes Risk Prediction using following 6 classification models.
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

**Dataset description** : https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
This dataset contains the sign and symptpom data of newly diabetic or would be diabetic patient.
Dataset Shape: (520, 17)

**Comparison Table with the evaluation metrics for all 6 models**

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|
|Logistic Regression|0\.9327|0\.9797|0\.9332|0\.9327|0\.9328|0\.8587|
|Decision Tree|0\.9904|0\.9922|0\.9906|0\.9904|0\.9904|0\.9800|
|kNN|0\.9442|0\.9925|0\.9470|0\.9442|0\.9446|0\.8861|
|Naive Bayes|0\.9000|0\.9592|0\.9005|0\.9000|0\.9002|0\.7896|
|Random Forest (Ensemble)|0\.9981|1\.0|0\.9981|0\.9981|0\.9981|0\.9959|
|XGBoost (Ensemble)|0\.9942|1\.0|0\.9943|0\.9942|0\.9942|0\.9879|

**Obervations about model performance** :
|ML Model Name|Observation about model performance|
|---|---|
|Logistic Regression|Provides a reliable baseline with balanced precision and recall. However, it is significantly outperformed by tree-based models, suggesting non-linear relationships in the data that a linear model cannot fully capture.|
|Decision Tree|Shows exceptional performance for a single (non-ensemble) learner, with an accuracy of $99.04\%$ and a high MCC of $0.9800$. This indicates that the dataset likely has clear hierarchical decision boundaries.|
|kNN|Exhibits strong discriminative ability with a very high AUC of $0.9925$. While it performs better than Naive Bayes and Logistic Regression, it falls behind the tree-based architectures in overall classification accuracy.|
|Naive Bayes|The lowest performing model across all metrics. While an accuracy of $90\%$ is decent, its MCC of $0.7896$ is the lowest in the group, suggesting it is the least robust model for this specific dataset.|
|Random Forest (Ensemble)|It achieves the highest scores across all metrics, including a perfect AUC ($1.0$) and a near-perfect MCC ($0.9959$), making it the most reliable model for this data.|
|XGBoost (Ensemble)|Like Random Forest, it achieves a perfect AUC ($1.0$) and maintains extremely high precision and recall, demonstrating the superior power of ensemble boosting techniques for this task.|

