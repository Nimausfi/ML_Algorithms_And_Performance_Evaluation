# Supervised Machine Learning Algorithms and Performance Evaluation
\
In supervised learning, we can train the machine using data which is already labeled. That means our data is already tagged with the correct answer.

Supervised learning algorithms learn from labeled training data, and help us to predict outcomes for unforeseen data.


-----------------------------------------------------------------------------------------


The source code shows the utilization of Scikit-learn library to apply the classification methods on the prediction. Nine classification methods are implemented, which are:
\
\
1-	**Decision Tree**
\
2-	**Support Vector Machines**
\
3-	**Stochastic Gradient Descent**
\
4-	**Random Forest**
\
5-	**Gaussian Naive Bayes**
\
6-	**Gradient Boosting**
\
7-	**Logistic Regression**
\
8-	**K-Nearest Neighbor**
\
9-	**AdaBoost**
\
\
Furthermore, performance metrics are utilized to measure the performance of the algorithms, which are accuracy, precision, recall and F1-score. Specifically, accuracy is the percentage of tweets that are predicted correctly, which can be calculated by (TP+TN)/(TP+FP+TN+FN). TP is the number of positive tweets that are predicted as positive, whereas TN is the number of positive tweets that are predicted as negative. FP is the number of negative tweets that are predicted as negative and FN is the negative tweets that are predicted as positive. Precision is the method to measure the percentage of tweets that are actually positive out of all that predicted as positive. Recall is used to measure how many tweets are predicted correctly as positive out of all the positive tweets. F1-Score is a measure of balance between precision and recall. The equation of F1-Score is (2*precision*recall)/ (precision + recall).

