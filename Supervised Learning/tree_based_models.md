Descision Trees

![alt text](image-13.png)
![alt text](image-14.png)

Decision Trees Algorithm

Recursive partitioning algorithm
- greedily select best split variable (lets say xi) and best split criterion (lets say si)
- push training examples into left and right nodes (xi < si -> left and xi > si -> right)
- further divide each of these two nodes (left and right) in similar manner.

- greedy divide and conquer algorithm

![alt text](image-15.png)

Splitting example

![alt text](image-16.png)

node impurity : gini impurity criteria

combined gini = weighted average = ...

lowest gini value -> best splitting criteria

- Node Impurity

maximum gini impurity there's exactly equal label instance in binary case.

high gini value is bad - 

- Gini Impurity Index
- Entropy measure

![alt text](image-18.png)

Descision Trees example

What depths should we stop or should we stop anywhere ? 

deeper the tree is more chances of overfitting the training data.

smaller the tree best it would be

![alt text](image-19.png)

Decision tree pruning can be used to remove subtree that are non-critical (e.g. subtree that have the least information gain), thus reducing overfitting.

Ensemble Learning

![alt text](image-20.png)

- Ensemble techniques:
  - Bagging :
    - ensemble technique where bootstrap samples (random samples with replacement) of training data are created.
    - multiple models (having low bias and high variance) are trained on each bootstrap sample and then the prediction from these models can be combined (e.g. by voting or averaging.)
    - The final model has low variance due to aggregation of multiple model predictions.

  - Boosting : 
    - an ensemble technique where multiple weak learners are trained in iterative fashion.
    - each learner tries to reduce error of previous models.

Bagging (Bootstrap-aggregation)

![alt text](image-21.png)

### Random Forest
Bagging of Random Decision Trees

![alt text](image-22.png)

Boosting & AdaBoost

![alt text](image-23.png)
![alt text](image-24.png)

## Gradient Boosting

![alt text](image-25.png)
![alt text](image-26.png)
![alt text](image-27.png)
![alt text](image-28.png)
![alt text](image-29.png)
![alt text](image-30.png)
