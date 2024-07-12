Loss functions

- how to find a *good* model F that fits the training data ?
  - select F minimizing loss function L on the training data D.

  ![alt text](/.assets/image-1.png)

Linear models.

- An important class of models parametereized by weights W

```
  F(x) = W.X
```

Losses of linear models. what does loss compute ?

- infinite number of possible linear functions
- want to minimize loss

loss computes
step function

descision boundary

- Overfitting.

model fits training data well -> low training error
but does not generalize well to unseen data -> poor test error

complex models with large number of parameters capture not only good patterns (that generalize) but also noisy ones.

on High dimensional Polynomial Model - High prediction error on test data

also fitting to noise or randomness that coming with the pattern ..

overfitting is not good way ... should be pattern not the noice always.

Underfitting

High dimensional data but model we are fitting not sufficiently complex

- model lacks the

Linear Models : regularization.

- Regularization prevents overfitting by penalizing large weights.

![alt text](/.assets/image-2.png)

Bias-variance trade-off.

How do we qunatify and evaluate poor fit?
use error on test sample

- **variance** : variation in predictios of models from different samples : far from each other : overfitting
- **Bias** : difference of their average from true target : far from the ground truth : underfitting

![alt text](/.assets/image-3.png)


