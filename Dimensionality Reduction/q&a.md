## Q&A

**Backlog** : Revise form the recorded

---

### Set 1

- **What is t-SNE and its main functionality ?**

  - stochastic neighbour embedding
  - pairwise similarity probability
  - capture how close to data points are
  - we want to learn y1 ... yn, these are data points present in 2D or 3D.
  - gaussian distribution, heavy tail distribution
  - discriminate different clusters in data point

  - used to convert high dimension embeddings to low level representation which we can visualize
  
- **t-SNE and SNE Difference**

  - t technique vs gaussian distribution

- **what's with SNE ?**

  - not able to get visually separate cluster.

- **how to solve clouding problem in t-SNE ?**

  - solved used heavy-tails

- **why t-SNE use t disribution ?**

  - gaussian distribution present in past, issue that no discriminative cluster provided
  - _**heavy tail distribution**_, removes overcrowding problem

- **how distance measures between clusters in t-SNE? are the quantitative or qualitative ?**

  - more of a way to visualize rather then qunatitively interpret the distance

- **how to get the t-SNE output ?**

- **What is t-SNE ?**

  - additional parameters called degrees of freedom can go from 0 to inf
  - when degree is inf, if 0 it becomes very heavy tail distribution

- **why apply graident descent in t-SNE ?**

  - forward, backward differentiation
  - default algo for most machine learning algorithm XD, whenever there's expression, minimization problem but some exception expectation propagation

- **As there's no loss of data from high dimension to low dimension space ?**

  - main purpose of tSNE is about to visualization, distance is there don't have any meaning

- **how does tSNE works ? and PCA ?**

- **why can't tSNE be used with large dataset**

  - we have to compute the parameters, we want to learn from y1 ... yn

- **how does umap different from tSNE ?**

- **while tSNE known for visualization is there any application where it used out of visualization task ?**

  - just to discriminate cluster in low dimension space
  - tSNE is sensitive to hyperparameters
  - try with different intiallization

- **limitations of tSNE**

  - not for large dataset
  - sensitive to initiallization
  - distance are not meaninfuly, quntatively

- **tSNE in ML Pipeline**

  - only used for visual interpretation
  - not meaningful though

### Set 2

- **Intution behind SVD**

  - method for factorizing metrics
  - able to reduce the rank of the matrix

- **Overview of PCA**

  - more of visualization technique
  - project the metrix along the principal components

- **Different between SVD and PCA**

  - SVD in three part while in PCA to figure out the Prinicipal component analysis

- **how do you know the principal components, and then retain them in PCA ?**

  - eigenvalues of data metrix
  - in high to low value order

  - deciding the threshold

- **how to represent feature of higher dimension in the vector of lower dimension ?**

  - help us project in lower dimension
  - where to visualize better
  - project all the points either x axis or y axis.

- tradeoff of information from high to low dimension

- **ALS and Matrix Factiorization**

  - method of decomposing a metrix into two or more
  - very widely used in recommender system
  - million users, million products, a huge metrix of purchase, a very sparse metrix as well, storing it a huge task
  - decompose let matrix n into two let u and i with dimension `d * n`, `d * m` respectively, now easy to compute

- **how does ALS work ?**

  - user metrix u and item metrix i, we freeze one of these and then we apply gradient descent for second one
  - in second round we freeze item one and apply gradient descent for user metrix
  - we freeze one and optimize other one

- PCA works for linear data, will not work well in non-linear case
- tells us the principal component covers most of the variance of data

- **what are other divergences apart from KL ?? i have heard of wassertian distance is that also common ??**

  Great question! Yes, researchers experiment with various distances - Wasserstein distance, Jeffrey's divergence, f-divergences are some of the widely divergences.

- **is the matrix decomposition is similar to LORA technique in LLMs?**

  Yes, LoRA is Low Rank Adaption technique is also based on matrix decompostion.

skipped ... 30 min in between

- **What is curse of dimensionality in t-SNE and uMap??**

  - _**Crowding distrbution**_
  - riemannian space or metrics, more descriptive

  requires more and more data, whenever dimensionality of a point grows

- **Can you explain about riemannian space**

  You can think our sphere as a 2-dimensional rimeannian space. Say you want to say the distance between Bangalore and Newyork, we wouldn't simply take the coordinates between the two cities and compute the euclidean distance. Rather, you would compute something known as 'Geodesic Distance'. Each Riemannain space is endowed with a Riemannian metric, which induces the concept of distance in that space.

- **when to use L1 and L2 ?**

  - L1 promotes sparsity, pushes value can be to zero
  - L2 compress to more smaller value

  - Feature Selection

- **PCA works only in unsupervised learning ?**

  - we don't care about label
  - it works with both usecases with and without label

- **Challenges to deploy LLMs in Prod**

  - main issue is training LLMs is faster then generation
  - architecture level changes required
  - caching internals, destill LLMs

- **Avoding Overlapping topics in LDA ?**

  - can use topic hyperparameter, unsure

embedding some sort of dimensionality reduction technique

- **different between LDA and LSI**

  - LSI, nothing but PCA means deterministic, while LDA is probabilistic
  - LDA more later version, peform well then compare to LSI

- **what's AV testing ?**

  - randomized control train
  - divide user into two group, control group, group A ( existing feature as it ) and second group, group B ( new feature ), and then compare between them

- **what metrics we used to evaluate effectiveness of dimensionality reduction techinque ?**

  - exact metrics depends upon ..
  - mutual information, capture how effective the technique is
  - InterCluster distance and IntraCluster distance

**transformers** and **mamba** : more of representation learning technique, not dimensionality learning

- **In terms of recommendation systems, what can be some suitable evaluation metrics (because there is absence of ground truth)to test and compare different recommendation models/techniques.**

  In recommender systems, usually metrics like precision@K, NDCG are used. The idea is to look at the next set of items user has purchased/viewed and see if the item we are recommending falls in that set

domain and vision of NLP

- **difference between feature extraction and feature reduction**

  - Feature selection (extraction) is about looking at which features contributthe most to predicting my target without hurting the performance of my model, 
  - while feature reduction (dimensionality reduction) is about modifying my current feature space into a lower dimensional space so that the variance in my data is preserved. 
  - Interpretability will stay when you are doing feature selection while dimensionality reduction hurts interpretability of your model.

- **emerging trends in dimensionality reduction**

  linear, euclidean and then to non-linear, riemannian space

- **KL divergance**

  - widely used, because like mean_squared_error, cross_entroy_loss, it observes that minimizing divergance lead to these loss or error
  - mother of all the losses in ml we have
  - convex losses, where on finding local optima there's for sure there be global optima
  - logistic regressoin is a convex loss

- **What are the limitations of linear dimensionality reduction techniques in capturing complex structures within data, and how do non-linear techniques address these limitations?**

  Linear dimensionality reduction techniques will not help you capture non-linear correlation in the data. Which is why we have non linear techniques like kernel PCA. Here, we use kernel functions to map our data into a higher dimensional space, where the pattern in your data resembles the linear structure required for PCA

- **I am working on a dataset that has both numerical and categorical features because of which i am having issues with finding the most important features, can you guide as to what methods can be used in such dataset?**

  While PCA works for continuous (numerical) data, you can convert your categorical features into one hot encoding (or any other encoding) and apply PCA on your data

we can use neural networks for dimensionality reduction

- **quantum pca**

- **how do scaling choice result of pca ?**

- **some good refernce books**

  - chirtoper bishop, deep neural networks
  - probabilistic machine learning, cilian murphy

- **sammon mapping**

- **autoencoders**

  - autoencoders are pretty old
  - issue that its deterministic
  - not always gurantees that the information we get form bottlenech is interpretable.

- **Explain how SVD is applied in recommender systems. What are the advantages of using SVD in this context ?**

  https://analyticsindiamag.com/singular-value-decomposition-svd-application-recommender-system/
