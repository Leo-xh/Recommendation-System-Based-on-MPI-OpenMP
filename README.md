# Recommendation System Based on MPI&OpenMP
## Dataset
Using movielen lates small, ratings.csv(recording user behaviors) and movies.csv(mapping movieId to movieName).
## Algorithm - ItemCF
We have steps:

0. calculate the vitality.
     $$
         V_u
     $$

1. calculate the similarity of the items
    $$
    w_{ij} = \frac{\sum_{u \in N(i) \cap N(j)}{\frac{1}{log(1+|V(u)|)}}}{\sqrt{|N(i)||N(j)|}}
    $$

2. normalization of weights
    $$
    w_{ij}' = \frac{w_{ij}}{\max_j w_{ij}}
    $$

3. calculate the preference of users
    $$
    p_{ij} = \sum_{i \in N(i)\cap S(j,k)}w_{ji}r_{ui}
    $$

## Another choice of algorithm - Collaborative Filtering on Ratings
1. the similarity of items - adjust cosine similarity
    $$
    w_{ij} = \frac{\sum_{u\in U}(r_{ui}-\bar{r_{i}})\times(r_{uj}-\bar{r_{j}})}{\sqrt{\sum_{u\in U}(r_{ui}-\bar{r_{i}})^2\sum_{u\in U}(r_{uj}-\bar{r_{j}})^2}}
    $$
2. predict the rating from user u to item i
    $$
    \hat{r_{ui}} = \bar{r_{i}} + \frac{\sum_{j\in S(i,K) \cap N(u)}w_{ij}(r_{uj}-\bar{r_i})}{\sum_{j\in S(i,K) \cap N(u)}|w_{ij}|}
    $$

## Model merge
Model merge is a regress problem, here I choose a simple method that the models are added by weights, and the weights come from the least square method.
1. how to merge?
    $$
    \hat{r} = \sum_{k=1}^{K}\alpha_k\hat{r}^{(k)}
    $$
2. how to determine the weights?
    - split the train set to A_1 and A_2, so we can use least square method to calculate the weights.
## Evaluation
Split the data set to train set and test set.
## Libraries
Using MPI to distribute jobs and OpenMp to parallelize the computation.

### MPI
1. Distribute by keys of items in step 1 and 2, each node(except for the master) calculate the similarity of sizeOfItem / numOfNodes items with all other items. Send a matrix of weights to the master. For the master, when data received, concate the data.
2. The weight file is too big, so the calculation of preference won't be distributed.

### OpenMP
Optimize the loop in calculating the neighbor of items and the conjunction of them, as well as the preference of items in the nodes.
Optimize the loop in calculating the weights in the master node.