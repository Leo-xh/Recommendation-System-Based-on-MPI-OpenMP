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