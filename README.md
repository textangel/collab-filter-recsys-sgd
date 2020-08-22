## SGD Matrix Factorization

This script trains a collaborative filtering algorithm, using the algorithm commonly called "SGD Matrix Factorization", originally used to win the Netflix Prize.

Given a set of unique users `u`, a set of unique items `i`, and a sparse list of user-item ratings `r`, and supposing an embedding dimension `embed_size`,
we compute the collaborative filter as follows:

1. Compute a gloal bias `mu`. This is the average rating value (for existing ratings, we do not impute missing ratings). We decrement all of our ratings by the global average `mu`.
2. For each user, maintain a embedding matrix `M_u` of `embed_size` and a bias term `b_u`. Thus the matrix corresponding to users has size `(len(u), embed_size + 1)`.
3. For each item, maintain a embedding matrix `M_i` of `embed_size` and a bias term `b_i`. Thus the matrix corresponding to items has size `(len(i), embed_size + 1)`.

For each user-item pair, the predicted rating is `r = mu + b_u + b_i + M_u M_i`. The associated loss is thus `(r - mu + b_u + b_i + M_u M_i) ^2`.
We use Alternating Least Squares (Two-variable Stochastic Gradient Descent) to optimize this loss.

In addition to this, we have a regularization term, penalizing the absolute values of our parameters from getting too large. 
This is a standard square regularization term over all parameter values, which means that the parameters are adjusted linearly in the gradient updates.

We initialize the biases to each's user's (or item's) average rating minus `mu`. We initialize the embedding matrices uniformly from 0 to `a`, picking `a` such that the dot product of any corresponding rows in `M_u` and `M_i` will equal to `mu`.
