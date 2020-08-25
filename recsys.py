import numpy as np, pandas as pd
import pickle
# orders_raw= pd.read_csv('instacart/orders.csv')
# orders = orders_raw[orders_raw.eval_set=='prior'][['order_id','user_id']]
# products = pd.read_csv('instacart/order_products__prior.csv')[['order_id', 'product_id']]

# #32 million rows
# user_item = orders.merge(products, on='order_id')[['user_id', 'product_id']]
# user_item_counts = pd.DataFrame(user_item
#                 .groupby(['user_id', 'product_id'],as_index=False).size())
#                 .reset_index().rename({0:'count'}, axis=1)


# Recommendation with User, Item Co-Occurence or Rating matrix (Matrix Factorization)
# Recommendation with a lot of geatures (FM)
# Recommendation with Categorical Features (FFM)

class CollabFilter(object):
    def __init__(self, user_item, user_attributes=None, item_attributes=None, latent_size=20, reg=0.2, counts=False, lr = 0.01,
                 neg_sample_ratio=1.0):
        """
        `user_item` is a pandas dataframe of user-product purchase counts with at least three columns:
        1. `user_id`
        2. `product_id`
        3. `count` - the number of times the user purchased the product if `counts` is True, else
            the score given to the product. If `counts` is true, we do not treat the 
            value as a popularity score, but instead divide the count by the average count across all users, and treat it as a proportion.
        """
        if user_item is None:
            user_item = pd.DataFrame([[0, 0, 0]], columns=['user', 'item', 'score'])
        self.users = np.array(sorted(user_item.user.unique()))
        self.id2user = dict(enumerate(list(self.users)))
        self.user2id = {v:k for k,v in self.id2user.items()}
        self.items = np.array(sorted(user_item.item.unique()))
        self.id2item = dict(enumerate(list(self.items)))
        self.item2id = {v:k for k,v in self.id2item.items()}
        self.user_item_counts = user_item.groupby('user').count().sort_values('item').reset_index()
        self.latent_size = latent_size
        self.matrix = user_item
        self.matrix['item'] = self.matrix['item'].map(lambda x: self.item2id[x])
        self.matrix['user'] = self.matrix['user'].map(lambda x: self.user2id[x])
        self.neg_sample_ratio = neg_sample_ratio

        
        # Overall Average
        self.mu = np.mean(self.matrix['score']) / (self.neg_sample_ratio + 1)

        # User Bias
        self.matrix.sort_values('user')
        self.ub = np.array(self.matrix.groupby('user').agg({'score':'mean'})['score']) / (self.neg_sample_ratio + 1) - self.mu
        
        # Item Bias
        self.matrix.sort_values('item')
        self.ib = np.array(self.matrix.groupby('item').agg({'score':'mean'})['score']) / (self.neg_sample_ratio + 1) - self.mu

        # each of the dimensions have to have square values on average `self.mu/self.latent_size`.
        # Taking the integral 1/3 a^3, dividing by 2 and then solving, we obtain the below formula for the range of our random init values.
        rand_max = ( ( self.mu/self.latent_size ) * 6 ) ** (1/3)
        # Item Embedding
        self.ie = np.random.uniform(0, rand_max, (len(self.items), latent_size))
        # User Embedding
        self.ue = np.random.uniform(0, rand_max, (len(self.items), latent_size))

        self.train, self.dev, self.test = np.split(self.matrix.sample(frac=1), [int(.70*len(self.matrix)), int(.85*len(self.matrix))])

        self.lr = lr
        self.reg = reg

        if user_attributes is not None:
            self.user_attributes = user_attributes
        if item_attributes is not None:
            self.item_attributes = item_attributes

    def train_item_item_cf(self, batch_sz=512, num_epochs=30):
        train_tmp = self.train
        if self.neg_sample_ratio > 0:
            neg_sample = self.get_negative_sample(self.train)
            train_tmp = self.train.append(neg_sample)

        for epoch in range(num_epochs):
            train = np.array(train_tmp.sample(frac=1)[['user', 'item', 'score']])
            loss, early_stopped = self.train_epoch(train, batch_sz, epoch)
            if early_stopped:
                print("Early Stopped!")
                break

    # Train using Stochastic Gradient Descent
    def get_negative_sample(self, base):
        sample_ratio = self.neg_sample_ratio
        neg_sample_train = pd.DataFrame(np.random.choice(list(self.id2user.keys()), int(sample_ratio * len(base))), columns=["user"])
        neg_sample_train['item'] = pd.Series(np.random.choice(list(self.id2item.keys()), int(sample_ratio * len(base))))
        neg_sample_train['score'] = 0
        neg_sample_train = neg_sample_train[['user', 'item', 'score']]
        return neg_sample_train

    def train_epoch(self, train, batch_sz=128, epoch=0):
        num_rows = train.shape[0]
        count = 0
        loss = 0
        last_40_loss = [10e9]
        for batch_ix in range(0, num_rows, batch_sz):
            batch = train[batch_ix:batch_ix + batch_sz, :]
            u = batch[:,0]
            i = batch[:,1]
            s = batch[:,2]
            tmp_u = self.ue[u] #"(bt, 20)"
            tmp_i = self.ie[i] #"(bt, 20)"
            diff = s - (np.sum(tmp_u * tmp_i, axis=1) + self.ub[u] + self.ib[i] + self.mu)

            """
            We cannot use decrement the indexed array (self.ub[u] -= gradient)" 
            this because python will NOT decrement all occurences of the index, it will only perform the last decrement.

            >>> b =np.array( [0.,10.,20.,30.])
            >>> a = np.array([3,3,1])
            # b[a] array([30., 30., 10.])
            >>> b[a] += np.array([0.1, 0.01, 0.001])
            >>> b
            array([ 0.   , 10.001, 20.   , 30.01 ])
            
            Note that not every occurence was added. This is because python loops through the array to perform the add.
            Each of the "30" above references the original b[3]. When this is modified, only the last one persists.
            
            This is why `np.add.at` was invented - "Performs unbuffered in place operation on operand
            ‘a’ for elements specified by ‘indices’. For addition ufunc, this method is equivalent to 
            `a[indices] += b`, except that results are accumulated for elements that are indexed more than once.
            https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
            """
            grad_ub = self.lr * 2 * (-1 * diff + self.reg * self.ub[u])
            np.add.at(self.ub, u, -1 * grad_ub)
            grad_ib = self.lr * 2 * (-1 * diff + self.reg * self.ib[i])
            np.add.at(self.ib, i, -1 * grad_ib)

            grad_ue = self.lr * 2 * (np.expand_dims(diff,-1) * (-1 * tmp_i) + self.reg * tmp_u)
            np.add.at(self.ue, u, -1 * grad_ue)
            grad_ie = self.lr * 2 * (np.expand_dims(diff, -1) * (-1 * tmp_u) + self.reg * tmp_i)
            np.add.at(self.ie, i, -1 * grad_ie)

            count += 1
            if count % 10 == 0:
                loss = self.compute_dev_loss()
                print(f"Ep {epoch}: {count * batch_sz} out of {num_rows} processed. Loss = {loss / 100}")
                if loss >= np.mean(last_40_loss):
                    print("Early Stop!")
                    return loss, True
                if len(last_40_loss) >= 40:
                    last_40_loss = last_40_loss[1:]
                last_40_loss.append(loss)
        return loss, False

    def compute_dev_loss(self):
        dev_temp = self.dev
        if self.neg_sample_ratio > 0:
            neg_sample_dev = self.get_negative_sample(self.dev)
            dev_temp = self.dev.append(neg_sample_dev)
        dev = np.array(dev_temp)
        u_bias = self.ub[dev[:,0]]
        i_bias = self.ib[dev[:,1]]
        cross_term = np.sum(self.ue[dev[:,0]] * self.ie[dev[:,1]], axis=1)
        loss =  np.sum( (dev_temp['score'] - (self.mu + u_bias + i_bias + cross_term)) ** 2) + \
            self.reg * (np.sum((self.ue) ** 2) + np.sum((self.ie) ** 2) + np.sum((self.ub) ** 2) + np.sum((self.ib) ** 2))
        return loss

    def save_model(self, filename):
        f = open(filename, 'wb+')
        pickle.dump(self.__dict__, f)
        f.close()

    def load_model(self, filename):
        f = open(filename, 'rb')
        from_file = pickle.load(f)
        f.close()
        self.__dict__.update(from_file)

    def get_recommendations_user(self, user_id, repeat_old=False, verbose=False, n=20):
        u = self.user2id[user_id]
        old_items_ = self.matrix[self.matrix['user'] == u]
        old_items_['id'] = old_items_['item'].map(lambda x: self.id2item[x])

        scores = np.sum(self.ue[u] * self.ie, axis=1) + self.ub[u] + self.ib + self.mu
        topn_ids = np.argpartition(-scores, n)[:n]
        topn = [self.id2item[a] for a in topn_ids]
        topn_scores = -np.partition(-scores, n)[:n]
        recommendations = pd.DataFrame(topn_ids, columns=['index'])
        recommendations['id'] = topn
        recommendations['score'] = topn_scores

        if verbose:
            print(f"User {u} has previously consumed: ")
            if self.item_attributes is not None:
                old_items_ = old_items_.merge(self.item_attributes, on='id')
                for ix, row in old_items_.iterrows():
                    print(f" - {row['name']} {float(row['score']):.2f}")
            else:
                print(f" - Items: {old_items}")
            print(f"We recommend for user {u}: ")

            if self.item_attributes is not None:
                recommendations = recommendations.merge(self.item_attributes, on='id')
                for ix, row in recommendations.iterrows():
                    print(f" - {row['name']} {float(row['score']):.2f}")
            else:
                print(f" - Items: {topn}")

if __name__ == '__main__':
    # user_book_raw= pd.read_csv('book-crossing/BX-Book-Ratings.csv',engine='python', sep=";", error_bad_lines=False )
    # user_book_raw = user_book_raw[user_book_raw["Book-Rating"]>0]
    # user_book_raw = user_book_raw.set_axis(["user", "item", "score"], axis=1, inplace=False)
    #
    #
    # item_attributes = pd.read_csv('book-crossing/BX-Books.csv',engine='python', sep=";", error_bad_lines=False )
    # item_attributes = item_attributes[["ISBN", "Book-Title", "Book-Author"]]
    # item_attributes['name'] = item_attributes["Book-Title"] + " (" + item_attributes["Book-Author"] + ")"
    # item_attributes['id'] = item_attributes['ISBN']
    # item_attributes = item_attributes[['id', 'name']]
    # item_attributes = item_attributes.set_axis(['id', 'name'], axis=1, inplace=False)
    #
    # user_attributes = pd.read_csv('book-crossing/BX-Users.csv', engine='python', sep=";", error_bad_lines=False)
    # user_attributes['name'] = "Loc: " + user_attributes["Location"] + "; Age:" + user_attributes["Age"].astype(str)
    # user_attributes['id'] = user_attributes['User-ID']
    # user_attributes = user_attributes[['id', 'name']]
    # user_attributes = user_attributes.set_axis(['id', 'name'], axis=1, inplace=False)
    # cf = CollabFilter(user_book_raw, user_attributes=user_attributes, item_attributes=item_attributes,
    #                   latent_size=40, neg_sample_ratio=10.0)
    #
    # # cf = CollabFilter(user_book_raw)
    # cf.train_item_item_cf(batch_sz=512, num_epochs=15)
    # cf.save_model("models/collab_filter_model_3.model")
    # del cf

    cf = CollabFilter(None)
    cf.load_model("models/collab_filter_model_3.model")
    cf.get_recommendations_user(107784, verbose=True)