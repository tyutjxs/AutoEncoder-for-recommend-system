import numpy as np


class DataHelper:

    def __init__(self):
        print("initialize datahelper object")

        self.dataset_path = ""
        self.num_user = 0
        self.num_item = 0
        self.ratio_train_test = 0

    def dataset_preprocess(self):
        print("preprocess dataset")

        data_root_path = self.dataset_path
        if "ml-100k" in self.dataset_path:
            self.dataset_path = self.dataset_path + "/u.data"
            self.splitter = '\t'
        elif "ml-1m" in self.dataset_path:
            self.dataset_path = self.dataset_path + "/ratings.dat"
            self.splitter = '::'

        self.users = set()
        self.items = set()

        fp = open(self.dataset_path)
        lines = fp.readlines()
        for line in lines:
            userID, itemID, rating, timestamp = line.split(self.splitter)
            userID = int(userID)
            itemID = int(itemID)
            self.users.add(userID)
            self.items.add(itemID)

        self.num_user = int(len(self.users))
        self.num_item = int(len(self.items))

        assert (self.num_user == 6040)
        assert (self.num_item == 3706)

        self.index_map = {}
        fp = open(data_root_path + "/index_map.dat")
        lines = fp.readlines()
        for line in lines:
            raw_index, map_index = line.split('::')
            self.index_map[int(raw_index)] = int(map_index)

        print("num_user: {:d}\t num_item: {:d}".format(self.num_user, self.num_item))

    def read(self, path):
        print("read data")

        self.dataset_path = path

        self.dataset_preprocess()

        self.rating_matrix = np.zeros((self.num_user, self.num_item))
        self.rating_mask   = np.zeros((self.num_user, self.num_item))

        fp = open(self.dataset_path)
        lines = fp.readlines()
        for line in lines:
            userID, itemID, rating, timestamp = line.split(self.splitter)
            userID = int(userID) - 1
            itemID = int(self.index_map[int(itemID)]) - 1    # item 索引映射
            self.rating_matrix[userID][itemID] = rating
            self.rating_mask[userID][itemID]   = 1

    def split(self, ratio):
        print("split raw ratings into train_set and test_set")

        self.train_rating_matrix = np.zeros((self.num_user, self.num_item))
        self.train_rating_mask   = np.zeros((self.num_user, self.num_item))
        self.test_rating_matrix  = np.zeros((self.num_user, self.num_item))
        self.test_rating_mask    = np.zeros((self.num_user, self.num_item))


        for user in range(len(self.rating_matrix)):
            items_ratings = self.rating_matrix[user]
            num_ratings = int(np.sum(self.rating_mask[user]))
            ratings_index = np.argsort(self.rating_mask[user])[-num_ratings:]
            ratings_index_random = np.random.permutation(ratings_index)

            train_rating_index = ratings_index_random[:int(num_ratings * ratio)]
            test_rating_index  = ratings_index_random[int(num_ratings * ratio):]

            self.train_rating_matrix[user][train_rating_index] = self.rating_matrix[user][train_rating_index]
            self.train_rating_mask[user][train_rating_index]   = self.rating_mask[user][train_rating_index]
            self.test_rating_matrix[user][test_rating_index]   = self.rating_matrix[user][test_rating_index]
            self.test_rating_mask[user][test_rating_index]     = self.rating_mask[user][test_rating_index]

        return self.train_rating_matrix, self.train_rating_mask, \
               self.test_rating_matrix, self.test_rating_mask

    def get_num_user(self):
        return self.num_user

    def get_num_item(self):
        return self.num_item