import numpy as np


class DataHelper:

    def __init__(self):
        print("initialize datahelper object")

        self.dataset_path = ""
        self.num_user = 0
        self.num_item = 0
        self.ratio_train_test = 0

    def dataset_preprocess(self, path):
        print("preprocess dataset")

        self.splitter = '::'

        self.users = set()
        self.items = set()

        fp = open(path)
        lines = fp.readlines()
        for line in lines:
            userID, itemID, rating, timestamp = line.split(self.splitter)
            userID = int(userID)
            itemID = int(itemID)
            self.users.add(userID)
            self.items.add(itemID)

        self.num_user = int(max(self.users))
        self.num_item = int(max(self.items))

        self.real_num_user = len(self.users)
        self.real_num_item = len(self.items)

        print("num_user: {:d}\t num_item: {:d}\t real_num_user: {:d}\t real_num_item: {:d}".format(self.num_user, self.num_item, self.real_num_user, self.real_num_item))

        self.missing_items = []
        for item in range(self.num_item):
            item = item + 1
            if(item not in self.items):
                self.missing_items.append(item)

        with open("missing_items.dat", 'w') as f:
            for item in self.missing_items:
                f.write(str(item))
                f.write('\n')

        self.result_map = {}
        num_missing = 0
        for item in range(self.num_item):
            item = item + 1
            if(item in self.missing_items):
                num_missing = num_missing + 1
                continue
            self.result_map[item] = item - num_missing

        assert (len(self.result_map)==self.real_num_item)

        with open("index_map.dat",'w') as f:
            for raw_indx, map_index in self.result_map.items():
                f.write(str(raw_indx) + "::" + str(map_index))
                f.write("\n")



    def read(self, path):
        print("read data")

        self.dataset_path = path

        self.dataset_preprocess(self.dataset_path)

        self.rating_matrix = np.zeros((self.num_user, self.num_item))
        self.rating_mask   = np.zeros((self.num_user, self.num_item))

        fp = open(self.dataset_path)
        lines = fp.readlines()
        for line in lines:
            userID, itemID, rating, timestamp = line.split(self.splitter)
            userID = int(userID) - 1
            itemID = int(itemID) - 1
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


data = DataHelper()
path = "./ratings.dat"
data.dataset_preprocess(path)
