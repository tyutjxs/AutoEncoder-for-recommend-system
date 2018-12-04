from data import DataHelper
from AE import AutoRec
from argument import Argument
import tensorflow as tf
import math
import time

args = Argument()

tf.set_random_seed(args.random_seed)

data_name = 'ml-1m'

data_path = "./data/%s" % data_name

datahelper = DataHelper()
datahelper.read(data_path)

train_R, train_mask_R, test_R, test_mask_R = datahelper.split(args.ratio)
num_users = datahelper.get_num_user()
num_items = datahelper.get_num_item()

for _ in range(args.repeat_number):
    tf.reset_default_graph()

    with tf.Session() as sess:
        autoRec = AutoRec(sess,args,
                          num_users,num_items,
                          train_R, train_mask_R, test_R, test_mask_R)
        autoRec.run()

        result_path = ".result/" + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

        autoRec.make_recode(result_path)



