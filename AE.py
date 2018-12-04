import tensorflow as tf
import time
import numpy as np
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class AutoRec():
    def __init__(self,sess,args,
                      num_users,num_items,
                      train_R, train_mask_R, test_R, test_mask_R):

        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.train_R = train_R.T
        self.train_mask_R = train_mask_R.T
        self.test_R = test_R.T
        self.test_mask_R = test_mask_R.T

        self.num_batch = int(math.ceil(self.num_items / float(self.args.batch_size)))

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_step = self.args.decay_epoch_step * self.num_batch
        self.learning_rate = tf.train.exponential_decay(self.args.base_lr, self.global_step, self.decay_step, 0.9, staircase=True)

        self.MAE_min  = 9999.0
        self.RMSE_min = 9999.0
        self.train_loss_list = []
        self.test_loss_list  = []
        self.test_rmse_list  = []
        self.test_mae_list   = []
        self.test_recall_list= []

    def run(self):
        self.build_model()
        self.prepare_data()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.args.train_epoch):
            self.train_model(epoch)
            self.test_model(epoch)

    def build_model(self):

        self.input_R      = tf.placeholder(dtype=tf.float32, shape=[None, self.num_users], name="input_R")
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_users], name="input_mask_R")

        V   = tf.get_variable(name="V",  initializer=tf.truncated_normal(shape=[self.num_users,       self.args.hidden_dim], mean=0, stddev=0.03), dtype=tf.float32)
        W   = tf.get_variable(name="W",  initializer=tf.truncated_normal(shape=[self.args.hidden_dim, self.num_users],       mean=0, stddev=0.03), dtype=tf.float32)

        mu  = tf.get_variable(name="mu", initializer=tf.zeros(shape=self.args.hidden_dim), dtype=tf.float32)
        b   = tf.get_variable(name="b",  initializer=tf.zeros(shape=self.num_users),       dtype=tf.float32)

        Encoder = tf.matmul(self.input_R,V) + mu
        self.Encoder = tf.nn.sigmoid(Encoder)

        Decoder = tf.matmul(self.Encoder,W) + b
        self.Decoder = tf.identity(Decoder)

        reconstruction_loss = tf.multiply((self.input_R - self.Decoder) , self.input_mask_R)
        self.reconstruct_loss = tf.reduce_sum(tf.square(reconstruction_loss))

        regularization_loss = tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(V))
        self.regularization_loss = 0.5 * self.args.lambda_value * regularization_loss

        self.loss = self.reconstruct_loss + self.regularization_loss

        if self.args.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.args.optimizer_method == "GD":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("Optimizer Key ERROR")

        self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

    def prepare_data(self):

        random_index = np.random.permutation(self.num_items)
        self.train_index_list = []
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_idx = random_index[i * self.args.batch_size:]
            elif i < self.num_batch - 1:
                batch_idx = random_index[i * self.args.batch_size : (i+1) * self.args.batch_size]
            self.train_index_list.append(batch_idx)

    def get_batch(self,i):

        index = self.train_index_list[i]
        data  = self.train_R[index,:]
        mask  = self.train_mask_R[index,:]
        return data, mask

    def train_model(self,itr):
        start_time = time.time()

        loss = 0.0
        for i in range(self.num_batch):
            train_R, train_mask_R = self.get_batch(i)

            _, l = self.sess.run(
                [self.optimizer, self.loss],
                feed_dict={self.input_R: train_R,
                           self.input_mask_R: train_mask_R})

            loss = loss + l
        self.train_loss_list.append(loss)

        if (itr+1) % self.args.display_step == 0:
            print ("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(loss),
               "Elapsed time : %d sec" % (time.time() - start_time))

    def rmse(self, prediction, test_R, test_mask_R):

        Estimated_R = prediction.clip(min=1, max=5)
        pre_numerator = np.multiply((Estimated_R - test_R), test_mask_R)
        numerator = np.sum(np.square(pre_numerator))
        denominator = int(np.sum(test_mask_R))
        RMSE = np.sqrt(numerator / float(denominator))

        return RMSE

    def mae(self, prediction, test_R, test_mask_R):
        Estimated_R = prediction.clip(min=1, max=5)
        pre_numerator = np.multiply((Estimated_R - test_R), test_mask_R)
        numerator_mae = np.sum(np.abs(pre_numerator))
        denominator = int(np.sum(test_mask_R))
        MAE = numerator_mae / float(denominator)

        return MAE

    def recall(self, prediction, test_R, test_mask_R):
        recall_sum = 0.0
        recall_num = 0

        prediction = prediction.T
        test_R = test_R.T
        test_mask_R = test_mask_R.T

        for user in range(len(prediction)):
            num_item_for_user = np.sum(test_mask_R[user])
            recall_index = np.argsort(prediction[user])[-300:]
            hits = np.sum(test_mask_R[user][recall_index])
            if (num_item_for_user != 0):
                recall = hits / float(num_item_for_user)

                recall_sum = recall_sum + recall
                recall_num = recall_num + 1

        recall_average = recall_sum / float(recall_num)

        return recall_average

    def test_model(self, n):

        Decoder = self.sess.run(
            self.Decoder,
            feed_dict={self.input_R:      self.train_R,
                       self.input_mask_R: self.test_mask_R})
        self.prediction = Decoder

        loss = self.sess.run(
            self.loss,
            feed_dict={self.input_R:      self.test_R,
                       self.input_mask_R: self.test_mask_R})

        self.test_loss_list.append(loss)

        recall = self.recall(self.prediction, self.test_R, self.test_mask_R)
        self.test_recall_list.append(recall)

        RMSE = self.rmse(self.prediction, self.test_R, self.test_mask_R)
        if RMSE < self.RMSE_min:
            self.RMSE_min = RMSE
        self.test_rmse_list.append(RMSE)

        MAE = self.mae(self.prediction, self.test_R, self.test_mask_R)
        if MAE < self.MAE_min:
            self.MAE_min = MAE
        self.test_mae_list.append(MAE)

        print("test RMSE = {:.5f}\t".format(RMSE), "MAE = {:.5f}\t".format(MAE), "recall = {:.5f}\t".format(recall), "cost = {:.5f}".format(loss))

    def make_recode(self, result_path):
        print("make recode...")

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        basic_info = result_path + "/basic_info.txt"
        train_record = result_path + "/train_record.txt"
        test_record = result_path + "/test_record.txt"

        with open(train_record,'w') as f:
            f.write(str("loss:"))
            f.write('\t')
            for itr in range(len(self.train_loss_list)):
                f.write(str(self.train_loss_list[itr]))
                f.write('\n')
            f.write('\n')

        with open(test_record,'w') as g:
            g.write(str("loss"))
            g.write('\t')
            g.write(str("recall"))
            g.write('\t')
            g.write(str("RMSE"))
            g.write('\t')
            g.write(str("MAE"))
            g.write('\n')

            for itr in range(len(self.test_loss_list)):
                g.write(str(self.test_loss_list[itr]))
                g.write('\t')
                g.write(str(self.test_recall_list[itr]))
                g.write('\t')
                g.write(str(self.test_rmse_list[itr]))
                g.write('\t')
                g.write(str(self.test_mae_list[itr]))
                g.write('\t')
                g.write('\n')

            g.write(str("Mininum RMSE:"))
            g.write(str(self.RMSE_min))
            g.write('\n')

            g.write(str("Mininum MAE:"))
            g.write(str(self.MAE_min))
            g.write('\n')

        with open(basic_info,'w') as h:

            h.write("icf\n")

            h.write('hidden_dim = '       + str(self.args.hidden_dim) + "\n")
            h.write('lambda_value = '     + str(self.args.lambda_value) + "\n")
            h.write('train_epoch = '      + str(self.args.train_epoch) + "\n")
            h.write('batch_size = '       + str(self.args.batch_size) + "\n")
            h.write('optimizer_method = ' + str(self.args.optimizer_method) + "\n")
            h.write('base_lr = '          + str(self.args.base_lr) + "\n")
            h.write('decay_epoch_step = ' + str(self.args.decay_epoch_step) + "\n")
            h.write('random_seed = '      + str(self.args.random_seed) + "\n")
            h.write('display_step = '     + str(self.args.display_step) + "\n")
            h.write('ratio = '            + str(self.args.ratio) + "\n")
            h.write('repeat_number = '    + str(self.args.repeat_number) + "\n")


        Train = plt.plot(self.train_loss_list,label='Train')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(result_path + "/train_loss.png")
        plt.clf()

        Test = plt.plot(self.test_loss_list,label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(result_path+"/test_loss.png")
        plt.clf()

        Test = plt.plot(self.test_rmse_list,label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(result_path+"/RMSE.png")
        plt.clf()

        Test = plt.plot(self.test_mae_list, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(result_path + "/MAE.png")
        plt.clf()

        Test = plt.plot(self.test_recall_list, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('recall')
        plt.legend()
        plt.savefig(result_path + "/recall.png")
        plt.clf()

        with open(result_path + "/prediction.dat", 'w') as f:
            for user in range(len(self.prediction)):
                for item in range(len(self.prediction[user])):
                    rating = self.prediction[user][item]
                    f.write(str(int(user)) + "::" + str(int(item)) + "::" + str(int(rating)))
                    f.write('\n')
