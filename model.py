import tensorflow as tf
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')


class PsRMFNN:
    # Initialize the class
    def __init__(self, inputs, layers_e, layers_n, layers_linear, labels, batch_size, lr, y_low):

        # initialized fields for model points
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.learning_rate = lr
        self.y_low = y_low

        self.layers_e = layers_e
        self.layers_n = layers_n
        self.layers_linear = layers_linear

        self.weights_e, self.biases_e = self.initialize_NN(self.layers_e)
        self.weights_n, self.biases_n = self.initialize_NN(self.layers_n)
        self.weights_linear, self.biases_linear = self.initialize_NN(self.layers_linear)
        # tf.placeholder
        # self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.inputs_tf = tf.placeholder(tf.float32)
        self.labels_tf = tf.placeholder(tf.float32)
        self.y_low_tf = tf.placeholder(tf.float32)

        # Initialize parameters
        self.alpha_1 = tf.Variable([0.0], dtype=tf.float32)
        self.alpha_2 = tf.Variable([0.0], dtype=tf.float32)

        # tf Graphs
        self.low_pred1, self.low_pred = self.neural_net(self.inputs_tf, self.weights_e, self.biases_e)
        self.labels_pred = self.neural_net1(self.inputs_tf, self.weights_n, self.biases_n, self.weights_linear,
                                            self.biases_linear)

        # Loss
        self.loss1 = self.loss_cal(self.labels_tf, self.labels_pred)
        self.loss2 = self.loss_cal(self.y_low_tf, self.low_pred)

        # Optimizers
        self.optimizer_Adam1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam1 = self.optimizer_Adam1.minimize(self.loss2)
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss1)

        # tf session
        self.lbfgs_buffer = []
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        mum_layers = len(layers)
        for l in range(0, mum_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32),
                           dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return H, Y

    def neural_net1(self, X, weights_n, biases_n, weights_linear, biases_linear):
        low1, low = self.neural_net(X, self.weights_e, self.biases_e)
        # linear
        N = tf.concat([X, low], 1)
        y_hi_l = tf.add(tf.matmul(N, weights_linear), biases_linear)

        # nonlinear
        mum_layers_n = len(weights_n) + 1
        H = tf.concat([X, low1], 1)
        for n in range(0, mum_layers_n - 2):
            W = weights_n[n]
            b = biases_n[n]
            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
        W_n = weights_n[-1]
        b_n = biases_n[-1]
        Y_hi_nl = tf.add(tf.matmul(H, W_n), b_n)

        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        y = tf.tanh(alpha_1) * y_hi_l + tf.tanh(alpha_2) * Y_hi_nl
        Y = tf.add(Y_hi_nl, y)
        return Y

    def loss_cal(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true) / y_true)

    def save_NN(self, fileDir1, fileDir2, fileDir3):
        weights_e = self.sess.run(self.weights_e)
        biases_e = self.sess.run(self.biases_e)
        weights_n = self.sess.run(self.weights_n)
        biases_n = self.sess.run(self.biases_n)
        weights_l = self.sess.run(self.weights_linear)
        biases_l = self.sess.run(self.biases_linear)
        with open(fileDir1, 'wb') as f:
            pickle.dump([weights_e, biases_e], f)
            print("Save nonlinear NN successfully...")
        with open(fileDir2, 'wb') as f:
            pickle.dump([weights_n, biases_n], f)
            print("Save nonlinear NN successfully...")
        with open(fileDir3, 'wb') as f:
            pickle.dump([weights_l, biases_l], f)
            print("Save linear NN successfully...")

    def callback(self, loss):
        self.lbfgs_buffer = np.append(self.lbfgs_buffer, loss)
        print('Loss:', loss)

    def train(self, nIter):
        loss2 = []
        loss_adam2 = []
        num_iter2 = []
        N_data = self.inputs.shape[0]
        for it in range(nIter):
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))

            inputs_batch = self.inputs[idx_data, :]
            y_low_batch = self.y_low[idx_data, :]
            tf_dict = {self.inputs_tf: inputs_batch,
                       self.y_low_tf: y_low_batch,
                       # self.learning_rate: learning_rate1
                       }
            self.sess.run(self.train_op_Adam1, tf_dict)

            if it % 10 == 0:
                loss_2 = self.sess.run(self.loss2, tf_dict)
                loss_adam2.append(loss_2)
                num_iter2.append(it)
            loss2.append(self.sess.run(self.loss2, tf_dict))

        loss = []
        loss_adam = []
        num_iter = []
        for it in range(nIter):
            idx_data = np.random.choice(N_data, min(self.batch_size, N_data))

            inputs_batch = self.inputs[idx_data, :]
            labels_batch = self.labels[idx_data, :]
            tf_dict = {self.inputs_tf: inputs_batch,
                       self.labels_tf: labels_batch,
                       # self.learning_rate: learning_rate1
                       }
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                loss_value = self.sess.run(self.loss1, tf_dict)
                loss_adam.append(loss_value)
                num_iter.append(it)

            loss.append(self.sess.run(self.loss1, tf_dict))
        return loss2, loss_adam2, num_iter2, loss, loss_adam, num_iter

    def predict(self, X_star):

        tf_dict = {self.inputs_tf: X_star}
        label_pred = self.sess.run(self.labels_pred, tf_dict)

        return label_pred
