import tensorflow as tf
from utils import *
from detector import detector
from loss import loss_fun, loss_yhx
from sample_generator import generator


class MMNet_graph():
    def __init__(self, params):
        self.params = params

    def build(self):

        tf.compat.v1.disable_eager_execution()  # Disable eager execution
        print("TensorFlow version:", tf.__version__)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("TensorFlow is using GPU")
        else:
            print("TensorFlow is using CPU")

        with tf.device('/cpu:0'):
            tf.compat.v1.set_random_seed(self.params['seed'])

            # Placeholders for feed dict
            batch_size = tf.compat.v1.placeholder(tf.int32, shape=(), name='batch_size')
            lr = tf.compat.v1.placeholder(tf.float32, shape=(), name='lr')
            snr_db_max = tf.compat.v1.placeholder(tf.float32, shape=(), name='snr_db_max')
            snr_db_min = tf.compat.v1.placeholder(tf.float32, shape=(), name='snr_db_min')
            train_flag = tf.compat.v1.placeholder(tf.bool, shape=(), name='train_flag')

            # MIMO sample generator model
            mimo = generator(self.params, batch_size)

            # Generate transmitted signals
            constellation = mimo.constellation
            indices = mimo.random_indices()
            x = mimo.modulate(indices)

            # Send x through the channel
            if self.params['data']:
                H = tf.compat.v1.placeholder(tf.float32, shape=(None, 2 * self.params['N'], 2 * self.params['K']),
                                             name='H')
                y, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, H, self.params['data'],
                                                            self.params['correlation'])
            else:
                y, H, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, [], self.params['data'],
                                                               self.params['correlation'])

            # Zero-forcing detection
            x_mmse = mmse(y, H, noise_sigma)
            x_mmse_idx = demodulate(x_mmse, constellation)
            x_mmse = tf.gather(constellation, x_mmse_idx)
            acc_mmse = accuracy(indices, demodulate(x_mmse, constellation))

            # MMNet detection
            x_NN, helper = detector(self.params, constellation, x, y, H, noise_sigma, indices,
                                    batch_size).create_graph()
            loss = loss_fun(x_NN, x)

            temp = []
            for i in range(self.params['L']):
                temp.append(accuracy(indices, demodulate(x_NN[i], constellation)))
            acc_NN = tf.reduce_max(temp)

            # Training operation
            train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(tf.reduce_mean(loss))

            # Init operation
            init = tf.compat.v1.global_variables_initializer()

            # Define saver
            saver = tf.compat.v1.train.Saver()

            # Create session and initialize all variables
            sess = tf.compat.v1.Session()

            if len(self.params['start_from']) > 1:
                saver.restore(sess, self.params['start_from'])
            else:
                sess.run(init)

            nodes = {'measured_snr': actual_snrdB, 'batch_size': batch_size, 'lr': lr, 'snr_db_min': snr_db_min,
                     'snr_db_max': snr_db_max, 'x': x, 'x_id': indices, 'H': H, 'y': y, 'sess': sess,
                     'train': train_step, 'accuracy': acc_NN, 'loss': loss, 'mmse_accuracy': acc_mmse,
                     'constellation': constellation, 'logs': helper, 'init': init}
        return nodes
