import numpy as np
import tensorflow as tf
import os
import pickle
from tf_session import *
from parser import parse
from exp import get_data

params, args = parse()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_channel_samples = 100


def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=-1)
    h2 = np.concatenate([Hi, Hr], axis=-1)
    out = np.concatenate([h1, h2], axis=1)
    return out


def resize_or_pad_data(data, target_shape):
    """
    Resize or pad the data to match the target shape.
    If the data is larger than target_shape, crop the data.
    If the data is smaller, pad it.
    """
    data_shape = data.shape
    if len(data_shape) != len(target_shape):
        raise ValueError("Data and target shapes must have the same number of dimensions")

    new_data = data
    for dim, (current_size, target_size) in enumerate(zip(data_shape[1:], target_shape)):
        if current_size < target_size:
            # Pad
            padding = [(0, 0)] * dim + [(0, target_size - current_size)] + [(0, 0)] * (len(target_shape) - dim - 1)
            new_data = np.pad(new_data, padding, mode='constant')
        elif current_size > target_size:
            # Crop
            slices = [slice(None)] * dim + [slice(None, target_size)] + [slice(None)] * (len(target_shape) - dim - 1)
            new_data = new_data[tuple(slices)]

    return new_data


if args.data:
    H_dataset = np.load(args.channels_dir)
    print("Loaded H_dataset shape:", H_dataset.shape)
    assert H_dataset.shape[-1] == args.x_size
    assert H_dataset.shape[-2] == args.y_size

    H_dataset = np.reshape(H_dataset, (-1, H_dataset.shape[-2], H_dataset.shape[-1]))
    H_dataset = complex_to_real(H_dataset)
    Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.)
    params['Hdataset_powerdB'] = Hdataset_powerdB
    print('Channels dataset power (dB): %f' % Hdataset_powerdB)

    train_data_ref = H_dataset
    test_data_ref = H_dataset
    print('Channels dataset shape: ', H_dataset.shape)
    rndIndx = np.random.randint(0, train_data_ref.shape[0], num_channel_samples)
    train_data_ref = train_data_ref[rndIndx]
    test_data_ref = test_data_ref[rndIndx]
    print('Sampled channel indices: ', rndIndx)

else:
    test_data = []
    train_data = []

    # Build the computational graph
mmnet = MMNet_graph(params)
nodes = mmnet.build()

# Get access to the nodes on the graph
sess = nodes['sess']
x = nodes['x']
H = nodes['H']
x_id = nodes['x_id']
constellation = nodes['constellation']
train = nodes['train']
snr_db_min = nodes['snr_db_min']
snr_db_max = nodes['snr_db_max']
lr = nodes['lr']
batch_size = nodes['batch_size']
accuracy = nodes['accuracy']
mmse_accuracy = nodes['mmse_accuracy']
loss = nodes['loss']
logs = nodes['logs']
measured_snr = nodes['measured_snr']
init = nodes['init']

# Training loop
for r in range(num_channel_samples):
    sess.run(init)
    train_data = np.expand_dims(train_data_ref[r], axis=0)
    test_data = np.expand_dims(test_data_ref[r], axis=0)
    results = {}
    for it in range(args.train_iterations + 1):
        feed_dict = {
            batch_size: args.batch_size,
            lr: args.learn_rate,
            snr_db_max: params['SNR_dB_max'],
            snr_db_min: params['SNR_dB_min'],
        }
        if args.data:
            sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
            feed_dict[H] = train_data[sample_ids]

        sess.run(train, feed_dict)

        # Test
        if it == args.train_iterations:
            for snr_ in range(int(params['SNR_dB_min']), int(params['SNR_dB_max']) + 1):
                feed_dict = {
                    batch_size: args.test_batch_size,
                    snr_db_max: snr_,
                    snr_db_min: snr_,
                }
                if args.data:
                    sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
                    feed_dict[H] = test_data[sample_ids]

                test_accuracy_, test_loss_, measured_snr_, log_ = sess.run([accuracy, loss, measured_snr, logs],
                                                                           feed_dict)
                print('Test SER of %f on channel realization %d after %d iterations at SNR %f dB' % (
                1. - test_accuracy_, r, it, measured_snr_))
                results[str(snr_)] = {}
                for k in log_:
                    results[str(snr_)][k] = log_[k]['stat']
                results[str(snr_)]['accuracy'] = test_accuracy_
            results['cond'] = np.linalg.cond(test_data[sample_ids][0])
            path = args.output_dir + '/OnlineTraining_%s_NT%sNR%s_%s/' % (
            args.modulation, args.x_size, args.y_size, args.linear)
            if not os.path.exists(path):
                os.makedirs(path)
            savePath = path + 'reults%d.pkl' % r
            with open(savePath, 'wb') as f:
                pickle.dump(results, f)
            print('Results saved at %s' % savePath)
