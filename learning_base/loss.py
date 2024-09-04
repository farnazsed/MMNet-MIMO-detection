import tensorflow as tf
from utils import batch_matvec_mul

def loss_fun(xhat, x):
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss = 0.
    for xhatk in xhat:
        lk = mse_loss_fn(y_true=x, y_pred=xhatk)
        loss += lk
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, lk)
    #print("Only using the last layer loss")
    #loss = lk
    #print("regularizations added")
    #loss += 0.01 * tf.reduce_mean(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
    return loss

def loss_yhx(y, xhat, H):
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    loss = 0.
    for xhatk in xhat:
        prediction = batch_matvec_mul(H, xhatk)
        lk = mse_loss_fn(y_true=y, y_pred=prediction)
        loss += lk
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, lk)
    return loss
