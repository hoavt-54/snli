import tensorflow as tf
from tensorflow import Tensor as ts



def tile_repeat(n, repTime):
    print n, repTime
    idx = tf.range(n)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a len(yp) x 1 matrix.
    idx = tf.tile(idx, [1, repTime])  # Create multiple columns.
    y = tf.reshape(idx, [-1])
    return y

def gather_along_second_axis(x, idx):
    ''' 
    x has shape: [batch_size, sentence_length, word_dim]
    idx has shape: [batch_size, num_indices]
    Basically, in each batch, get words from sentence having index specified in idx
    However, since tensorflow does not fully support indexing,
    gather only work for the first axis. We have to reshape the input data, gather then reshape again
    '''
    idx1= tf.reshape(idx, [-1]) # [batch_size*num_indices]
    idx_flattened = tile_repeat(tf.shape(idx)[0], tf.shape(idx)[1]) * tf.shape(x)[1] + idx1
    y = tf.gather(tf.reshape(x, [-1,tf.shape(x)[2]]),  # flatten input
                idx_flattened)
    y = tf.reshape(y, tf.shape(x))
    return y





x = tf.constant([
            [[1,2,3],[3,5,6]],
            [[7,8,9],[10,11,12]],
            [[13,14,15],[16,17,18]]
    ])
idx=tf.constant([[0,1],[1,0],[1,1]])

#idx1= tf.reshape(idx, [-1])
#idx_flattened = tile_repeat(tf.shape(idx)[0], tf.shape(idx)[1]) * tf.shape(x)[1] + idx1
#y = tf.gather(tf.reshape(x, [-1,int(tf.shape(x)[2])]), idx_flattened) 
#y = tf.reshape(y, tf.shape(x))

y = gather_along_second_axis(x, idx)
with tf.Session(''):
    print y.eval()
    print tf.Tensor.get_shape(y)
    #y1 = tf.reshape(y, x.shape)
    #print tf.Tensor.shape(y1)
    #y1.eval()
