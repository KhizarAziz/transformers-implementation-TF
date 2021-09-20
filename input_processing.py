import numpy as np
import tensorflow as tf

'''
####################################################
########################## POSITIONAL ENCODER ####################
####################################################
'''

def get_angles(pos,i,d_model): # getting angles for cos and sin as in PE formula
  angle_rates = 1/np.power(10000,(2*(i//2)) / np.float32(d_model))
  return pos * angle_rates ## Positional encoding formula from the paper


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            # angles_rads.shape = 2048x512 cuz hmara sentence is of length 2048 with each word's vector of size 512.
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]  # appending new dimension in the start

    return tf.cast(pos_encoding, dtype=tf.float32)  # casting type of input tensor

position = 2048 # total positions 2048 which means our input sentence length is considered 2048
d_model = 512  # this is the length of word embedding... in the paper they used word2vec to create 512 sized embedding for each word

'''
####################################################
########################## MASKING ####################
####################################################
'''
def create_padding_mask(seq):
  bool_masked_arr = tf.math.equal(seq,0) # jdr jdr "seq(array) == 0" udhr 1 put kro
  seq = tf.cast(bool_masked_arr,tf.float32) # cast bool array to float32

  # add extra dimensions to add the padding
  # to attention logits
  return seq[:,tf.newaxis,tf.newaxis,:] # (batch_size,1,1,seq_len)


def create_look_ahead_mask(size): # lookahead mask, most probably for decoding part
  dummy_arr = tf.ones((size,size)) # original seq k size ki dummy ones ki array bna rhe... phr iss arr se masks bnayngay
  mask = 1 - tf.linalg.band_part(dummy_arr,-1,0) # isme hum uss sizexsize matrix k diagonal se niche sb values ko 0 kr rhe hn, so that is our mask
  return mask

