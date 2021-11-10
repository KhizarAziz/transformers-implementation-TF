import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
import transformer_model
import input_processing
import os
import tqdm


'''
GPU RELATED STUFF BEING INIT
'''




os.environ["CUDA_VISIBLE_DEVICES"]="0"
logging.getLogger('MyTensorflowLogger').setLevel(logging.ERROR)  # suppress warnings

'''
ENABLE THIS CODE WHEN THE MULTI GPU TRAINING IS WORKING
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

'''


'''
####################################################
############## DATASET ##################
####################################################
'''
# download dataset (portugese dataset from tf datasets) ...... (Portuguese-English translation dataset from the TED Talks Open Translation Project.)
#examples , metadata = tfds.load('ted_hrlr_translate/pt_to_en',with_info=True,as_supervised=True)
#train_examples,val_examples = examples['train'], examples['validation']
import jsonlines
inputs = []
labels = []
hits = 0
with jsonlines.open('dataset/realnews/realnews.jsonl') as f:
    count = 0
    for line in tqdm.tqdm(f.iter()):
        if line['summary'] != None and line['text'] != None:
            if len(line['text']) < 1500:
                inputs.append(line['summary'])
                labels.append(line['text'])
                hits +=1
            count +=  1
            if count > 99999:
                break
print('\n\nGot {} articles from total of {} articles \n\n'.format(hits,count))
# convert to numpy
train_inputs = np.array(inputs)
train_labels = np.array(labels)
print('##############    Input and label shapes: ',train_inputs.shape, train_labels.shape)

# create tfds from using numpy arrays
train_examples = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
val_examples = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))


'''
####################################################
##########################  TOKENIZER ##################
####################################################
'''

# download and unzip tokenizer model (that is designed and optimized specially for this dataset)
model_name = "ted_hrlr_translate_pt_en_converter"
tf.keras.utils.get_file( # tensorflow built in method to download a file ... very useful
    f"{model_name}.zip", # give file name to name the downloaded file
    f"https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip", # url of the file
    cache_dir = '.',cache_subdir='',extract=True # other flags
)
tokenizers = tf.saved_model.load(model_name) # loading model!!!!!!!!!!!!
'''
####################################################
##########################  INPUT PIPELINE ##################
####################################################
'''
# input params -> "pt and en" are tensors with shape (batchsize,1) => "batchsize" number of strings inside list.
def tokenize_pairs(input_seq,tar_seq): #converting a pair (input,label) into tokenized tensors and reutnr
  input_seq = tokenizers.en.tokenize(input_seq) # converting a batch of strings into tokens, the encoded object is of ragged tensor
  input_seq = input_seq.to_tensor() # converting ragged tensor to normal tensor
  tar_seq = tokenizers.en.tokenize(tar_seq)
  tar_seq = tar_seq.to_tensor()

  # pt = tf.cast(pt, tf.float32)
  # en = tf.cast(en, tf.float32)
  return input_seq, tar_seq

BUFFER_SIZE = 2000
BATCH_SIZE = 32

def make_batches(ds): # creating batches of dataset and also doing something on it.
  return (
      ds # our TF dataset object
      #  .c()ache # cacheing the dataset so that dont load again againRAMS: ',transformer.summary(),'\n\n\n')
      .shuffle(BUFFER_SIZE) # shuffle entries inside ds
      .batch(BATCH_SIZE) # create batches
      .map(tokenize_pairs,num_parallel_calls=tf.data.AUTOTUNE) # parallel tokenize????
      .prefetch(tf.data.AUTOTUNE) # aisa krna hta ha?????
  )

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

'''
####################################################
Set Hyperparameters ####################
####################################################
'''
# To keep this example small and relatively fast, the values for `num_layers, d_model, dff` have been reduced.
# Real paper  used: `num_layers=6, d_model=512, dff=2048`.
num_layers = 27 # number of encoder/decoder layer
d_model = 512    # word_embedding size
dff = 512   # maximum seq length... (padded)
num_heads = 16  # Z0, Z1, Z2, Z3 .... Z15
dropout_rate = 0.1


'''
####################################################
OPTIMIZER ####################
####################################################
'''

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)




'''
####################################################
Loss and metrics ####################
####################################################
'''

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask  = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real,pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
'''
####################################################
Training and checkpointing ####################
####################################################
'''


transformer = transformer_model.Transformer(
    num_layers = num_layers, # 12
    d_model = d_model, # 512
    num_heads = num_heads, # 16
    dff=dff, # 512
    input_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    target_vocab_size = tokenizers.en.get_vocab_size().numpy(),
    pe_input=7000,
    pe_target=7000,
    rate=dropout_rate)

#print('\n\n model PARAMS: ',transformer.summary(),'\n\n\n')

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')


EPOCHS = 22


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))


step_count = 0

for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_batches):
    step_count += 1
      #    print('||||||||||||||||||||| STEP {} START |||||||||||||'.format(batch))
#    print('||||||||||||||||||||| INPUT {} TARGET {}|||||||||||||'.format(inp.shape,tar.shape))
    train_step(inp, tar)
    print('Step: ',step_count)
#    print('\n\n model PARAMS: ',transformer.summary(),'\n\n\n')
#    exit()
#    print('||||||||||||||||||||| STEP {} END |||||||||||||'.format(batch))
    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
