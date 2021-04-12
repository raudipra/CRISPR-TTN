import os
import datetime

import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


zf = zipfile.ZipFile('/content/drive/MyDrive/COMS4762_ML4G/Project/ML4FG_final_project/Dataset/CIRCLE_seq_data.zip') # having First.csv zipped file.
print(zf.namelist())
df_cut = pd.read_csv(zf.open('CIRCLE_seq_cut_5bp_flank.csv'))
df_noncut = pd.read_csv(zf.open('CIRCLE_seq_noncut_5bp_flank.csv'))

TRAINING_RATIO = 0.6
tokenize = lambda x: [np.array([c for c in x[0]]), np.array([c for c in x[1]])]

df_cut_filtered = df_cut[['context_seq', 'grna']].sample(frac=0.01).reset_index(drop=True).values
df_noncut_filtered = df_noncut[['context_seq', 'grna']].sample(frac=0.003).reset_index(drop=True).values

df_cut_tokenized = np.apply_along_axis(tokenize, 1, df_cut_filtered)
df_noncut_tokenized = np.apply_along_axis(tokenize, 1, df_noncut_filtered)

train_cut = df_cut_tokenized[:int(TRAINING_RATIO * df_cut_tokenized.shape[0])]
train_noncut = df_noncut_tokenized[:int(TRAINING_RATIO * df_noncut_tokenized.shape[0])]
val_cut = df_cut_tokenized[int(TRAINING_RATIO * df_cut_tokenized.shape[0]):]
val_noncut = df_noncut_tokenized[int(TRAINING_RATIO * df_noncut_tokenized.shape[0]):]

train = np.concatenate((train_cut, train_noncut), axis=0)
val = np.concatenate((val_cut, val_noncut), axis=0)
train_target = np.concatenate((np.ones(train_cut.shape[0]), np.zeros(train_noncut.shape[0])), axis=0) 
val_target = np.concatenate((np.ones(val_cut.shape[0]), np.zeros(val_noncut.shape[0])), axis=0)

train = np.concatenate((train, np.expand_dims(train_target, axis=1)), axis=1)
val = np.concatenate((val, np.expand_dims(val_target, axis=1)), axis=1)

def gen_train():
    np.random.shuffle(train)
    for idx, row in enumerate(train):
        feature = tf.ragged.constant([row[0].tolist(), row[1].tolist()])
        yield feature, row[2]

def gen_val():
    np.random.shuffle(val)
    for idx, row in enumerate(val):
        feature = tf.ragged.constant([row[0].tolist(), row[1].tolist()])
        yield feature, row[2] 

raw_train_ds = tf.data.Dataset.from_generator(
    gen_train,
    output_signature=(
        tf.RaggedTensorSpec(shape=(2, None), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

raw_val_ds = tf.data.Dataset.from_generator(
    gen_val,
    output_signature=(
        tf.RaggedTensorSpec(shape=(2, None), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

for feat, targ in raw_train_ds.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

VOCAB = ["A", "G", "T", "N"] # Why N? for one encoding purpose, last character = [0, 0, 0, 0]
stringLookup = StringLookup(vocabulary=VOCAB)

@tf.function
def one_hot(sequence):
    sequence = stringLookup(sequence)
    C = tf.constant(5)
    one_hot_matrix = tf.one_hot(
        sequence,
        C,
        on_value=1.0,
        off_value=0.0,
        axis =-1
    )
    return one_hot_matrix[:, 1:]

def preprocess(feature, label):
    RNA_seq = one_hot(feature[0])
    sgRNA_seq = one_hot(feature[1])
    return tf.concat([RNA_seq, sgRNA_seq], 0), label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256
SHUFFLE_SIZE = 1000

encoded_train_ds = raw_train_ds.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=AUTOTUNE)
encoded_train_ds = encoded_train_ds.map(preprocess)
encoded_val_ds = raw_val_ds.cache().map(preprocess)
# encoded_test_ds = raw_test_ds.cache().map(preprocess)

train_ds = encoded_train_ds.cache().batch(BATCH_SIZE) # cache data in mempry
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

val_ds = encoded_val_ds.cache().batch(BATCH_SIZE)
# test_ds = vectorized_test_ds.cache()

model = TwoTowerModel(sequence_length=33, sgRNA_length=23)
model.build((None, 56, 4))
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=CustomTripletLoss(),
    metrics=[CustomAccuracy()]
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

model_dir = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(model_dir)

model_path = model_dir + "/best_model.h5"
save_best = ModelCheckpoint(model_path, monitor='val_custom_accuracy', mode='max', verbose=1, save_best_only=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=1,
                                                 profile_batch='210, 220')

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[stop_early, save_best, tboard]
)

# A plotting function you can reuse
def plot(history):
  
  # The history object contains results on the training and test
  # sets for each epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # Get the number of epochs
  epochs = range(len(loss))

  _ = plt.figure()
  plt.title('Training and validation loss')
  plt.plot(epochs, loss, color='blue', label='Train')
  plt.plot(epochs, val_loss, color='orange', label='Val')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  
plot(history)