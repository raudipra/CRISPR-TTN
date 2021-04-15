import sys

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from model import TwoTowerModel
from metric import PairTripletAccuracy
from triplet import ConstrainedTripletLoss
from data_loader import DataLoader


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

@tf.function
def one_hot(sequence):
    sequence = string_lookup(sequence)
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
    rna_seq = one_hot(feature[0])
    sgrna_seq = one_hot(feature[1])
    return tf.concat([rna_seq, sgrna_seq], 0), label

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python train.py [input_path] [output_model_path]")
        sys.exit()
    
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    data_loader = DataLoader(input_path, training_ratio=0.7)

    raw_train_ds, raw_val_ds = data_loader.load()

    # Why N? for one encoding purpose, last character = [0, 0, 0, 0]
    VOCAB = ["A", "G", "T", "N"]
    string_lookup = StringLookup(vocabulary=VOCAB)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 512
    SHUFFLE_SIZE = 1000

    encoded_train_ds = raw_train_ds.cache().shuffle(SHUFFLE_SIZE)
    encoded_train_ds = encoded_train_ds.prefetch(buffer_size=AUTOTUNE)
    encoded_train_ds = encoded_train_ds.map(preprocess)
    encoded_val_ds = raw_val_ds.cache().map(preprocess)

    train_ds = encoded_train_ds.cache().batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = encoded_val_ds.cache().batch(BATCH_SIZE)

    rna_length = data_loader.get_rna_length()
    grna_length = data_loader.get_grna_length()

    model = TwoTowerModel(rna_length=rna_length, grna_length=grna_length)
    model.build((None, rna_length + grna_length, 4))
    print(model.summary())

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=ConstrainedTripletLoss(num_labels=2),
        metrics=[PairTripletAccuracy()]
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    save_best = ModelCheckpoint(model_path, monitor='val_loss', 
                                mode='max', verbose=1, save_best_only=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[stop_early, save_best]
    )
      
    plot(history)