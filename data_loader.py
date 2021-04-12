import numpy as np
import pandas as pd
import tensorflow as tf

class DataLoader():
    """
        Load the CSV file as a dataframe and do some preprocessing before 
        return it as tf.Dataset.
        Args:
            input_path: str, path to the csv file.
            training_ratio: float, ratio for training validation split.
        Returns:
            raw_train_ds, raw_val_ds: tf.Dataset typed of train and val set.
    """
    def __init__(self, input_path, training_ratio=0.7):
        self.input_path = input_path

    def load(self):
        df = pd.read_csv(self.input_path)
        dataset = df[['rna', 'grna', 'label']].values
        X = dataset[:, :2]
        y = dataset[:, -1]

        # Tokenize every nucleotides
        tokenize = lambda x: [np.array([c for c in x[0]]), 
                              np.array([c for c in x[1]])]
        X = np.apply_along_axis(tokenize, 1, X)

        X_train = X[:int(training_ratio * X.shape[0])]
        X_val = X[int(training_ratio * X.shape[0]):]

        y_train = y[:int(training_ratio * y.shape[0])] 
        y_val = y[int(training_ratio * y.shape[0]):]

        self.train = np.concatenate(
            (X_train, np.expand_dims(y_train, axis=1)), 
            axis=1,
        )
        self.val = np.concatenate(
            (X_val, np.expand_dims(y_val, axis=1)),
            axis=1,
        )

        raw_train_ds = tf.data.Dataset.from_generator(
            self.gen_train,
            output_signature=(
                tf.RaggedTensorSpec(shape=(2, None), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        )

        raw_val_ds = tf.data.Dataset.from_generator(
            self.gen_val,
            output_signature=(
                tf.RaggedTensorSpec(shape=(2, None), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        )

        return raw_train_ds, raw_val_ds

    # Output it as a ragged type, because we want to 
    # preprocess it through tf.Dataset map built-in.
    def gen_train(self):
        np.random.shuffle(self.train)
        for idx, row in enumerate(self.train):
            feature = tf.ragged.constant([row[0].tolist(), row[1].tolist()])
            yield feature, row[2]

    def gen_val(self):
        np.random.shuffle(self.val)
        for idx, row in enumerate(self.val):
            feature = tf.ragged.constant([row[0].tolist(), row[1].tolist()])
            yield feature, row[2]
