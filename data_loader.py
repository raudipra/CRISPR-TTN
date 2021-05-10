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
        self.training_ratio = training_ratio
        self.grnas = np.array([])

    def load_test_set(self):
        df = pd.read_csv(self.input_path)
        
        X = df[['rna', 'grna']].values
        
        # Fix the label into 2 x unique grna (cut and non cut) 
        self.grnas = np.unique(X[:, 1])

        if X.shape[0]:
            self.rna_length = len(X[0][0])
            self.grna_length = len(X[0][1])
        
        # Tokenize every nucleotides
        tokenize = lambda x: [np.array([c for c in x[0]]), 
                              np.array([c for c in x[1]])]
        self.test = np.apply_along_axis(tokenize, 1, X)

        raw_test_ds = tf.data.Dataset.from_generator(
            self.gen_test,
            output_signature=(
                tf.RaggedTensorSpec(shape=(2, None), dtype=tf.string)
            )
        )

        return raw_test_ds

    def load(self):
        df = pd.read_csv(self.input_path)
        
        dataset = df[['rna', 'grna', 'label']].values
        np.random.seed(42)
        np.random.shuffle(dataset)

        # Fix the label into 2 x unique grna (cut and non cut) 
        self.grnas = np.unique(dataset[:, 1])

        dataset[:, 2] = np.array([
            (np.where(self.grnas == grna)[0][0] * 2 + label) \
                for grna, label in zip(dataset[:, 1], dataset[:, -1])
        ])

        X = dataset[:, :2]
        y = dataset[:, -1]

        if X.shape[0]:
            self.rna_length = len(X[0][0])
            self.grna_length = len(X[0][1])
        
        # Tokenize every nucleotides
        tokenize = lambda x: [np.array([c for c in x[0]]), 
                              np.array([c for c in x[1]])]
        X = np.apply_along_axis(tokenize, 1, X)

        X_train = X[:int(self.training_ratio * X.shape[0])]
        X_val = X[int(self.training_ratio * X.shape[0]):]

        y_train = y[:int(self.training_ratio * y.shape[0])] 
        y_val = y[int(self.training_ratio * y.shape[0]):]

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

    def get_rna_length(self):
        if self.rna_length == None:
            print("Error: rna_length is missing. Run DataLoader.load() to generate it.")
        return self.rna_length

    def get_grna_length(self):
        if self.grna_length == None:
            print("Error: grna_length is missing. Run DataLoader.load() to generate it.")
        return self.grna_length

    def get_num_labels(self):
        if not len(self.grnas):
            print("Error: grnas is missing. Run DataLoader.load() to generate it.")
        return len(self.grnas) * 2

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

    def gen_test(self):
        np.random.shuffle(self.test)
        for idx, row in enumerate(self.test):
            feature = tf.ragged.constant([row[0].tolist(), row[1].tolist()])
            yield feature
