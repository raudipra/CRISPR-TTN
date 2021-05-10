import sys

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from model import TwoTowerModel
from data_loader import DataLoader
from utils import one_hot, convert_embed_to_label


def preprocess_test(feature):
    RNA_seq = one_hot(feature[0])
    sgRNA_seq = one_hot(feature[1])
    return tf.concat([RNA_seq, sgRNA_seq], 0)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python predict.py [input_path] [model_path] [output_path]")
        sys.exit()
    
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    data_loader = DataLoader(input_path)

    BATCH_SIZE = 2048

    raw_test_ds = data_loader.load_test_set()
    encoded_test_ds = raw_test_ds.map(preprocess_test)
    test_ds = encoded_test_ds.cache().batch(BATCH_SIZE)
    
    rna_length = data_loader.get_rna_length()
    grna_length = data_loader.get_grna_length()

    model = TwoTowerModel(rna_length=rna_length, grna_length=grna_length)
    model.build((None, rna_length + grna_length, 4))
    print(model.summary())

    model.load_weights(model_path)
    y_pred_embedding = model.predict(test_ds)
    y_pred = convert_embed_to_label(y_pred_embedding).numpy()

    pd.DataFrame(y_pred, columns=['prediction']).to_csv(output_path, index=False)
