import tensorflow as tf


class TwoTowerModel(tf.keras.Model):
    def __init__(
        self,
        RNA_length,
        gRNA_length,
        embedding_dim=128,
        name="twotowermodel",
        **kwargs
    ):
        super(TwoTowerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.RNA_length = RNA_length
        self.gRNA_length = gRNA_length

        self.RNA_encoder = tf.keras.Sequential([
            # Shape of second dimension comes from the number of nucleotides
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', 
                                   activation='relu', name="rna_conv_1",
                                   input_shape=(self.RNA_length, 4)),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="rna_max_pool_1"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', 
                                   activation='relu', name="rna_conv_2"),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="rna_max_pool_2"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.embedding_dim, activation=None, 
                                  name="rna_dense_1"), # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
        ], name="rna_sequential")
        print(self.RNA_encoder.summary())

        self.gRNA_encoder = tf.keras.Sequential([
            # Shape of second dimension comes from the number of nucleotides
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', 
                                   activation='relu', name="g_rna_conv_1",
                                   input_shape=(self.gRNA_length, 4)),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="g_rna_max_pool_1"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', 
                                   activation='relu', name="g_rna_conv_2"),
            tf.keras.layers.MaxPooling1D(pool_size=2, name="g_rna_max_pool_2"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.embedding_dim, activation=None, 
                                  name="g_rna_dense_1"), # No activation on final dense layer
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
        ], name="g_rna_sequential")
        print(self.gRNA_encoder.summary())
    
    def call(self, inputs):
        """
            Args:
                inputs: 2-D float `Tensor` of one-hot nucleotides encoded embedding. 
                    The one-hot nucleotides embedding is a concatenation of RNA_seq 
                    and gRNA_seq pair embedding. [batch, RNA_length + gRNA_length]
            Returns:
                outputs: 3-D float `Tensor` of representational embedding of RNA and gRNA.
                    [batch, 2, embedding_dim]
        """
        RNA_seq = inputs[:, :RNA_length]
        gRNA_seq = inputs[:, RNA_length:]
        
        RNA_embedding = self.RNA_encoder(RNA_seq)        
        gRNA_embedding = self.gRNA_encoder(gRNA_seq)
        
        outputs = tf.stack((RNA_embedding, gRNA_embedding), axis=1)
        
        return outputs
