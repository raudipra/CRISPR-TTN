import tensorflow as tf


class TwoTowerModel(tf.keras.Model):
    def __init__(
        self,
        rna_length,
        grna_length,
        embedding_dim=128,
        name="twotowermodel",
        **kwargs
    ):
        super(TwoTowerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.rna_length = rna_length
        self.grna_length = grna_length

        self.rna_encoder = tf.keras.Sequential([
            # Shape of second dimension comes from the number of nucleotides
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', 
                                   activation='relu', name="rna_conv_1",
                                   input_shape=(self.rna_length, 4)),
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
        print(self.rna_encoder.summary())

        self.grna_encoder = tf.keras.Sequential([
            # Shape of second dimension comes from the number of nucleotides
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', 
                                   activation='relu', name="g_rna_conv_1",
                                   input_shape=(self.grna_length, 4)),
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
        print(self.grna_encoder.summary())
    
    def call(self, inputs):
        """
            Args:
                inputs: 2-D float `Tensor` of one-hot nucleotides encoded embedding. 
                    The one-hot nucleotides embedding is a concatenation of rna_seq 
                    and grna_seq pair embedding. [batch, rna_length + grna_length]
            Returns:
                outputs: 3-D float `Tensor` of representational embedding of rna and grna.
                    [batch, 2, embedding_dim]
        """
        rna_seq = inputs[:, :self.rna_length]
        grna_seq = inputs[:, self.rna_length:]
        
        rna_embedding = self.rna_encoder(rna_seq)        
        grna_embedding = self.grna_encoder(grna_seq)
        
        outputs = tf.stack((rna_embedding, grna_embedding), axis=1)
        
        return outputs
