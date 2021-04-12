from typeguard import typechecked
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from utils import euclidean_distance, manhattan_distance


@tf.function
def constrained_triplet_loss_function(y_true: TensorLike, y_pred: TensorLike, 
                                      margin: FloatTensorLike = 1.0, 
                                      distance_metric: Union[str, Callable] = "L2"):
    """
        Calculate triplet loss function of two embedding pairs `y_pred`. This function is 
        made due to CRISPR-specific case, given comparison is only allowed for samples 
        with same gRNA. Hence this is a modified version of Tensorflow Triplet Loss 
        implementation with the key modifications are the form of `y_pred` and 
        triplet pair generation.
        Args:
            y_true: 1-D integer `Tensor` with shape `[batch_size]` of
              multiclass integer labels. Expected only two class in one batch. 
              0 positive pair, 1 negative pair.
            y_pred: 3-D float `Tensor` of embedding vectors. Embeddings should
              be l2 normalized. [batch_size, 2, embedding_dimension]
            margin: Float, margin term in the loss definition.
            distance_metric: `str` or a `Callable` that determines distance metric.
              Valid strings are "L2" for l2-norm distance,
              "squared-L2" for squared l2-norm distance,
              A `Callable` should take a batch of embeddings as input and
              return the pairwise distance matrix.
        Returns:
            triplet_loss: float scalar with dtype of `y_pred`.
    """

    labels = tf.convert_to_tensor(y_true, name="labels")
    embeddings = tf.convert_to_tensor(y_pred, name="embeddings")
    
    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or \
            embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) \
            if convert_to_float32 else embeddings
    )

    most_freq_label, max_freq = get_most_frequent_label(labels)
    ndata = labels.shape[0]

    squeezed_labels = tf.squeeze(labels, axis=1, name='squeeze_label')
    most_label_cond = tf.equal(squeezed_labels, most_freq_label)
    least_label_cond = tf.logical_not(most_label_cond)
    
    most_label_item = tf.squeeze(tf.gather(precise_embeddings, 
                                tf.where(most_label_cond)), 
                                axis=1, name='squeeze_most_label')
    least_label_item = tf.squeeze(tf.gather(precise_embeddings,
                                  tf.where(least_label_cond)),
                                  axis=1, name='squeeze_least_label')
    
    if distance_metric == "L2":
        dist_most_label = euclidean_distance(most_label_item[:, 0], 
                                             most_label_item[:, 1])
        dist_least_label = euclidean_distance(least_label_item[:, 0], 
                                              least_label_item[:, 1])
    elif distance_metric == "squared-L2":
        dist_most_label = tf.square(euclidean_distance(most_label_item[:, 0], 
                                                       most_label_item[:, 1]))
        dist_least_label = tf.square(euclidean_distance(least_label_item[:, 0], 
                                                        least_label_item[:, 1]))
    elif distance_metric == "L1":
        dist_most_label = manhattan_distance(most_label_item[:, 0], 
                                             most_label_item[:, 1])
        dist_least_label = manhattan_distance(least_label_item[:, 0], 
                                              least_label_item[:, 1])
    else: # Callable
        dist_most_label = distance_metric(most_label_item[:, 0], 
                                             most_label_item[:, 1])
        dist_least_label = distance_metric(least_label_item[:, 0], 
                                              least_label_item[:, 1])

    # Generate pair counterparts for dist_most_label from dist_least_label.
    least_label_idx = tf.random.uniform([max_freq], minval=0, 
                                        maxval=(ndata - max_freq - 1), 
                                        dtype=tf.dtypes.int32)
    dist_pair = tf.gather(dist_least_label, least_label_idx)
    
    # Cut label
    if tf.cast(most_freq_label, tf.dtypes.int32):
        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(dist_most_label - dist_pair + margin, 0.0)
            ),
            tf.cast(max_freq, tf.dtypes.float32),
        )
    else: # Non cut label
        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(dist_pair - dist_most_label + margin, 0.0)
            ),
            tf.cast(max_freq, tf.dtypes.float32),
        )
    
    if convert_to_float32:
        return tf.cast(triplet_loss, embeddings.dtype)
    else:
        return triplet_loss
  
class ConstrainedTripletLoss(LossFunctionWrapper):
    """
    Wrapper for constrained triplet loss function
    """

    @typechecked
    def __init__(
        self,
        margin: FloatTensorLike = 1.0, 
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            triplet_loss_function,
            name=name,
            margin=margin,
            distance_metric=distance_metric,
        )