from typeguard import typechecked
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from utils import euclidean_distance, manhattan_distance

@tf.function
def group_dist_mat_by_label(i, labels, dist_mat, grouped_dist):
    # half of batch
    num_triplet_pair_per_sample = tf.math.floordiv(tf.size(labels), 2)
    idx = tf.squeeze(tf.where(tf.equal(labels, i)), axis=1)

    if tf.shape(idx)[0] == 0:
        if i == 0.:
            grouped_dist = tf.zeros((1, num_triplet_pair_per_sample))
        else:
            grouped_dist = tf.concat([grouped_dist, tf.zeros((1, num_triplet_pair_per_sample))], axis=0)
        
        i = i + 1
        return i, labels, dist_mat, grouped_dist

    probability = tf.zeros([tf.size(labels)], dtype=tf.float32) # batch size
    updates = tf.fill(tf.shape(idx), tf.truediv(1.0, tf.cast(tf.size(idx), tf.float32)))
    probability = tf.tensor_scatter_nd_update(probability, tf.expand_dims(idx, 1), updates)

    probability
    triplet_pair_idx = tf.squeeze(
        tf.random.categorical(
            tf.math.log(tf.expand_dims(probability, 0)), 
            num_triplet_pair_per_sample
        ),
        axis=0
    )
    
    triplet_pair = tf.squeeze(tf.gather(dist_mat, triplet_pair_idx, name="triplet_pair_gather"), axis=1)
   
    if i == 0.:
        grouped_dist = tf.expand_dims(triplet_pair, axis=0)
    else:
        grouped_dist = tf.concat([grouped_dist, tf.expand_dims(triplet_pair, axis=0)], axis=0)
    
    i = i + 1
    
    return i, labels, dist_mat, grouped_dist

@tf.function
def constrained_triplet_loss_function(y_true: TensorLike, y_pred: TensorLike,  
                                      num_labels: int,
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

    labels = tf.squeeze(labels, axis=1)
      
    if distance_metric == "L2":
        dist_mat = euclidean_distance(precise_embeddings[:, 0], 
                                      precise_embeddings[:, 1])
    elif distance_metric == "squared-L2":
        dist_mat = tf.square(euclidean_distance(precise_embeddings[:, 0], 
                                                precise_embeddings[:, 1]))
    elif distance_metric == "L1":
        dist_mat = manhattan_distance(precise_embeddings[:, 0], 
                                      precise_embeddings[:, 1])
    else: # Callable
        dist_mat = distance_metric(precise_embeddings[:, 0], 
                                   precise_embeddings[:, 1])
    
    # make it even for cut and non cut pairs, because why not?
    num_labels += num_labels % 2

    condition = lambda i, labels, dist_mat, grouped_dist: i < num_labels

    i = tf.constant(0.)
    grouped_dist = tf.convert_to_tensor([[]], name="grouped_dist")
    __, __, __, grouped_dist = tf.while_loop(
        condition, group_dist_mat_by_label, 
        loop_vars=[i, labels, dist_mat, grouped_dist],
        shape_invariants=[i.get_shape(), labels.get_shape(), 
                         dist_mat.get_shape(), tf.TensorShape([None, None])]
    )

    cut_dist_mat = grouped_dist[1::2]
    noncut_dist_mat = grouped_dist[::2]
    
    # check if all zeros, meaning missing label
    cut_dist_total = tf.reduce_sum(tf.abs(cut_dist_mat), axis=1)
    is_valid_cut_pairs = tf.not_equal(cut_dist_total, 0.0)
    noncut_dist_total = tf.reduce_sum(tf.abs(noncut_dist_mat), axis=1)
    is_valid_noncut_pairs = tf.not_equal(noncut_dist_total, 0.0)
    
    # if one index has all zeros value in either cut or noncut, 
    # then remove the counterpart pair as well
    is_valid_pairs = tf.math.logical_and(is_valid_cut_pairs, is_valid_noncut_pairs)
    valid_idx = tf.squeeze(tf.where(is_valid_pairs), axis=1)
    cut_dist_mat = tf.squeeze(tf.gather(cut_dist_mat, valid_idx, name="gather_cut_dist_mat"))
    noncut_dist_mat = tf.squeeze(tf.gather(noncut_dist_mat, valid_idx, name="gather_noncut_dist_mat"))
    
    # Cut label
    triplet_loss = tf.math.truediv(
        tf.math.reduce_sum(
            tf.math.maximum(cut_dist_mat - noncut_dist_mat + margin, 0.0)
        ),
        tf.cast(tf.size(cut_dist_mat), tf.dtypes.float32),
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
        num_labels: int,
        margin: FloatTensorLike = 1.0, 
        distance_metric: Union[str, Callable] = "L2",
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            constrained_triplet_loss_function,
            name=name,
            num_labels=num_labels,
            margin=margin,
            distance_metric=distance_metric,
        )