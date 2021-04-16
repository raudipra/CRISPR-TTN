from typeguard import typechecked
from typing import Optional, Union, Callable

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import dispatch
from tensorflow.python.keras import backend as K
from tensorflow_addons.metrics import MeanMetricWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from utils import euclidean_distance, manhattan_distance


@dispatch.add_dispatch_support
@tf.function
def pair_triplet_accuracy(y_true: TensorLike, y_pred: TensorLike, 
                          margin: FloatTensorLike = 1.0, 
                          distance_metric: Union[str, Callable] = "L2"):
    """
        Calculates how often predictions matches the cut/non cut labels.
        Convert two embeddings into label `labels_pred` by calculating 
        distance and threshold using margin.
        It computes the frequency with which `labels_pred` matches `y_true`. 
        This frequency is ultimately returned as `pair accuracy`.
        Args:
            y_true: Integer ground truth values.
            y_pred: 3-D float `Tensor` of representational embedding of
                RNA and gRNA. [batch, 2, embedding_dim]
            margin: Float, threshold distance.
            distance_metric: String, distance metric in use.
        Returns:
            Float, accuracy values.
    """
    embeddings = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    labels = ops.convert_to_tensor_v2_with_dispatch(y_true)
    
    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or \
            embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) \
            if convert_to_float32 else embeddings
    )

    if distance_metric == "L2":
        dist = euclidean_distance(precise_embeddings[:, 0], 
                                  precise_embeddings[:, 1])
    elif distance_metric == "squared-L2":
        dist = tf.square(euclidean_distance(precise_embeddings[:, 0],
                                            precise_embeddings[:, 1]))
    elif distance_metric == "L1":
        dist = manhattan_distance(precise_embeddings[:, 0],
                                  precise_embeddings[:, 1])
    else: # Callable
        dist = distance_metric(precise_embeddings[:, 0],
                               precise_embeddings[:, 1])
        
    labels_pred = dist <= margin
    
    return math_ops.cast(math_ops.equal(
        tf.cast(tf.math.floormod(y_true, 2), tf.dtypes.bool), labels_pred), K.floatx()
    )
  
class PairTripletAccuracy(MeanMetricWrapper):
    """
        Wrapper for pair triplet accuracy
    """

    @typechecked
    def __init__(self, margin: FloatTensorLike = 1.0, 
                distance_metric: Union[str, Callable] = "L2", 
                name='pair_triplet_accuracy', dtype=None):
        super(PairTripletAccuracy, self).__init__(
            pair_triplet_accuracy, name, dtype=dtype, 
            margin=margin, distance_metric=distance_metric)