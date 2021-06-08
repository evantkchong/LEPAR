import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def _pairwise_distances(embeddings, squared=False):
    """This function is obtained from
    https://omoindrot.github.io/triplet-loss#batch-all-strategy
    This function computes the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        tf.expand_dims(square_norm, 0)
        - 2.0 * dot_product
        + tf.expand_dims(square_norm, 1)
    )
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_multi_label_triplet_mask(triplet_loss, labels):
    """A modified version of the `_get_mask` function available at
    https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    that tries to implement the multi-label triplet loss available at
    https://github.com/abarthakur/multilabel-deep-metric/blob/master/src/utils.py

    Returns a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    
    In the original funnction, A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    When trying to accomodate multiple labels, there isn't a clear definition of whether
    a triplet is positive or negative. Instead we consider the number of matches in two
    multi-hot encoded labels which can be given by the dot product between two labels
    (we transpose one of them so that we get a scalar output).

    As in the repo by abarthakur, a valid triplet is one such that
        - i, j, k (anchor, positive, negative) are distinct
        - distance(a, p) + margin > distance(a, n)
        - num_matches(a, p) < num_matches(a, n)

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # We don't need a mask to fufil distance(a, p) + margin - distance(a, n) > 0
    # as it is already handled in the `batch_all_triplet_loss` function
    # valid_distances = tf.math.greater(triplet_loss, tf.constant([0], dtype=tf.float32))

    # Get mask for triplets that have less matching labels between the anchor and the
    # positive example than the anchor and the negative example
    # num_matches(a, p) < num_matches(a, n)
    num_match_matrix = tf.matmul(labels, labels, transpose_b=True)
    ij_matches = tf.expand_dims(num_match_matrix, 2)
    ik_matches = tf.expand_dims(num_match_matrix, 1)
    ij_less_than_ik = tf.math.less(ij_matches, ik_matches)

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, ij_less_than_ik)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """This function is obtained from
    https://omoindrot.github.io/triplet-loss#batch-all-strategy
    This function builds the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_multi_label_triplet_mask(triplet_loss, labels)
    mask = tf.cast(mask, dtype=tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), dtype=tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def select_triplets(
    embeddings, y_batch, max_negatives_per_pos, max_trips_per_anchor, factors
):
    triplets = []
    for i in range(0, embeddings.shape[0]):
        anchor = embeddings[i, :]
        anchor_y = y_batch[i, :]
        # get similarity scores
        sim_scores = np.sum(y_batch * (anchor_y * factors), axis=1)
        # get embedding distances
        dists = np.sqrt(np.sum((embeddings - anchor) ** 2, axis=1))
        distance_order = np.argsort(dists)
        sim_distance_order = sim_scores[distance_order]
        num_anchor_triplets = 0
        positive_idcs = np.nonzero(sim_distance_order > 0)[0]
        num_fine = 0
        num_coarse = 0
        # mine positives first, starting with the back
        for j, pos_idx in enumerate(np.flip(positive_idcs)):
            # its the anchor
            if distance_order[pos_idx] == i:
                continue
            pos_sim = sim_distance_order[pos_idx]
            # generate fine triplets
            positive_misorderings = np.logical_and(
                sim_distance_order[:pos_idx] < pos_sim, sim_distance_order[:pos_idx] > 0
            )
            for neg_idx in np.nonzero(positive_misorderings)[0]:
                triplets.append((i, distance_order[pos_idx], distance_order[neg_idx]))
                num_anchor_triplets += 1
                num_fine += 1
            # generate coarse triplets
            zero_idcs = np.nonzero(sim_distance_order[:pos_idx] == 0)[0]
            if len(zero_idcs) == 0:
                continue
            num_negatives = np.minimum(max_negatives_per_pos, zero_idcs.shape[0])
            for _ in range(0, num_negatives):
                # choose a negative randomly, because there are a lot of negatives
                # and since as we go down the positive_idcs, the previous zero_idcs
                # is included, so we don't want to keep choosing the same negatives
                k = np.random.randint(0, len(zero_idcs))
                neg_idx = zero_idcs[k]
                triplets.append((i, distance_order[pos_idx], distance_order[neg_idx]))
                num_anchor_triplets += 1
                num_coarse += 1
                print((i, distance_order[pos_idx], distance_order[neg_idx]))

            if num_anchor_triplets >= max_trips_per_anchor:
                break
    return triplets


def get_triplets(
    embeddings, y_batch, max_negatives_per_pos, max_trips_per_anchor, factors
):
    trips_list = select_triplets(
        embeddings, y_batch, max_negatives_per_pos, max_trips_per_anchor, factors
    )
    if len(trips_list) == 0:
        return None
    anchors = []
    positives = []
    negatives = []
    for (a, p, n) in trips_list:
        anchors.append(embeddings[a, :].reshape(1, embeddings.shape[1]))
        positives.append(embeddings[p, :].reshape(1, embeddings.shape[1]))
        negatives.append(embeddings[n, :].reshape(1, embeddings.shape[1]))
    anchors = tf.stack(anchors)
    positives = tf.stack(positives)
    negatives = tf.stack(negatives)
    return anchors, positives, negatives


class MultiLabelTripletSemiHard(tf.keras.losses.Loss):
    """An adapted Triplet Loss based on:
    https://github.com/abarthakur/multilabel-deep-metric
    """

    def __init__(
        self,
        margin=1.0,
        distance_metric="angular",
        squared=False,
        max_negatives_per_pos=3,
        max_num_triplets=100,
        name="multi_label_triplet_semi_hard",
        **kwargs
    ):
        """
        Args:
            margin: Float, margin term in the loss definition.
            distance_metric: (unused) `str` or a `Callable` that determines distance
                metric. Valid strings are "L2" for l2-norm distance,
                "squared-L2" for squared l2-norm distance, and "angular" for
                cosine similarity.

                A `Callable` should take a batch of embeddings as input and
                return the pairwise distance matrix.
            squared: Boolean. If true, output is the pairwise squared euclidean distance
                matrix. If false, output is the pairwise euclidean distance matrix.
            max_negatives_per_pos: Integer. Maximum number of negative examples per
                positive example.
            max_num_triplets: Integer. Maximum number of triplets to generate per batch.
        """
        super(MultiLabelTripletSemiHard, self).__init__(name=name, **kwargs)
        self.margin = margin
        self.distance_metric = distance_metric
        self.squared = squared
        self.max_negatives_per_pos = max_negatives_per_pos
        self.max_num_triplets = (max_num_triplets,)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: 2-D integer `Tensor` with shape `[batch_size, num_classes]` of
                multi-hot encoded class labels.
            y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
                be l2 normalized.
        Returns:

        """
        loss = batch_all_triplet_loss(
            y_true, y_pred, margin=self.margin, squared=self.squared
        )
        return loss
