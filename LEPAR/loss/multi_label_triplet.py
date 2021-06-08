import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def select_triplets(
    embeddings,
    y_batch,
    max_negatives_per_pos,
    max_trips_per_anchor,
    factors
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
    embeddings,
    y_batch,
    max_negatives_per_pos,
    max_trips_per_anchor,
    factors,
    debug=False,
):
    trips_list = select_triplets(
        embeddings,
        y_batch,
        max_negatives_per_pos,
        max_trips_per_anchor,
        factors,
        debug,
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
        max_negatives_per_pos=3,
        max_num_triplets=100,
        name="multi_label_triplet_semi_hard",
        **kwargs
    ):
        """
        Args:
            margin: Float, margin term in the loss definition.
            distance_metric: `str` or a `Callable` that determines distance
                metric. Valid strings are "L2" for l2-norm distance,
                "squared-L2" for squared l2-norm distance, and "angular" for
                cosine similarity.

                A `Callable` should take a batch of embeddings as input and
                return the pairwise distance matrix.
        """
        super(MultiLabelTripletSemiHard, self).__init__(name=name, **kwargs)
        self.margin = margin
        self.distance_metric = distance_metric
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
        factors = tf.zeros(y_true.shape[1]) + 1.0
        anchors, positives, negatives = get_triplets(
            y_pred, y_true, self.max_negatives_per_pos, self.max_num_triplets, factors
        )
        d_pos = tf.reduce_sum(tf.square(anchors - positives), 1)
        d_neg = tf.reduce_sum(tf.square(anchors - negatives), 1)
        loss = tf.maximum(0.0, self.margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)
        return loss
