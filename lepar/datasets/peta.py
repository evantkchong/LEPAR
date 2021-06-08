import os

import tensorflow as tf
import numpy as np

from scipy.io import loadmat


class PETAGenerator:
    """A class that parses the PETA dataset provided by
    https://github.com/dangweili/pedestrian-attribute-recognition-pytorch and applies
    the dataset transformations made in his `transform_peta.py` script

    Args:
        dataset_dir: (Optional) Directory containing the unzipped PETA dataset.
        img_size: (Optional) What width and height to resize the images to.
        partition_type: (Optional) A string indicating which dataset partition to use.
            Allowed values are "train", "valid", "test" and "trainval".
    """

    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "raw")

    def __init__(
        self,
        dataset_dir=DEFAULT_DATA_DIR,
        img_size=224,
        partition_type="trainval",
    ):
        self.dataset_dir = dataset_dir
        self.partition_type = partition_type
        self.img_size = img_size
        self.attribute_map = None
        self.label_weights = None
        self.labels, self.file_indexs = self._read_mat()

    def _read_mat(self):
        data = loadmat(os.path.join(self.dataset_dir, "PETA.mat"))
        partition_index = 0
        multi_hot_label_list = []
        image_idx_list = []
        attribute_idx_list = list(range(35))

        # Attribute names
        self.attribute_map = [
            data["peta"][0][0][1][idx, 0][0] for idx in attribute_idx_list
        ]
        mat_annotations = np.array(data["peta"][0][0][0])
        # Annotations
        for idx in range(len(mat_annotations)):
            multi_hot_label_list.append(
                mat_annotations[idx, 4:][attribute_idx_list].tolist()
            )

        partitions = data["peta"][0][0][3][partition_index][0][0][0]
        train = (partitions[0][:, 0] - 1).tolist()
        val = (partitions[1][:, 0] - 1).tolist()
        test = (partitions[2][:, 0] - 1).tolist()

        partition_str = self.partition_type.lower()
        if partition_str == "trainval":
            image_idx_list = train + val
        elif partition_str == "train":
            image_idx_list = train
        elif partition_str == "valid":
            image_idx_list = val
        elif partition_str == "test":
            image_idx_list = test
        else:
            msg = (
                "Allowed values for `partition_type` are `train`, `valid`, `test` and ",
                "`trainval`",
            )
            raise ValueError(msg)

        weight = np.mean(
            data["peta"][0][0][0][image_idx_list, 4:].astype("float32") == 1, axis=0
        ).tolist()
        self.label_weights = np.array(weight)[attribute_idx_list].tolist()

        return multi_hot_label_list, image_idx_list

    def _read_image(self, filename):
        img = tf.io.read_file(filename)
        img = tf.io.decode_image(img, channels=3)
        if self.img_size is not None:
            img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.cast(img, tf.uint8)
        return img.numpy()

    def parse(self):
        """Use this method with tf.data.Dataset.from_generator()"""
        for label, index in zip(self.labels, self.file_indexs):
            # The image files are 5-digit numbered pngs
            filepath = os.path.join(
                self.dataset_dir, "images", "{0:0=5d}.png".format(index)
            )
            print(filepath)
            image = self._read_image(filepath)
            yield image, label


if __name__ == "__main__":
    parser = PETAGenerator()
    for img, label in parser.parse():
        print(img, label)
