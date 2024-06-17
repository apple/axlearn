# pylint: disable=invalid-name
"""ImageNet utils."""
import pickle

import numpy as np

# Obtained by OpenAI model's estimation, include
#   - target2esti: for every category, obtain its most frequent confused categories
#   - S: (1000x1000) np array, S = text_embedding.T * text_embedding
FN_OPENAI_TARGET2ESTI = "openai_clip_pred_1tfrecord_target2esti.pickle"


class ImageNet_SimilarClass:
    """Identifies similar classes."""

    def __init__(self) -> None:
        with open(FN_OPENAI_TARGET2ESTI, "rb") as handle:
            target2esti, S = pickle.load(handle)
        self.target2esti = target2esti
        self.S = S

    def most_similar_class(self, c):
        mylist = None
        if c in self.target2esti:
            mylist = [tmp for tmp in self.target2esti[c] if tmp != c]
        if mylist:
            return max(set(mylist), key=mylist.count)

        # If there is not confused class, choose the similar from text embeddings.
        sc = self.S[c]
        sc[c] = 0
        return np.argmax(sc)
