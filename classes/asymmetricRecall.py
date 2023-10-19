import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class AsymmetricRecall:
    """
    This class is responsible for calculating the topk matches to our query.
    """
    def __init__(self, left_ebds, right_ebds):

        self.M = -cosine_similarity(left_ebds, right_ebds)
        self.sorted_idx_1 = self.M.argsort(1)
        self.sorted_idx_2 = self.M.T.argsort(1)

    def eval(self, at):
        sens1 = np.array(
            [lbl in self.sorted_idx_1[lbl][:at] for lbl in range(len(self.M))]
        )
        sens2 = np.array(
            [lbl in self.sorted_idx_2[lbl][:at] for lbl in range(len(self.M))]
        )

        return (sens1 | sens2).mean()