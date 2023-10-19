import numpy as np
import torch

from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """
    This is was initally created to load the embeddings i.e the extracted features, to keep up with 
    the pytorch training classical style using a DataLoader.
    Nevertheless, we always load all the embeddings at once since they are light in memory.

    [Warning]: this is still doesn't support the case of Simple dataset, it assumes that the embds
    are serialized in a form of a dictionnary that contains the following keys:
        ['left', 'right', 'left_name', 'right_name']
    """
    def __init__(self, embs: dict, only_original=True):
        """
        Args:
            path (str): the serialized file that contains the extracted embeddings.
            only_original (bool, optional): if is true, it means that we are going to use only the
            first half of the embeddings, and that is because of the way we constructed the TTLDataset
            when it is augmented, the features of the original images are located at the first half
            then followed by their augmented versions.
            In short, if you want to load all the embds, weather it contains augmented version or not
            use only_original=False, otherwise it will load only the first half.
        """
        self.embds = self._load_embds(embs, only_original)
        self.augmented = not only_original
        
        assert "left" in self.embds and "left_name" in self.embds
        assert "right" in self.embds and "right_name" in self.embds
        assert len(self.embds["left"]) == len(self.embds["right"]) and len(
            self.embds["left"]
        ) == len(self.embds["left_name"])
        assert len(self.embds["right"]) == len(self.embds["right_name"])

    def _load_embds(self, embs: dict, only_original : bool) -> dict:
        # e = np.load(path, allow_pickle=True).reshape(-1)[0]
        # print(e)
        # if 'left_ebds' in e.keys():
        #     e['left'] = e['left_ebds']
        #     e['right'] = e['right_ebds']

        # if only_original:
        #     length = len(e['left'])
        #     e['left'] = e['left'][: length // 2]
        #     e['left_name'] = e['left_name'][: length // 2]
        #     e['right'] = e['right'][: length // 2]
        #     e['right_name'] = e['right_name'][: length // 2]

        return embs

    def load_all_to_device(self, device):
        embds, _ = self[:]
        embds['left'].to(device)
        embds['right'].to(device)

        return embds

    def __getitem__(self, i):
        left, right = self.embds["left"][i], self.embds["right"][i]
        left, right = torch.from_numpy(left).float(), torch.from_numpy(right).float()

        return ({"left": left, "right": right}, self.embds["left_name"][i])

    def __len__(self):
        return len(self.embds["left"])
    