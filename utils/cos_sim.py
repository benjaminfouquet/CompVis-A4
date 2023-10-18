import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
import time


def cos_sim(left, candidate):
    """Computes the cosine similarity between 2 tensors

    Arguments:
        left:   torch.Size([1, 245, 200, 3])
        candidate: torch.Size([1, 245, 200, 3])
    """

    return F.cosine_similarity(left.reshape(1, -1), candidate.reshape(1, -1)).item()


def cos_sim_array(left, candidates):
    """Computes [cos_sim(left, candidate0), ...]

    Arguments:
        left: torch.Size([1, 245, 200, 3])
        candidates: torch.Size([1, 20, 245, 200, 3]) - 1 batch of 20 tensors
                    representing 20 candidates of torch.Size([1, 245, 200, 3])
    """
    return np.array([cos_sim(left, candidate) for candidate in candidates[0]])


def cos_sim_make_output(test_loader, test_candidates_csv):
    """Make the output csv file for cosine similarity, and return the dataframe"""
    column_names = ["left"] + [f"c{i}" for i in range(20)]
    test_candidates = pd.read_csv(test_candidates_csv)

    df = []
    start_time = time.time()  # Record the start time
    for i, data in enumerate(test_loader):
        result = [test_candidates.iloc[i, 0]]
        left, candidates = data
        # convert from uint8 to float32
        left = left.type(torch.float32)
        candidates = candidates.type(torch.float32)
        # calculate cosine similarity
        cos_sim_arr = cos_sim_array(left, candidates)
        result += cos_sim_arr.tolist()
        df.append(result)

        if (i + 1) % 200 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processed {i+1} samples in {elapsed_time:.2f} seconds")

    df = pd.DataFrame(df, columns=column_names)
    df.to_csv("output/cos_sim.csv", index=False)

    return df
