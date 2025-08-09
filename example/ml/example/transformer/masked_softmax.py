import torch
import torch.nn as nn   
from d2l import torch as d2l

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_utils`"""
    maxlen = X.size(1)
    v = valid_len[:, None]
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] # < valid_len[:, None]
    mask = mask < v
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        val = nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        val = nn.functional.softmax(X.reshape(shape), dim=-1)
    return val

X = torch.rand(2, 2, 4)
print(X)
valid_lens = torch.tensor([2, 3])
print(valid_lens)
d = masked_softmax(X, valid_lens)
print(X.size(1))
d2l.DotProductAttention()
d2l.AdditiveAttention()
X = torch.rand(2, 2, 4)
d2l.AdditiveAttention