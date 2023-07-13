import numpy as np

def Dict_1D(ps, nd):
    Dict = np.zeros((ps, nd))
    for i in range(nd):
        atom = np.cos(np.arange(0, ps) * i * np.pi / nd)
        if i > 0:
            atom = atom - np.mean(atom)
        Dict[:, i] = atom / np.linalg.norm(atom)

    return Dict


def Dict_4D(patch_shape, sparse_shape):
    Dict = np.array([1])
    for i in range(4):
        D = Dict_1D(patch_shape[i], sparse_shape[i])
        Dict = np.kron(Dict, D)
    ds = Dict.shape
    Dict = Dict.reshape(patch_shape[0],patch_shape[1],patch_shape[2],patch_shape[3],-1)
    Dict = Dict.transpose(3,2,1,0,4)
    Dict = Dict.reshape(ds)
    Dict = Dict.dot(np.diag(1 / np.sqrt(np.sum(Dict ** 2, axis=0))))
    return Dict