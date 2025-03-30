import numpy as np

def normal(loc: np.ndarray, scale: float, shape: tuple[int,int]) -> np.ndarray:
    x = np.random.normal(0, scale, shape)
    x[0, :] += loc[0]
    return x

def gen_dataset(size: int, st_pos: tuple[float, float], n_classes : float, spasing : float, scale: float):
    x1_class_pos = np.ones((n_classes, 1)) * st_pos[0]
    x2_class_pos = np.ones((n_classes, 1)) * st_pos[1]

    if (n_classes >= 2):
        x1_class_pos[1] += spasing
    
    if (n_classes >= 3):
        x2_class_pos[2] += spasing

    if (n_classes == 3):
        x1_class_pos[2] += spasing / 2       
    
    if (n_classes >= 4):
        x1_class_pos[3] += spasing
        x2_class_pos[3] += spasing

    x1 = np.zeros((n_classes, size))
    x2 = np.zeros((n_classes, size))
    for i in range(n_classes):
        x1[i, :] = normal(x1_class_pos[i], scale, (1, size))
        x2[i, :] = normal(x2_class_pos[i], scale, (1, size))

    y = np.arange(n_classes)
    y = y.repeat(size)

    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)

    x = np.concatenate([x1, x2], 0)

    shuffle_idx = np.random.choice(
        np.arange(x.shape[1]), x.shape[1], replace=False)
    x[0, :] = x[0, shuffle_idx]
    x[1, :] = x[1, shuffle_idx]
    y = y[shuffle_idx]

    return x, y