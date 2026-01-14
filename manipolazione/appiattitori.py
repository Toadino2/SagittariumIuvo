import numpy as np

def appiattisci(array: np.ndarray | list):
    if isinstance(array, np.ndarray):
        return array.flatten()
    else:
        if isinstance(array[0], list) or isinstance(array[0], set):
            return [elemento for lista in array for elemento in lista]
        else:
            return array