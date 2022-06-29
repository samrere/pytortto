from numpy import save as np_save, load as np_load
def save(obj, f, allow_pickle=True):
    data = {'data': obj}
    # save state dict as numpy array
    np_save(f, data, allow_pickle=allow_pickle)

def load(file, mmap_mode=None, allow_pickle=True,encoding='ASCII'):
    loaded = np_load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle,encoding=encoding).item()['data']
    return loaded