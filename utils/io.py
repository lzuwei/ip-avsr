import sys
import scipy.io as sio
sys.path.insert(0, '../')
try:
    import cPickle as pickle
except:
    import pickle


def load_mat_file(path):
    """
    Loads .mat file
    :param path: path to .mat file
    :return: dictionary containing .mat data
    """
    return sio.loadmat(path)


def save_model(model, path):
    pickle.dump(model, open(path, 'wb'))


def load_model(path):
    return pickle.load(open(path, 'rb'))
