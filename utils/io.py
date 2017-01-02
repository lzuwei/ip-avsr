import sys
import scipy.io as sio
import lasagne as las
sys.path.insert(0, '../')
try:
    import cPickle as pickle
except:
    import pickle


def read_data_split_file(path, sep=','):
    with open(path) as f:
        subjects = f.readline().split(sep)
        subjects = [int(s) for s in subjects]
    return subjects


def load_mat_file(path):
    """
    Loads .mat file
    :param path: path to .mat file
    :return: dictionary containing .mat data
    """
    return sio.loadmat(path)


def save_mat(dict, path):
    print('save matlab file...')
    sio.savemat(path, dict)


def save_model(model, path):
    pickle.dump(model, open(path, 'wb'))


def load_model(path):
    return pickle.load(open(path, 'rb'))


def save_model_params(network, path):
    all_param_values = las.layers.get_all_param_values(network)
    pickle.dump(all_param_values, open(path, 'wb'))


def load_model_params(network, path):
    all_param_values = pickle.load(open(path, 'rb'))
    las.layers.set_all_param_values(network, all_param_values)
    return network
