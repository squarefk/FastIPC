import matplotlib.pyplot as plt
import pickle

_plotter_data = dict()


def Plotter_Record(key, value):
    if key not in _plotter_data:
        _plotter_data[key] = []
    _plotter_data[key].append(value)


def Plotter_Dump(directory, plot=False, binary=True):
    for key in _plotter_data:
        if plot:
            filename = directory + key + '.png'
            fig = plt.figure()
            plt.plot(_plotter_data[key])
            fig.savefig(filename)
            plt.close(fig)
        if binary:
            pickle.dump(_plotter_data[key], open(directory + key + '.p', 'wb'))
