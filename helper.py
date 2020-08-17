import numpy as np
import math
from haversine import haversine
import matplotlib.pyplot as plt
import scipy.optimize as optimize

colors = "krgbmkrgbm"

developing = ["Kiev", "Bandung", "Surabaya", "Yogyakarta", "Jakarta"]
developed = ["Seoul", "Tokyo", "Singapore", "NewYorkCity", "Chicago"]

# This is Tassos's code
def bin_data(data, min_value, max_value=10**6, STEPS=50):
    """
    Filters the values contained in data, removing all values smaller than
    min_value and all values larger than max_value.
    """

    small = min(data)
    N = len(data)
    if small == 0:
        print 'Data with zero values: filtering out them'
        data = filter(lambda x: x > 0, data)
        small = min(data)

    if min_value:
        print 'Filtering out values smaller than ', min_value
        data = filter(lambda x: x >= min_value, data)

    if max_value:
        print 'Filtering out values larger than ', max_value
        data = filter(lambda x: x <= max_value, data)

    MIN_D = math.log10(min(data))
    MAX_D = math.log10(max(data))

    def scale_distance(s):
        """Logarithmic binning"""
        v1 = int(s*30)
        v2 = float(v1)/30
        l = math.log10(s)
        i = (l-MIN_D)/(MAX_D-MIN_D)
        v = float(int(STEPS*i))/STEPS
        v2 = MIN_D + v*(MAX_D-MIN_D)
        return 10**v2

    data = map(scale_distance, data)
    N2 = len(data)
    print "%d initial values, %d final values" % (N, N2)

    v = {}
    for x in data:
        v.setdefault(x, 0)
        v[x] += 1

    x = np.array(sorted(v), dtype='float')
    y = [v[k] for k in x]
    tot = sum(y)

    y = [1.0*i/tot for i in y]

    # Bin normalization
    for i in range(len(x) - 1):
        y[i] = y[i]/(x[i+1]-x[i])

    x = x[:-1]
    y = y[:-1]

    # pdf value normalization
    tot = sum(y)
    y = [1.0*i/tot for i in y]

    return x, y

def get_distribution(data, min_value, max_value=10**6, STEPS=50):
    """ 
    Not great style to do this, I know, but this is the same as the above
    but not pdf normalized 
    """
    small = min(data)
    N = len(data)
    if small == 0:
        print 'Data with zero values: filtering out them'
        data = filter(lambda x: x > 0, data)
        small = min(data)

    if min_value:
        print 'Filtering out values smaller than ', min_value
        data = filter(lambda x: x >= min_value, data)

    if max_value:
        print 'Filtering out values larger than ', max_value
        data = filter(lambda x: x <= max_value, data)

    MIN_D = math.log10(min(data))
    MAX_D = math.log10(max(data))

    def scale_distance(s):
        """Logarithmic binning"""
        v1 = int(s*30)
        v2 = float(v1)/30
        l = math.log10(s)
        i = (l-MIN_D)/(MAX_D-MIN_D)
        v = float(int(STEPS*i))/STEPS
        v2 = MIN_D + v*(MAX_D-MIN_D)
        return 10**v2

    data = map(scale_distance, data)
    N2 = len(data)
    print "%d initial values, %d final values" % (N, N2)

    v = {}
    for x in data:
        v.setdefault(x, 0)
        v[x] += 1

    x = np.array(sorted(v), dtype='float')
    y = [v[k] for k in x]
    tot = sum(y)

    y = [1.0*i/tot for i in y]

    # Bin normalization
    for i in range(len(x) - 1):
        y[i] = y[i]/(x[i+1]-x[i])

    x = x[:-1]
    y = y[:-1]

    return x, y


def powerlaw_function(x, N, a):
        return N * x ** (-a)


def get_path(city, path):
    t = path + "transitions/" + city + "_transitions.txt"
    v = path + "venues/" + city + "_venues.txt"
    return t, v


def create_venues_dict(handler):
    coords = {}
    with open(handler, 'r') as f:
        for rows in f:
            rows = rows.split("\t")
            ID = rows[0]
            coordinates = (float(rows[2]), float(rows[3]))
            coords[ID] = coordinates
    return coords


def plot_pdf(glob, xlabel):
    x = []
    y = []
    i = 0
    for f in glob:
        name = f.split('_')
        name = name[2].split('.csv')
        name = name[0]
        print name
        for l in open(f, "r"):
            l = l.split(',')
            x.append(float(l[0]))
            y.append(float(l[1]))
        col = colors[i]
        if name in developed:
            col += "--"
        plt.loglog(x, y, col, label=name)
        x = []
        y = []
        i += 1
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('pdf')
    plt.show()

def read_coords_in(f):
    x = []
    y = []
    for l in open(f, "r"):
        l = l.split(',')
        x.append(float(l[0]))
        y.append(float(l[1]))
    return x, y
 
def read_data_in(f):
    handler = open(f, "r")
    data = []
    for val in handler:
        val = val.strip()
        if val:
            data.append(float(val))
    return data


def fit_data(x, y):
    xdata = x
    ydata = y
    fitfunc = lambda p, x: p[0] * (x+p[1])**(-p[2])
    pinit = [1.0,1.0,1.0] 

    errfunc = lambda p, x, y: np.log(y) - np.log(fitfunc(p, x))
    out = optimize.leastsq(errfunc, pinit, args=(xdata, ydata), xtol=1e-12)

    pfinal = out[0]
    fitted = np.array([fitfunc(pfinal,i) for i in xdata])
    fit_label = r'$%.2f (x+%.2f)^{-%.2f}$'%(pfinal[0],pfinal[1],pfinal[2])

    return fitted, fit_label, pfinal[2]
