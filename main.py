# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import mlrose_hiive as mlr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM


class Logs():
    cur_log_file = None
    log_time_tag = None
    log_dir = './logs'

    def set_log_file(tag, src_level= -2):
        Logs.cur_log_file = Logs.get_log_fn(tag=tag, src_level=src_level)

    def get_log_fn(tag=None, src_level=-2):

        if tag is None:
            return Logs.cur_log_file

        import datetime, os

        Logs.log_time_tag = datetime.datetime.now().strftime("%m_%d_%H_%M")

        log_time_tag_parts = Logs.log_time_tag.split('_')
        log_start = 15 * (int(log_time_tag_parts[-1]) // 15)
        day = get_day_prefix(Logs.log_time_tag)

        log_time_tag_parts[-1] = str(log_start)

        fn = tag + '_log_' + '_'.join(log_time_tag_parts) + '.txt'
        target = os.sep.join([Logs.log_dir, day])
        if not os.path.exists(target):
            os.makedirs(target)

        log_fn = os.sep.join([Logs.log_dir, day, fn])
        return log_fn

def log(s=None, d=None, src_level=-2):
    def col_print(d, cols=4, proc=lambda i, x, d:str(x)):
        cols_out = []
        last = len(d) - 1
        for i, val in enumerate(d):
            val = proc(i, val, d)
            cols_out.append(val)
            if i == 0 and i != last:
                continue
            if i % cols == cols - 1:
                log_file.write('\t' + '   '.join(cols_out) + '\n')
                cols_out = []
            elif i == last:
                if len(cols_out) == 1:
                    log_file.write('\t' + cols_out[0] + '\n')
                elif len(cols_out) > 1:
                    log_file.write('\t' + cols_out[0] + '\n')

    log_fn = Logs.get_log_fn()

    import inspect, os
    recs = inspect.stack()
    path = str(recs[1][0].f_code).split('"')[src_level]

    lineNum = recs[1][0].f_lineno
    where = os.path.basename(path)
    where += ":" + str(lineNum)
    when = Logs.log_time_tag + ": "
    when += where + "-"

    with open(log_fn, 'a+') as log_file:
        log_file.write(when + s + '\n')
        special_handling = ['m_params', 'vc_params_perf', 'train_scores', 'test_scores', 'plot_data']
        if d is None:
            pass
        elif type(d) is list or str(type(d)) == "<class 'numpy.ndarray'>":
            if len(d) <= 1:
                log_file.write(str(d))
            else:
                col_print(d)
        elif type(d) is dict:
            proc = lambda i, x, d : str(x) + ': ' + str(d[x])
            col_print(d, proc=proc)
            print(d)

    if type(d) is dict and len([x for x in d if x in special_handling]) > 0:
        if 'ds' in d:
            ds = d['ds']
        else:
            ds = 'no_ds_given'

        if 'plot' in d:
            plot_type = d['plot']
        else:
            plot_type = 'no_pt_specified'

        log_special = '.'.join(log_fn.split('.')[0:-1]) + '_' + list(d)[0] + '_' + ds + '_' + plot_type + '.csv'
        d = d[list(d)[0]]  # de-nest, get to the actual data

        if type(d) is list:
            out_cols = [i for i, x in enumerate(d)]
        elif str(type(d)) == "<class 'numpy.ndarray'>":
            out_cols = [i for i, x in enumerate(d)]
        else:
            out_cols = list(d)

        import csv
        if not os.path.exists(log_special):
            with open(log_special, 'w', newline='') as log_file:
                wr = csv.writer(log_file)
                wr.writerow(['where', 'when', 'msg'] + out_cols)

        col_2 = s  # ' ' * len(where)
        col_0 = where
        col_1 = Logs.log_time_tag
        with open(log_special, 'a+', newline='') as log_file:
            wr = csv.writer(log_file)

            if d is dict or isinstance(d, dict):
                out_vals = [d[k] for k in out_cols]
            elif type(d) is list:
                out_vals = d
            elif str(type(d)) == "<class 'numpy.ndarray'>":
                out_vals = d.tolist()
            else:
                hook = True

            wr.writerow([col_0, col_1, col_2] + out_vals)

def get_day_prefix(now_str=None):
    import datetime
    if now_str is None:
        now_str = datetime.datetime.now().strftime("%m_%d_%H_%M")

    time_tag_parts = now_str.split('_')
    return "_".join(time_tag_parts[:2])

def get_plot_fn(tag=None, root_path='./plots'):
    import datetime, os
    time_tag = datetime.datetime.now().strftime("%m_%d_%H_%M")
    day = get_day_prefix(time_tag)
    target = os.sep.join([root_path, day])
    if not os.path.exists(target):
        os.makedirs(target)


    fn = tag + '_' + time_tag + ".png"
    full_fn = os.sep.join([root_path, day, fn])

    no_overwrite = 0
    test_fn = full_fn
    while os.path.exists(test_fn):
        no_overwrite += 1
        pos = len(full_fn) - 1
        while pos >= 0 and full_fn[pos] != '.':
            pos += -1

        fn_parts = full_fn[0:pos]

        part1 = fn_parts + "_" + str(no_overwrite)
        test_fn = part1 + '.png'

    full_fn = test_fn

    return full_fn

def get_converge(npl, ep=.0001):
    cnv = 0
    cnv_val = npl[0]

    for i, x in enumerate(npl):
        if x > (1-ep)*cnv_val and x < (1+ep)*cnv_val:
            continue
        else:
            cnv = i
            cnv_val = npl[i]

    return cnv, cnv_val

def get_plot_xmax(x, y):
    y_max_idx = np.argmax(y)
    y_min_idx = np.argmin(y)
    conv_idx, val = get_converge(y)
    idxs = [y_max_idx, conv_idx, y_min_idx]
    which_idx = np.argmax( [x[ii] for ii in idxs] )
    target_idx = idxs[which_idx]
    # x vals should be monotonic, so included 2 extra points should not result in a sub-optimal x value
    if target_idx < len(x) - 2:
        target_idx += 2

    x_max = x[target_idx]

    return x_max

def pr_fl(f, len=8):
    return str(f)[0:len]

def get_ranges(data):
    hook = True

    for k in data:
        if len(data[k].shape) == 1:
            col2 = [[x+1] for x in range(data[k].shape[0])]
            data[k] = np.column_stack((data[k], col2))
            hook = True

    #massage = lambda x, l: x + 2 if x + 2 < len(l) else x
    y_idx_at_max = [get_plot_xmax(x=data[k][:, 1], y = data[k][:, 0]) for k in data]


    all_max_x = max(y_idx_at_max)
    all_min_x = min([np.min(data[k][:, 1]) for k in data])

    all_max_y = max([np.max(data[k][:, 0]) for k in data])
    all_min_y = min([np.min(data[k][:, 0]) for k in data])

    return all_min_x, all_max_x, all_min_y, all_max_y

BSK = 'best_state'
BFK = 'best_fitness'
BFCK = 'best_fc_curve'

def get_plot_title(pr, dm, dim):
    return f"{pr}-{dm} size:{dim}"

def make_fitness_plots(
        title=None,  data=None, fn=None, rows=1, extra_data=None,
        x_label="iterations", y_label="fitness score", trunc_converged=True,
        zero_x_bnd=False, zero_y_bnd=False
    ):

    plt.close()
    fig, axes = plt.subplots(rows, 1, squeeze=False, figsize=(10, 10))

    if hasattr(axes, 'shape'):
        subplot_1 = axes[0, 0]

    all_min_x, all_max_x, all_min_y, all_max_y = get_ranges(data)

    lb_factor = .95
    ub_factor = 1.05
    if zero_x_bnd:
        xp_min = min(0, lb_factor * all_min_x)
    else:
        xp_min = lb_factor * all_min_x

    if zero_y_bnd:
        yp_min = min(0, lb_factor * all_min_y)
    else:
        yp_min = lb_factor * all_min_y

    x_lb = xp_min
    x_ub = ub_factor * all_max_x
    y_lb = yp_min
    y_ub = ub_factor * all_max_y

    print(f"Plot bounds = x:{pr_fl(x_lb)},{pr_fl(x_ub)}, y:{pr_fl(y_lb)},{pr_fl(y_ub)}")
    subplot_1.set_xlim(x_lb, x_ub)
    subplot_1.set_ylim(y_lb, y_ub)
    subplot_1.set_title(title)
    subplot_1.set_xlabel(x_label)
    subplot_1.set_ylabel(y_label)

    subplot_1.grid()
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    syms = ['+', '1', '2', '3', '4']

    for i, k in enumerate(data):
        log(f"{title}raw plot {k}", {'plot_data': data[k]})

        cur_x = data[k][:, 1]
        cur_y = data[k][:, 0]
        if trunc_converged:
            y_idx_at_conv, _ = get_converge(cur_y)
            y_idx_at_conv = min(len(cur_y) - 1, y_idx_at_conv + 2)
            cur_x_data_2 = cur_x[0: y_idx_at_conv + 1]
            cur_y_data_2 = cur_y[0: y_idx_at_conv + 1]

            cur_x = cur_x_data_2
            cur_y = cur_y_data_2

        log(f"{title} processed plot {k} x:", cur_x)
        log(f"processed plot {k} y:", cur_y)

        extra_text = 'None'
        lbl = k
        if not extra_data:
            pass
        else:
            if k in extra_data:
                extra_text = extra_data[k]
                buffer = '' if extra_text[0].isspace() else ' '
                lbl = k + buffer + extra_text

        log(f"plot extra data {k}:" + extra_text)

        sylen = len(syms)
        pt = syms[i%sylen]
        clen = len(colors)
        subplot_1.plot(cur_x, cur_y, color=colors[i%clen], marker=pt, linestyle='solid', label=lbl)

    subplot_1.legend(loc="lower right")

    fig.tight_layout()
    plt.margins(10, 0)

    plt.savefig(fn)
    plt.show()

def get_mmc_args(curve=True, pop_size=None, keep_pct=None):
    #problem, pop_size = 200, keep_pct = 0.2, max_attempts = 10,
    # max_iters = inf, curve = False, random_state = None, fast_mimic = False
    #args = mlr.DiscreteOpt(length=8, fitness_fn=fitness, maximize=True, max_val=8)
    kwa = {
        'pop_size':200, 'keep_pct': 0.2,
        'max_attempts': 10,
        'max_iters': 1000, 'curve': curve,
        'random_state': 1
    }
    if not pop_size:
        pass
    else:
        kwa['pop_size'] = pop_size

    if not keep_pct:
        pass
    else:
        kwa['keep_pct'] = keep_pct

    return kwa


def get_hc_args(curve=True):

    kwa = {
        'restarts': 0,
        'max_iters': 1000, 'curve': curve,
        'random_state': 1
    }
    #problem, max_iters = inf, restarts = 0, init_state = None, curve = False, random_state = None
    return kwa

def get_ga_args(curve=True, mutation_prob=None, pop_size=None):

    kwa = {
        'pop_size':200,
        'mutation_prob':0.1,
        'max_attempts': 100, 'max_iters': 1000, 'curve': curve,
        'random_state': 1
    }
    if not pop_size:
        pass
    else:
        kwa['pop_size'] = pop_size

    if not mutation_prob:
        pass
    else:
        kwa['mutation_prob'] = mutation_prob

    return kwa


def get_rhc_args(curve=True):

    kwa = {
        'max_attempts': 1000, 'max_iters': 1000, 'curve': curve,
        'init_state': None, 'random_state': 1
    }
    return kwa

def  get_sa_args(curve=True, schedule=None):
    kwa = {'schedule': mlr.ExpDecay(),
        'max_attempts': 100, 'max_iters': 1000, 'curve': curve,
        'init_state': None, 'random_state': 1
    }
    if not schedule:
        pass
    else:
        kwa['schedule']=schedule()

    return kwa

DISCR= 'discrete'
CONT= 'cont'
def get_prob_obj(dm_tag=None, fit_tag=None, dim=None):

    max_value = 2
    # if dm_tag == DISCR:
    #     if fit_tag == KNSACK:
    #         w, v = get_knapsack_wvs(length=dim)
    #     elif fit_tag == MXK:
    #         e = get_mxk_edges(length=dim)
    #     elif fit_tag == TSP:
    #         assert(False)
    #         c, d = get_tsp_cd()
    #         fitness_coords = mlr.TravellingSales(coords=c)
    #         return mlr.TSPOpt(length=length, fitness_fn=fitness_coords, maximize=False)

    problem = mlr.DiscreteOpt(
        length=dim, fitness_fn=get_fitness(fit_tag, dim=dim),
        maximize=True, max_val=max_value
    )

    # elif dm_tag == CONT:
    #     assert(False)
    #     if fit_tag in [QUEENS, KNSACK, MXK, FLFLOP, F4PEAKS, S6PEAKS, CONT_PEAKES, TSP]:
    #         return None
    #     problem = mlr.ContinuousOpt(length=8, fitness_fn=get_fitness(fit_tag), maximize=True, max_val=8)

    return problem

DS_1 = 'digits'
DS_2 = 'Bach'
DS_3 = 'Wine'

CODE_TEST_1 = 1
CODE_TEST_2 = 2
CODE_TEST_3 = 3

ds_names = {
    CODE_TEST_1:DS_1,
    CODE_TEST_2 : DS_2,
    CODE_TEST_3 : DS_3
}


def load_datasets(dset=1):
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_wine

    from numpy import genfromtxt
    from datetime import datetime as dt

    def yn2num(val):
        assert (val[0].lower() in ['y', 'n'])
        res = 1 if val.lower()[0] == 'y' else 0
        return res

    def note2num(val):
        letter = val[0].lower()
        notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        index = notes.index(letter)
        res = float(index)

        if len(val) > 1:
            if val[1] == 'b':
                res -= .5
            elif val[1:3].lower() == 'sh':
                res += .5
            else:
                assert (False)

        assert (res >= 0)
        return res

    def colzero(val):
        decoded = val.decode("utf-8")
        return decoded

    loaded = []
    incl_digits = False
    incl_wine = False

    # CODE_TEST_2 for Bach
    if dset == CODE_TEST_1:
        incl_digits = True
    elif dset == CODE_TEST_3:
        incl_wine = False
    elif dset != CODE_TEST_2:
        assert(False)

    d1 = dict()
    cur_ds = d1

    if incl_digits:
        X, y = load_digits(return_X_y=True)
        cur_ds['X'] = X
        cur_ds['Y'] = y
        cur_ds['name'] = DS_1
        loaded.append(cur_ds)
        return loaded

    if incl_wine:
        X, y = load_wine(return_X_y=True)
        ds2 = dict()
        ds2['X'] = X
        ds2['Y'] = y
        ds2['name'] = DS_3
        # unq_1, cnts_1 = np.unique(ds1['Y'], return_counts=True)
        #nq_2, cnts_2 = np.unique(ds2['Y'], return_counts=True)

        #log(f'For dataset {DS_1} unique = {unq_1}, counts = {cnts_1}')
        #log(f'For dataset {DS_2} unique = {unq_2}, counts = {cnts_2}')
        loaded.append(ds2)

        return loaded

    # d1 = dict()

    converters = dict()
    for i in range(17):
        converters[i] = colzero

    try:
        temp = genfromtxt(
            'data/jsbach_chorals_harmony/jsbach_chorals_harmony.data', delimiter=',',
            converters=converters
        )
    except Exception as ex:
        debug_hook = True
        import sys
        _, _, tb = sys.exc_info()
        raise ()
        pass

    labels = []
    pieces = set()
    d3 = dict()
    cur_ds = d3
    cur_ds['X'] = np.zeros((len(temp), len(temp[0]) - 2))
    # cur_ds['Y'] = np.zeros((len(temp), 1))

    for i, row in enumerate(temp):
        for j in range(len(row) - 1):
            if j == 0:
                pieces.add(row[j])
                continue

            if j > 0:
                if j in (1, 15):
                    cur_ds['X'][i, j - 1] = float(row[j])
                elif 2 <= j <= 13:
                    cur_ds['X'][i, j - 1] = yn2num(row[j])
                elif j == 14:
                    cur_ds['X'][i, j - 1] = note2num(row[j])
                else:
                    assert (False)

        labels.append(row[-1])

    label_index = set(labels)
    label_index = list(label_index)
    label_index.sort()

    label_count_1 = dict()
    for s in label_index:
        label_count_1[s] = labels.count(s)

    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import LabelEncoder
    # lb = LabelBinarizer()
    le = LabelEncoder()
    cur_ds['Y'] = le.fit_transform(labels)
    unq_1, cnts_1 = np.unique(cur_ds['Y'], return_counts=True)
    cur_ds['name'] = DS_2
    loaded.append(cur_ds)

    return loaded


def run_find_weights(algo=None):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder

    datasets = load_datasets(CODE_TEST_1)
    cur_ds = datasets[0]

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(cur_ds['X'])
    y_train = cur_ds['Y']

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()

    nn_model1 = mlr.NeuralNetwork(
        hidden_nodes=[37, 37], activation='relu',
        algorithm=algo, max_iters=1000,
        bias=True, is_classifier=True, learning_rate=0.00179,
        early_stopping=True, clip_max=5, max_attempts=100,
        random_state=3,
        curve=True
    )

    ret = nn_model1.fit(X_train_scaled, y_train_hot)

    return ret

def get_digits():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder

    datasets = load_datasets(CODE_TEST_1)
    cur_ds = datasets[0]

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(cur_ds['X'])
    y_train = cur_ds['Y']

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    return X_train_scaled, y_train_hot

def get_bach():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder

    datasets = load_datasets(CODE_TEST_2)
    cur_ds = datasets[0]

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(cur_ds['X'])
    y_train = cur_ds['Y']

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    return X_train_scaled, y_train_hot

def get_bach():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import OneHotEncoder

    datasets = load_datasets(CODE_TEST_2)
    cur_ds = datasets[0]

    # Normalize feature data
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(cur_ds['X'])
    y_train = cur_ds['Y']

    # One hot encode target values
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
    return X_train_scaled, y_train_hot

def run_gmm(k=2, get_ds=None, start=2, npoints=10, limit=None):
    # code based on https://github.com/vlavorini/ClusterCardinality/blob/master/Cluster%20Cardinality.ipynb
    # n_clusters = np.arange(2, 20)
    # sils = []
    # sils_err = []
    # iterations = 20
    # for n in n_clusters:
    #     tmp_sil = []
    #     for _ in range(iterations):
    #         gmm = GMM(n, n_init=2).fit(embeddings)
    #         labels = gmm.predict(embeddings)
    #         sil = metrics.silhouette_score(embeddings, labels, metric='euclidean')
    #         tmp_sil.append(sil)
    #     val = np.mean(SelBest(np.array(tmp_sil), int(iterations / 5)))
    #     err = np.std(tmp_sil)
    #     sils.append(val)
    #     sils_err.append(err)

    import numpy as np
    from sklearn.mixture import GaussianMixture as GMM
    from sklearn.metrics import silhouette_score

    X, y = get_ds()

    instances = X.shape[0]

    if not limit:
        limit = instances

    step = int((limit - start) / (npoints - 1))

    end = 1 + start + (npoints - 1) * step
    cl_list = list(range(start, end, step))
    print(f"Evaluting cluster sizes: {cl_list}")
    results = np.zeros((len(cl_list), 2))

    for i, k in enumerate(cl_list):
        print(f'{k} centers')

        gmm_obj = GMM(k)
        gmm_obj.fit(X)
        label = gmm_obj.predict(X)
        results[i] = [silhouette_score(X, label), k]

    return results


def run_k_means2(k=2, get_ds=None, start=2, npoints=10, limit=None, skipLast=False):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X, y = get_ds()

    instances = X.shape[0]

    if not limit:
        limit = instances

    step = int((limit - start)/(npoints - 1))

    end = 1 + start + (npoints - 1) * step
    cl_list = list(range(start, end, step))
    print(f"Evaluting cluster sizes: {cl_list}")
    results = np.zeros((len(cl_list), 2))

    for i, k in enumerate(cl_list):
        print(f'{k} clusters')
        if skipLast and i == len(cl_list) - 1:
            pass
        else:
            KMean = KMeans(n_clusters=k)
            KMean.fit(X)
            label = KMean.predict(X)
            results[i] = [silhouette_score(X, label), k]

    return results



KMEANS = 'k-means'
GMMK = 'gmm'
PCA = 'pca'

def main(args):

    if KMEANS in args:

        Logs.set_log_file('kmeans_init_clustering')
        if DS_1 in args:
            ranges = [(2, 42)]
            for t in ranges:
                st = t[0]
                lim = t[1]
                results = run_k_means2(start=st, limit=lim, get_ds=get_digits, npoints=40)

                max_idx = np.argmax(results[:, 0])
                xdata = {'kmeans':f"max y:{pr_fl(results[max_idx, 0])} x:{int(results[max_idx, 1])}"}
                make_fitness_plots(
                    title='Digits Kmeans Silhouette Scores', fn=get_plot_fn(f'k_means_1_digits_{st}_{lim}'), data={'kmeans':results},
                    x_label='k', y_label='Silh Scor', extra_data=xdata
                )

        if DS_2 in args:
            # ranges = [(1400, 1500), (1501, 1600), (1601, 1700), (1701, 1800)]
            ranges = [(1500, 1519, 20), (1520, 1539, 20), (1540, 1559, 20), (1560, 1579, 20), (1580, 1599, 20)]

            for t in ranges:
                st = t[0]
                lim = t[1]
                if len(t) > 2:
                    pts = t[2]
                else:
                    pts = 10

                results = run_k_means2(start=st, limit=lim, get_ds=get_bach, npoints=pts)

                max_idx = np.argmax(results[:, 0])
                xdata = {'kmeans':f"max y:{pr_fl(results[max_idx, 0])} x:{int(results[max_idx, 1])}"}
                make_fitness_plots(
                    title='Bach Harmony Kmeans Silhouette Scores', fn=get_plot_fn(f'k_means_1_bach_{st}_{lim}'), data={'kmeans': results},
                    x_label='k', y_label='Silh Scor', extra_data=xdata, zero_x_bnd=False, zero_y_bnd=False
                )

    if GMMK in args:
        Logs.set_log_file('gmm_init_clustering')
        if DS_1 in args:
            ranges = [(2, 50, 49)]
            for t in ranges:
                st = t[0]
                lim = t[1]

                if len(t) > 2:
                    pts = t[2]
                else:
                    pts = 10

                results = run_gmm(start=st, limit=lim, get_ds=get_digits, npoints=pts)

                max_idx = np.argmax(results[:, 0])
                xdata = {'gmm': f"max y:{pr_fl(results[max_idx, 0])} x:{int(results[max_idx, 1])}"}
                make_fitness_plots(
                    title='Digits GMM Silhouette Scores', fn=get_plot_fn(f'gmm_1_digits_{st}_{lim}'),
                    data={'gmm': results},
                    x_label='k', y_label='Silh Score', extra_data=xdata
                )

        if DS_2 in args:
            print("Bach clustering")
            # ranges = [(1400, 1500), (1501, 1600), (1601, 1700), (1701, 1800)]
            # ranges = [(1500, 1519, 20), (1520, 1539, 20), (1540, 1559, 20), (1560, 1579, 20), (1580, 1599, 20)]
            #ranges = [(2, 1000, 20), (1001, 2000, 20), (2001, 3000, 20), (3001, 4000, 20), (4001, 5000, 20), (5001, 6000, 20)]
            ranges = [#(2, 1000, 5), (1001, 2000, 5), (2, 500, 50),
                #(501, 1000, 20), (1001, 1500, 20), (1501, 2000, 20)
                 #(2000, 2500, 20), (1900, 2000, 101), (1800, 1900, 101)
                (2000, 2024, 25 ), (2025, 2049, 25 ), (2050, 2074, 25 ), (2075, 2099, 25 )
            ]
            for t in ranges:
                st = t[0]
                lim = t[1]
                if len(t) > 2:
                    pts = t[2]
                else:
                    pts = 10

                results = run_gmm(start=st, limit=lim, get_ds=get_bach, npoints=pts)

                max_idx = np.argmax(results[:, 0])
                xdata = {'gmm': f"max y:{pr_fl(results[max_idx, 0])} x:{int(results[max_idx, 1])}"}
                make_fitness_plots(
                    title='Bach Harmony Kmeans Silhouette Scores', fn=get_plot_fn(f'gmm_1_bach_{st}_{lim}'),
                    data={'gmm': results},
                    x_label='k', y_label='Silh Score', extra_data=xdata, zero_x_bnd=False, zero_y_bnd=False
                )


max_silh_km_bach = 1575
max_silh_km_digits = 15
max_silh_gmm_bach = 2056
max_silh_gmm_digits = 17

if __name__ == '__main__':
    import sys
    import datetime
    main(sys.argv[1:])

    time_now = datetime.datetime.now().strftime("%m_%d_%H_%M")
    print(f"Finished execution at {time_now}")
    log("Finished execution")





# See PyCharm help at https://www.jetbrains.com/help/pycharm/

