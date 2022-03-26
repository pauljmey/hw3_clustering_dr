
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from functools import partial


class Logs():
    cur_log_file = None
    log_time_tag = None
    log_dir = './logs'

    def set_log_file(tag, src_level= -2):
        Logs.cur_log_file = Logs.get_log_fn(tag=tag, src_level=src_level)

    def get_log_fn(tag=None, src_level=-2):
        if Logs.cur_log_file is None and tag is None:
            tag = "LOG_default"

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

def cur_step(st=None, store=[]):
    if st is not None:
        if not store:
            store.append(st)
        else:
            store[0] = st

    return store[0]


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
    if not Logs.log_time_tag:
        Logs.log_time_tag = datetime.datetime.now().strftime("%m_%d_%H_%M")

    when = Logs.log_time_tag + ": "
    when += where + "-"

    if log_fn is None:
        log_fn = Logs.get_log_fn()

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

def get_plot_fn(tag=None, root_path='./plots', ftype='.png'):
    import datetime, os
    time_tag = datetime.datetime.now().strftime("%m_%d_%H_%M")
    day = get_day_prefix(time_tag)
    target = os.sep.join([root_path, day])
    if not os.path.exists(target):
        os.makedirs(target)

    fn = tag + '_' + time_tag + ftype
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

def make_plot(
        title=None,  data=None, fn=None, rows=1, extra_data=None,
        x_label="iterations", y_label="fitness score", trunc_converged=True,
        zero_x_bnd=False, zero_y_bnd=False, ebars=False
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
        if not ebars:
            subplot_1.plot(cur_x, cur_y, color=colors[i%clen], marker=pt, linestyle='solid', label=lbl)
        else:
            yerr = np.std(cur_y)
            subplot_1.errorbar(cur_x, cur_y, yerr=yerr, color=colors[i % clen], marker=pt, linestyle='solid', label=lbl)

    subplot_1.legend(loc="lower right")

    fig.tight_layout()
    plt.margins(10, 0)

    plt.savefig(fn)
    plt.show()




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


max_silh_km_bach = 1575
max_silh_km_digits = 15
max_silh_gmm_bach = 2056
max_silh_gmm_digits = 17

KMEANSK = 'kmeans'
GMMK = 'gmm'

STEP1 = 's1'
STEP2 = 's2'
STEP3 = 's3'
STEP4 = 's4'
STEP5 = 's5'

max_cluster_scores = {

    STEP1:{
        DS_1: {
            KMEANSK: {'max': max_silh_km_digits} ,
            GMMK: {'max': max_silh_gmm_digits}
        },
        DS_2: {
            KMEANSK: {'max': max_silh_km_bach},
            GMMK: {'max': max_silh_gmm_bach}
        }
    },
    STEP2:{
        DS_1: {
            KMEANSK: {'max': -1},
            GMMK: {'max': -1}
        },
        DS_2: {
            KMEANSK: {'max': -1},
            GMMK: {'max': -1}
        }
    }

}

def TEST_MODE(val=None, store=[]):
    if not store:
        if not val:
            store.append(False)
        else:
            store.append(val)

    if not val:
        pass
    else:
        store[0] = val

    return store[0]

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
    if TEST_MODE():
        #last = int(.1 * len(X_train_scaled))
        last = 81
        X_train_scaled = X_train_scaled[0:last]
        y_train_hot = y_train_hot[0:last]

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

    if TEST_MODE():
        #last = int(.1 * len(X_train_scaled))
        last = 81
        X_train_scaled = X_train_scaled[0:last]
        y_train_hot = y_train_hot[0:last]

    return X_train_scaled, y_train_hot

GETK= 'get_func'
TAGK = 'tag'
GET_DIGITS ={
    GETK:get_digits
}

GET_BACH ={
    GETK:get_bach
}

datasets_info = {
    DS_1: GET_DIGITS,
    DS_2: GET_BACH
}

def make_x_points(starts=None, points=None, limits=None):
    assert (len(starts) == len(points))
    assert (len(starts) == len(limits))

    res_list = []
    regions = len(starts)
    for i, s in enumerate(starts):
        cur_start = s
        cur_points = points[i]

        #validate cur_points, make sure cur_points not too large
        if i == regions - 1:  # last
            if limits[i] is not None:
                cur_points = min(cur_points, limits[i] - cur_start + 1)
        else:
            next_start = starts[i + 1]
            temp_limit = next_start - 1
            cur_points = min(cur_points, temp_limit - cur_start + 1)

        if limits[i] is None:
            if i == regions - 1:  # last
                cur_limit = cur_start + cur_points - 1
            else:
                cur_limit = starts[i + 1] - 1
        else:
            cur_limit = limits[i]

        step = int((cur_limit - cur_start) / (cur_points - 1))
        assert (step > 0)
        cur_list = list(range(cur_start, cur_limit + 1, step))
        res_list += cur_list

    return res_list

def validate_limit_list(start, limit, instances):
    if limit is None:
        limit = [None for x in start]
        limit[-1] = instances
    elif len(limit) == 1:
        last_val = limit[0]
        last_pos = len(start) - 1
        limit = [None if i != last_pos else last_val for i in range(len(start)) ]
    elif len(start) != len(limit):
        assert(False)

    if limit[-1] is None:
        limit[-1] = instances

    return limit

def gmm_score(X, k):
    from sklearn.metrics import silhouette_score

    gmm_obj = GMM(k)
    gmm_obj.fit(X)
    label = gmm_obj.predict(X)

    return silhouette_score(X, label)


def run_gmm(x_pts=None, get_ds=None):
    import numpy as np
    from sklearn.mixture import GaussianMixture as GMM

    X, y = get_ds[GETK]()
    tag = get_ds[TAGK]

    instances = X.shape[0]

    if x_pts is None:
        cl_list = list(range(2, X.shape[0])) #all instance
    else:
        cl_list = x_pts

    print(f"Evaluting cluster sizes ({tag}): {cl_list}")

    results = np.zeros((len(cl_list), 2))

    for i, k in enumerate(cl_list):
        print(f'{k} centers')
        sc = gmm_score(X, k)
        results[i] = [sc, k]

    return results


def run_k_means2(x_pts=None, get_ds=None, skipLast=False):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X, y = get_ds[GETK]()
    tag = get_ds[TAGK]

    instances = X.shape[0]
    #limit = validate_limit_list(start, limit, instances)
    if x_pts is None:
        cl_list = list(range(2, X.shape[0])) #all instance
    else:
        cl_list = x_pts

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

def dr_ica(X,y, ds_tag=None, do_whitening=True):
    from sklearn import decomposition
    from scipy.stats import norm
    from scipy.stats import kurtosis

    space_dim = X.shape[1]
    kur_results = np.zeros((space_dim, 2))
    for i in range(space_dim):
        num_components = i + 1
        #if num_components == space_dim:
        #    continue  # the max number of components tends to generate errors, evaluating not useful
        if num_components == 64:
            hook = True
        print(f"Evaluating {num_components} ICA components")
        ica = decomposition.FastICA(n_components=num_components, whiten=do_whitening,
            max_iter=800)

        try:
            ica.fit(X)
        except Exception as ex:
            hook = True
            raise

        A = ica.components_
        tally = np.zeros(A.shape[0])

        for ii, v in enumerate(ica.components_):
            one_axis = np.zeros(A.shape)
            one_axis[ii] = A[ii]
            proj = np.dot(X, one_axis.T)
            proj = proj[:, ii]
            kur_score = kurtosis(proj)
            tally[ii] = np.abs(kur_score)

        kur_results[i] = [np.min(tally), num_components]

    #kur_results = np.abs(kur_results)
    max_idx = np.argmax(kur_results[:,0])
    plot_tag = 'kur'
    data = {plot_tag: kur_results}
    xdata = {plot_tag: f"max y:{pr_fl(kur_results[max_idx, 0])} x:{int(kur_results[max_idx, 1])}"}
    max_y = kur_results[max_idx, 0]
    max_red = kur_results[max_idx, 1]/space_dim
    max_vals = np.repeat(max_y, len(kur_results))
    std = np.std(max_vals - kur_results[:, 0])
    thresh = (max_y - std)/ max_y

    #norm_fact = max_y/max_red
    def score2(x, y,  thresh, max_store=[]):

        cur_sc_ratio = y/max_y
        cur_red_ratio = (x/space_dim)/max_red
        score2 = cur_sc_ratio/cur_red_ratio * max_y

        if not max_store:
            max_store.append(score2)

        if cur_sc_ratio > thresh and x > .4 * space_dim: # ignore larger reductions
            if score2 > max_store[0]:
                max_store[0] = score2
        else:
            score2 = .9 * max_store[0]

        return score2 # scale for graph

    #effectiveness_score
    test = score2(kur_results[max_idx, 1], kur_results[max_idx, 0], thresh)
    eff_score = [score2(r[1], r[0], thresh) for r in kur_results[0:max_idx + 1]]
    eff_score = np.array(eff_score)
    eff_score = np.column_stack((eff_score, kur_results[:,1][0:max_idx + 1]))
    data['eff_score'] = eff_score
    max_idx = np.argmax(eff_score[:, 0])
    highest_eff_score = max_idx
    xdata['eff_score'] = f"max y:{pr_fl(eff_score[max_idx, 0])} x:{int(eff_score[max_idx, 1])}"

    make_plot(
        title=f"Kurtosis Score for ICA components ({ds_tag})", data=data, x_label="ICA component",
        y_label="Kur Score", extra_data=xdata, fn=get_plot_fn(f'ICA_exp_plot_{ds_tag}'),
    )

    num_components = int(kur_results[highest_eff_score, 1])
    ica = decomposition.FastICA(n_components=num_components, whiten=do_whitening)

    return ica, ica.fit_transform(X)

ICAK = 'ica'
DR_ICA={
    GETK:dr_ica
}

def dr_pca(X,y, ds_tag=None):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=.95)
    pca.fit(X)

    return pca, pca.transform(X)

DR_PCA={
    GETK:dr_pca
}


def ica_eval_plot(estimator=None, fn=None, ds_tag=None):
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis

    ica = estimator
    # for i, v in enumerate(ica.components_):


def pca_eval_plot(estimator=None, fn=None, ds_tag=None):
    import matplotlib.pyplot as plt
    pca = estimator
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()

    y = np.cumsum(pca.explained_variance_ratio_)
    xi = np.arange(1, y.shape[0] + 1, step = 1)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 11, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    cutoff = [(ii, v) for ii, v in enumerate(y) if v >= .95]
    last = cutoff[0][0]
    plt.title(f'The number of components needed to explain variance, cutoff component = {last}')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.savefig(fn)
    plt.show()

PCAK = 'pca'

EVALPLOTK = 'evp'
DRFUNCK = 'drfunc'
REDK = 'redmeth'

DR_TAGS = [PCAK, ICAK]
dim_reduce_info = {
    PCAK: {DRFUNCK:dr_pca, EVALPLOTK:pca_eval_plot},
    ICAK: {DRFUNCK:dr_ica, EVALPLOTK:ica_eval_plot}
}

def run_reduction_2(reduce_func=None, plot_func=None, get_ds=None, fn=None):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X, y = get_ds[GETK]()
    tag = get_ds[TAGK]

    instances = X.shape[0]
    drfunc = reduce_func[DRFUNCK]
    estimator, reduced_x = drfunc(X, y, ds_tag=tag)
    #eval plots
    plot_func(estimator=estimator, fn=fn, ds_tag=tag)

    return reduced_x

STEP_RES = "_STEP_RESULTS"

def format_ranges(ranges):
    starts = []
    limits = []
    points = []
    for t in ranges:
        st = t[0]
        lim = t[1]
        starts.append(st)
        limits.append(lim)
        if len(t) > 2:
            pts = t[2]
        else:
            pts = 10

        points.append(pts)

    return starts, points, limits


def set_series(
    data, xdata, ptag=None,
    st=None, ds_tag=None, cl_tag=None
):
    cur_s = get_np_array(max_cluster_scores, ds_tag, cl_tag, st=st)
    max_idx = np.argmax(cur_s[:, 0])
    xdata[ptag] = f"max y:{pr_fl(cur_s[max_idx, 0])} x:{int(cur_s[max_idx, 1])}"
    data[ptag] = cur_s


def run_km_clustering(title=None, ranges=None, get_ds=None, cl_tag=KMEANSK):

    #starts, points, limits = format_ranges(ranges)
    results = run_k_means2(x_pts=ranges, get_ds=get_ds)
    ds_tag = get_ds[TAGK]

    set_np_array(max_cluster_scores, ds_tag, cl_tag, results)

    data = {}
    xdata = {}

    st = cur_step()
    if st == STEP2:
        ptag = st + ' (reduced)'
        set_series(data, xdata, ptag=ptag, st=STEP2, ds_tag=ds_tag, cl_tag=cl_tag)
        set_series(data, xdata, ptag=STEP1, st=STEP1, ds_tag=ds_tag, cl_tag=cl_tag)
    elif st == STEP1:
        ptag = st
        set_series(data, xdata, ptag=ptag, st=STEP1, ds_tag=ds_tag, cl_tag=cl_tag)

    if ranges is None:
        range_text = "fullr"
    else:
        start = ranges[0][0]
        lim = ranges[0][-1]
        range_text = f'{start}-{lim}'


    ds_txt = ds_tag
    if REDK in get_ds:
        ds_txt = f'{ds_txt}_{get_ds[REDK]}'

    make_plot(
        title=title, fn=get_plot_fn(f'{cl_tag}_{ds_txt}_{st}_{range_text}'),
        data=data,
        x_label='k', y_label='Silh Scor', extra_data=xdata
    )

    return results


def run_gmm_clustering(title=None, get_ds=None, ranges=None, cl_tag=GMMK):

    #starts, points, limits = format_ranges(ranges)
    results = run_gmm(x_pts=ranges, get_ds=get_ds)

    st = cur_step()
    ds_tag = get_ds[TAGK]
    set_np_array(max_cluster_scores, ds_tag, cl_tag, results)
    title = f'{ds_tag} {cl_tag} Silhouette Scores'

    data = {}
    xdata = {}


    if st == STEP2:
        ptag = st + ' (reduced)'
        set_series(data, xdata, ptag=ptag, st=STEP2, ds_tag=ds_tag, cl_tag=cl_tag)
        set_series(data, xdata, ptag=STEP1, st=STEP1, ds_tag=ds_tag, cl_tag=cl_tag)
    elif st == STEP1:
        ptag = st
        set_series(data, xdata, ptag=ptag, st=STEP1, ds_tag=ds_tag, cl_tag=cl_tag)

    start = starts[0]
    lim = limits[-1]

    make_plot(
        title=title, fn=get_plot_fn(f'{cl_tag}_{ds_tag}_{st}_{start}_{lim}'),
        data=data,
        x_label='k', y_label='Silh Scor', extra_data=xdata
    )
    return results

KM_CLUSTERING = {
    GETK: run_km_clustering,
}
GMM_CLUSTERING = {
    GETK: run_gmm_clustering
}

CLUSTERING_TAGS = [KMEANSK, GMMK]

clustering_info = {
    KMEANSK: KM_CLUSTERING,
    GMMK: GMM_CLUSTERING
}


def get_step1_range(ds_tag=None, cluster_tag=None):

    max_idx = max_cluster_scores[STEP1][ds_tag][cluster_tag][STEP_RES]['max_idx']
    step1_series = get_np_array(max_cluster_scores, ds_tag, cluster_tag, st=STEP1)

    start = 2
    last_pt = int(1.1 * max_x)
    pts = last_pt - start + 1
    if last_pt - start < 20:
        ret_rng = list(range(2, last_pt + 1))
    else:
        r1 = make_x_points(starts=[start], limits= [last_pt], points=[15])
        r1 = set(r1)
        r1 = r1.union({max_x})
        ret_rng = list(r1)
        ret_rng.sort()
        assert(max_x in ret_rng)

    return [ret_rng], max_x

def run_reduction(
    get_ds=None, cluster_info=None, dr_func=None,
    plot_func=None
):
    def get_data_stub(d=None):
        return d, None

    ds_tag = get_ds[TAGK]
    dr_tag = dr_func[TAGK]
    ds_x_reduced = run_reduction_2(
        reduce_func=dr_func, plot_func=plot_func, get_ds=get_ds,
        fn=get_plot_fn(f'Step_2_PCA_{ds_tag}')
    )

    get_reduced = partial(get_data_stub, d=ds_x_reduced)
    GetReduced = {
        TAGK: get_ds[TAGK], GETK: get_reduced, REDK: dr_func[TAGK]
    }
    cl_tag = cluster_info[TAGK]
    cluster_func = cluster_info[GETK]
    ranges, s1_max = get_step1_range(ds_tag=ds_tag, cluster_tag=cl_tag)
    title = f'{ds_tag} {cl_tag} Silhouette Scores\nred. by {dr_tag} dim={ds_x_reduced.shape[1]}, s1 xmax = {s1_max}'
    cluster_func(title=title, ranges=ranges, get_ds=GetReduced)

def set_np_array(info, k1, k2, a, st=None): # ds = k1, k2 = CL METHOD

    if st is None:
        st = cur_step()

    if k1 not in info[st]:
        info[st][k1] = {}

    if k2 not in info[st][k1]:
        info[st][k1][k2] = {}

    if STEP_RES not in info[st][k1][k2]:
        info[st][k1][k2][STEP_RES] = {}

    cur_d = info[st][k1][k2][STEP_RES]
    if 'a' not in cur_d:
        old_a = None
    else:
        old_a = np.array(cur_d['a'])

    ba = old_a == a
    import os
    if not ba.all() or 'serialized' not in cur_d or not os.path.exists(cur_d['serialized']):

        if 'serialized' not in cur_d or not os.path.exists(cur_d['serialized']):
            a_fn = get_plot_fn(tag=f'saved_npa_{st}_{k1}_{k2}', ftype='.npy')
        else:
            a_fn = cur_d['serialized']

        max_idx = np.argmax(a[:,0])
        cur_d['max_idx']= int(max_idx)
        cur_d['a'] = a.tolist()
        np.save(a_fn, a)
        cur_d['serialized'] = a_fn

    save_cluster_scores(info)
    return a

def get_np_array(info, k1, k2, st=None):
    if st is None:
        st = cur_step()

    if k1 not in info[st] or k2 not in info[st][k1]:
        hook = True

    assert (k1 in info[st] and k2 in info[st][k1])
    assert(STEP_RES in info[st][k1][k2])
    cur_d = info[st][k1][k2][STEP_RES]
    a = None
    if 'serialized' in cur_d:
        a_fn = cur_d["serialized"]
        a = np.load(a_fn)
        #assert ((np.array(cur_d['a']) == a).all())
        cur_d['a'] = a.tolist()

    return a

def save_cluster_scores(scores, fn = './max_cluster_scores.json'):
    import json

    with open(fn, 'w') as jfp:
        json.dump(scores, jfp)

def read_cluster_scores(scores, fn='./max_cluster_scores.json'):
    import json
    with open(fn, 'r') as jfp:
        scores_pickled = json.load(jfp)

    src = scores_pickled
    targ = scores
    for k0 in [STEP1, STEP2]:
        keys = list(src[k0])
        for k1 in keys:
            for k2 in src[k0][k1]:
                cur_d = src[k0][k1][k2]
                if STEP_RES in cur_d:
                    a_fn = cur_d[STEP_RES]['serialized']
                    read_a = get_np_array(src, k1, k2, st=k0)
                    cur_d[STEP_RES]['safe_a'] = read_a.tolist()

                cur_targ = targ[k0][k1][k2]
                for k3 in cur_d:
                    new = cur_d[k3]
                    if k3 in cur_targ:
                        old = targ[k0][k1][k2][k3]
                    else:
                        old = None
                        cur_targ[k3] = None

                    if old != new:
                        print(f'updating {k0}-{k1}-{k2}-{k3}: {old} => {new}')
                        cur_targ[k3] = new
                    else:
                        print(f'{k0}-{k1}-{k2}-{k3}: original == new')

def main(args):

    if 'test' in args:
        TEST_MODE(val=True)

    if 'no_reload' not in args:
        read_cluster_scores(max_cluster_scores)

    if 'INIT' in args:
        get_ds = datasets_info[DS_2]
        get_ds[TAGK] = GMMK

        ds = get_ds[GETK]
        X, y = ds()
        lazy_search(X, y_func=gmm_score)

    if STEP1 in args:
        cur_step(st=STEP1)
        for cl_tag in CLUSTERING_TAGS:
            if cl_tag in args:
                Logs.set_log_file(f"{cl_tag}_Step_1")
                cluster_tag = KMEANSK
                for arg in [DS_1, DS_2]:
                    if arg in args:
                        print(f"KMEANS {arg} clustering")

                        get_ds = datasets_info[arg]
                        get_ds[TAGK] = arg

                        cl_info = clustering_info[cluster_tag]
                        cl_info[TAGK] = cluster_tag

                        results = cl_info[GETK](ranges=None, get_ds=get_ds)
                        set_np_array(max_cluster_scores, arg, cluster_tag, results, st=STEP1)

            save_cluster_scores(max_cluster_scores)

    if STEP2 in args:
        cur_step(st=STEP2)
        scores_recorded = True
        for k1 in max_cluster_scores[STEP1]:
            k2_keys = list(max_cluster_scores[STEP1][k1])
            filtered = [x for x in k2_keys if 'serialized' in x]
            if not filtered:
                scores_recorded = False
                break
        if not scores_recorded:
            read_cluster_scores(max_cluster_scores)


        for dr_tag in DR_TAGS:
            Logs.set_log_file(f"Step_2_{dr_tag}")
            if dr_tag in args:
                dr_func = dim_reduce_info[dr_tag]
                dr_func[TAGK] = dr_tag

                plot_func = dr_func[EVALPLOTK]

                for cluster_tag in [KMEANSK, GMMK]:
                    if cluster_tag in args:
                        cluster_info = clustering_info[cluster_tag]
                        cluster_info[TAGK] = cluster_tag

                        for ds_tag in [DS_1, DS_2]:
                            if ds_tag in args:
                                ds_info = datasets_info[ds_tag]
                                ds_info[TAGK] = ds_tag
                                print(f'STEP2 {dr_tag} {cluster_tag} {ds_tag}')
                                run_reduction(
                                    get_ds=ds_info, cluster_info=cluster_info, dr_func=dr_func,
                                    plot_func=plot_func
                                )


def lazy_search(s, y_func=None, stop=10):
    def valid(x):
        return x >= 0 and x <= len(s) - 1

    def hillclimb(x, y_func, s, step=1, forward=None):
        if not s_visited[x]:
            s_map[x, 1] = s_pts[x]
            print(f'calculating y at {x}')
            s_map[x, 0] = y_func(s, s_pts[x])
            cur_val = s_map[x, 0]
            s_visited[x] = True
        else:
            cur_val = s_map[x, 0]

        if forward is None:
            n_stops = [x, x]
            step = 1
            neighbors = [x - step, x + step]
            for i, v in enumerate(neighbors):
                n_stops[i] = None
                if valid(v):
                    forward = True if i == 1 else False
                    ret = hillclimb(v, y_func, s, step=step * 2, forward=forward)
                    assert(type(ret) is not list)
                    n_stops[i] = ret
            return n_stops[0], n_stops[1]
        else:
            prev_step = step/2
            prev_pos = int(x + prev_step) if not forward else int(x - prev_step)
            prev_val = s_vals[prev_pos]
            assert(s_visited[prev_pos])

            neighbor = x + step if forward else x - step

            if valid(neighbor):
                stop = x
                if max_mode:
                    if cur_val >= prev_val :
                        stop = hillclimb(neighbor, y_func, s, step=step * 2, forward=forward)
                else:
                    if cur_val < prev_val:
                        stop = hillclimb(neighbor, y_func, s, step=step * 2, forward=forward)

                return stop
            else:
                return x

    def adjust(b, f, store, pos):
        tr = store[pos]
        if bs != fs:
            if not bs:
                pass
            else:
                tr.append(bs)
            if not fs:
                pass
            else:
                tr.append(fs)

            tr = list(set(tr))
            tr.sort()
            store[pos] = tr

    trails = [[], []]

    s_map = np.zeros((s.shape[0], 2))
    s_vals = s_map[:, 0] # reversed x, y order
    for i in range(len(s)):
        s_map[i, 1] = i + 2
    s_pts = [int(x) for x in s_map[:, 1]]
    s_visited = np.repeat(False, len(s))

    starts = {0: 1, 1: len(s.shape)//2, 2: len(s) - 2}
    max_mode = True
    step = 1
    for cycle in range(stop+1):
        print(f'cycle {cycle}')
        for i, a in enumerate(trails):
            if not a:
                a.append(starts[i])

            cur_start = a[0]
            cur_end = a[-1]
            int_len = len(a)
            print(f'starting search at {cur_start}')
            bs, fs = hillclimb(cur_start, y_func, s)
            adjust(bs, fs, trails, i)

            if  int_len > 1:
                print(f'starting search at {cur_end}')
                bs, fs = hillclimb(cur_end, y_func, s)
                adjust(bs, fs, trails, i)

                print(f'endpoints = {bs, fs}')

        max_mode = not max_mode
        make_plot(data={'lazy':s_map}, title=f"Lazy search, cycle {cycle}",
                  fn=get_plot_fn(f"Lz_srch_cy_{cycle}")
        )


if __name__ == '__main__':
    import sys
    import datetime

    main(sys.argv[1:])

    time_now = datetime.datetime.now().strftime("%m/%d %H:%M")
    print(f"Finished execution at {time_now}")
    log("Finished execution")





# See PyCharm help at https://www.jetbrains.com/help/pycharm/

