# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import mlrose_hiive as mlr
import numpy
import numpy as np
import matplotlib.pyplot as plt


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


def get_ranges(data):
    hook = True

    for k in data:
        if len(data[k].shape) == 1:
            col2 = [[x+1] for x in range(data[k].shape[0])]
            data[k] = np.column_stack((data[k], col2))
            hook = True

    massage = lambda x, l: x + 2 if x + 2 < len(l) else x
    y_idx_at_max = [get_plot_xmax(x=data[k][:, 1], y= data[k][:, 0]) for k in data]

    all_max_x = max(y_idx_at_max)
    all_min_x = 0

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
        zero_x_bnd=True, zero_y_bnd=True
    ):

    plt.close()
    fig, axes = plt.subplots(rows, 1, squeeze=False, figsize=(10, 10))

    if hasattr(axes, 'shape'):
        subplot_1 = axes[0, 0]

    all_min_x, all_max_x, all_min_y, all_max_y = get_ranges(data)
    if zero_x_bnd:
        xp_min = min(0, .9 * all_min_x)
    else:
        xp_min = .9 * all_min_x

    if zero_y_bnd:
        yp_min = min(0, .9 * all_min_y)
    else:
        yp_min = .9 * all_min_y
    subplot_1.set_xlim(xp_min, 1.1 * all_max_x)
    subplot_1.set_ylim(yp_min, 1.1 * all_max_y)
    subplot_1.set_title(title)
    subplot_1.set_xlabel(x_label)
    subplot_1.set_ylabel(y_label)

    subplot_1.grid()
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    syms = ['+', '1', '2', '3', '4']

    for i, k in enumerate(data):
        log(f"raw plot {k}", {'plot_data': data[k]})

        cur_x = data[k][:, 1]
        cur_y = data[k][:, 0]
        if trunc_converged:
            y_idx_at_conv, _ = get_converge(cur_y)
            y_idx_at_conv = min(len(cur_y) - 1, y_idx_at_conv + 2)
            cur_x_data_2 = cur_x[0: y_idx_at_conv + 1]
            cur_y_data_2 = cur_y[0: y_idx_at_conv + 1]

            cur_x = cur_x_data_2
            cur_y = cur_y_data_2

        log(f"processed plot {k} x:", cur_x)
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

TK = 'tag'
AK = 'args'
FK = 'func'
PRK = 'pr-type'

HCK = 'hc'
GAK = 'ga'
RHCK = 'rhc'
SIMAK = 'sim-an'
MIMICK = 'mimic'

opt_algos = {
    HCK : {FK: mlr.hill_climb, AK: get_hc_args},
    GAK : { FK: mlr.genetic_alg, AK: get_ga_args},
    RHCK : { FK: mlr.random_hill_climb, AK: get_rhc_args},
    SIMAK : { FK: mlr.simulated_annealing, AK: get_sa_args},
    MIMICK : { FK: mlr.mimic, AK: get_mmc_args}
}


opt_algos_lookup = {
    mlr.hill_climb: HCK,
    mlr.genetic_alg: GAK,
    mlr.random_hill_climb: RHCK,
    mlr.simulated_annealing: SIMAK,
    mlr.mimic: MIMICK
}

ALG_ORDER = [GAK, SIMAK, MIMICK]
ALL_ALG_TAGS = [RHCK, SIMAK, GAK, MIMICK]
GDK='grad-desc'

net_algos = {RHCK:'random_hill_climb', SIMAK:'simulated_annealing', GAK:'genetic_alg', GDK:'gradient_descent'}

QUEENS = 'Queens'
CONT_PEAKES = 'Cont-Peaks'
FLFLOP = 'Flip-Flop'
F4PEAKS = '4-Peaks'
KNSACK = 'Knap-Sack'
MXK = 'Max-K-Color'
O1MAX = '1-Max'
TSP = 'Traveling-Salesperson'
S6PEAKS = 'Six-Peaks'

COMPLEX_PLOTS = {GAK:KNSACK, SIMAK:S6PEAKS, MIMICK:CONT_PEAKES}

all_problems = [
    QUEENS, MXK, O1MAX, FLFLOP, CONT_PEAKES, F4PEAKS, S6PEAKS, KNSACK
]

def get_fitness(tag, dim=None):
    if tag == QUEENS:
        return mlr.Queens()
    elif tag == CONT_PEAKES:
        return mlr.ContinuousPeaks()
    elif tag == FLFLOP:
        return mlr.FlipFlop()
    elif tag == F4PEAKS:
        return mlr.FourPeaks()
    elif tag == KNSACK:
        w, v = get_knapsack_wvs(length=dim)
        return mlr.Knapsack(w, v)
    elif tag == MXK:
        e = get_mxk_edges(length=dim)
        return mlr.MaxKColor(e)
    elif tag == O1MAX:
        return mlr.OneMax()
    elif tag == S6PEAKS:
        return mlr.SixPeaks()
    elif tag == TSP:
        c, d = get_tsp_cd()
        return mlr.TravellingSales(coords=c)
    else:
        assert(False)

def get_knapsack_wvs(length=None, store=[]):

    if not store:
        no_dupes = False
        wlist = np.array([x+1 for x in range(500)])
        vlist = np.array([x + 1 for x in range(200)])
        while not no_dupes:
            weights = np.random.choice(wlist, size=100)
            values = np.random.choice(vlist, size=100)

            dupe_check = [(weights[i], values[i]) for i, x in enumerate(weights)]
            dupe_check_len = len(set(dupe_check))
            if dupe_check_len == 100:
                no_dupes = True

        store.append(weights)
        store.append(values)

    w = store[0]
    v = store[1]
    return w[0:length], v[0:length]

def get_mxk_edges(length=8, store=[]):
    def get_degrees(nds, eds):
        degrees = {i:[] for i, x in enumerate(nds)}
        nodes_1 = [t[0] for t in eds]
        nodes_2 = [t[1] for t in eds]

        for x in nds:
            s1 = [1 if z == x else 0 for z in nodes_1]
            s2 = [1 if z == x else 0 for z in nodes_2]
            tot = sum(s1) + sum(s2)
            degrees[tot].append(x)

        return degrees

    def get_lowest_degree(d, eds):
        dkeys = list(d)
        dkeys.sort()
        ret_degree = None
        ret_node = None
        degs = []
        nds = []
        for k in dkeys:
            for idx, nd in enumerate(d[k]):
                if not nds:
                    nds.append(nd)
                    degs.append(k)
                else:
                    idx2 = idx
                    n1 = nds[0]
                    ed = (min(n1, nd), max(n1, nd))
                    if ed in eds:
                        continue
                    else:
                        eds.append(ed)
                        k1 = degs[0]
                        nds = []
                        degs = []
                        ret = d[k1].pop(0)
                        assert(ret == n1)
                        if k1 + 1 not in d:
                            d[k1 + 1] = []
                            dkeys.append(k1 + 1)

                        d[k1 + 1].append(ret)
                        k2 = k
                        if k2 + 1 not in d:
                            d[k2 + 1] = []
                            dkeys.append(k2 + 1)

                        if k1 == k2:
                            idx2 -= 1 # decrement due to popping IF in the same d item (ie same list)
                            dkeys.append(k1)

                        ret = d[k2].pop(idx2)
                        assert (ret == nd)
                        d[k2 + 1].append(ret)

                        print(d)
                        return True
        return False

    if not store:
        store.append([])
        store.append([])

    if len(store[0]) != length:

        store[0] = [x for x in range(length)]
        nodes = store[0]
        store[1] = []
        edges = store[1]
        sub1 = [x for x in range(int(.5 + length/2))]
        sub2 = [x + int(length/2) for x in range(int(.5 + length/2))]
        for i, x in enumerate(sub1):
            if i != 0:
                edges.append((0, x))
                n1 = sub2[0]
                ed = (n1, n1 + i)
                if ed not in edges:
                    edges.append(ed)

        edges.append((sub1[1], sub2[0])) # connect 2 subtrees

        degrees = get_degrees(nodes, edges)
        dx = None
        dy = None

        print(degrees)
        while True:
            if not get_lowest_degree(degrees, edges):
                break

            hook = True

        #print(f"|Edges| = {len(set(edges))}")
        #print(edges)

    assert(len(store[0]) == length) # make sure the nodes were actually initialized
    edges = store[1]
    sub_len = int(len(edges)/2)
    return edges[0:sub_len]


def get_tsp_cd():
    coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
    dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
        (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
    return coords, dists

def run_prob(exp_name=QUEENS, alg_list=None, pd=None, dim=None):

    # Define initial state
    plot_data = {}
    for k,v in alg_list.items():
        print(f'running {k}')
        pr_obj = get_prob_obj(fit_tag=exp_name, dm_tag=DISCR, dim=dim)
        if pr_obj is None:
            continue
        kwa = alg_list[k][AK]()
        ret = alg_list[k][FK](pr_obj, **kwa)
        cur_key = k
        plot_data[cur_key] = {}
        plot_data[cur_key][BSK] = ret[0]
        plot_data[cur_key][BFK] = ret[1]
        plot_data[cur_key][BFCK] = ret[2]
        print(f"Best state = {ret[0]}")
        print(f"Best fitness = {ret[1]}")

    return plot_data


def diff_series(s1, s2):
    r1 = s1 - s2
    if len(np.where(s1 == 0)[0].tolist()) > 0:
        r2 = None
    else:
        r2 = (s1  - s2) / s1

    return r1, r2


def run_alg_vs_complexity(pr_tag=None, start=None, to_explore=None, step=None,
        optimal=None, baseline=None, tbaseline=None
    ):

    print(f'Optimization Problem: {pr_tag}')
    x = numpy.zeros(to_explore)
    y = numpy.zeros(to_explore)
    plot_data = {}
    time_plot_data = {}
    xdata = {}
    txdata = {}

    for alg_tag in ALL_ALG_TAGS:
        print(f'running {alg_tag}')

        results = []
        import time
        tic = time.time()
        end_range = 1 + start + (to_explore - 1)*step
        for i in range(start, end_range, step):

            if i % 10 == 0:
                print(f"Problem dimension = {i}")

            pr_obj = get_prob_obj(fit_tag=pr_tag, dm_tag=DISCR, dim=i)
            if pr_obj is None:
                continue
            alg = opt_algos[alg_tag]
            if not optimal or alg_tag not in optimal:
                kwa = alg[AK](curve=False)
            else:
                kwa = alg[AK](curve=False, **optimal[alg_tag])

            ret = alg[FK](pr_obj, **kwa)
            toc = time.time() - tic
            results.append((i, ret[1], ret[0], toc))


        ax = [x[0] for x in results]
        ay = [x[1] for x in results]
        max_idx = np.argmax(ay)
        curtxt = f" max={ay[max_idx]} @ {ax[max_idx]}"
        xdata[alg_tag] = curtxt
        cur_key = alg_tag
        plot_data[cur_key] = {}
        time_plot_data[cur_key]={}
        #plot_data[cur_key][BSK] = None
        #plot_data[cur_key][BFK] = None
        plot_series = np.column_stack((np.array(ay), np.array(ax)))
        plot_data[cur_key] = plot_series
        axt = [x[0] for x in results]
        ayt = [x[3] for x in results]
        t_plot_series = np.column_stack((np.array(ayt), np.array(axt)))
        time_plot_data[cur_key] = t_plot_series
        txdata[alg_tag] = f" max={ayt[-1]} @ {axt[-1]}" # time is monotonic

    if not optimal:
        end = start + (to_explore - 1) * step
        title = get_plot_title(pr_tag, ":Performance-", f"size:{start}-{end}")
        log(f"{title}", {'plot_data': plot_data})
        fn_tag = pr_tag + '-' + DISCR + '-' + "complexity_"
        fn = get_plot_fn(tag=fn_tag)
        make_fitness_plots(title=title, data=plot_data, fn=fn, x_label="size", extra_data=xdata)
        title = get_plot_title(pr_tag, ":Time_Performance-", f"size:{start}-{end}")
        log(f"{title}", {'plot_data': time_plot_data})
        fn_tag = pr_tag + '-' + DISCR + '-' + "time-complexity_"
        fn = get_plot_fn(tag=fn_tag)
        make_fitness_plots(title=title, data=time_plot_data, fn=fn, x_label="size", y_label="time", extra_data=None)
    else:
        if not baseline:
            pass
        else:
            for k3 in tbaseline:
                if k3 in optimal:
                    diffs, rdiffs = diff_series(baseline[k3][:, 0], plot_data[k3][:, 0])
                    log(f"{k3} optimal deltas from baseline: ", diffs)
                    if rdiffs is None:
                        log("Zero divisors not calculating relative differences")
                    else:
                        log(f"{k3} optimal rel deltas: ", rdiffs)

            title = pr_tag + f":Perf-Tuned:\n{optimal}"
            log(f"{title}-tuned", {'plot_data': plot_data})
            log(f"optimal args=", optimal)
            fn_tag = pr_tag + '-' + DISCR + '-' + "complexity_tuned_"
            fn = get_plot_fn(tag=fn_tag)
            pd2 = {k + "_0":v for k,v in baseline.items() if k in optimal}
            for k2 in baseline:
                if k2 in optimal:
                    ax = [x[1] for x in baseline[k2]]
                    ay = [x[0] for x in baseline[k2]]
                    max_idx = np.argmax(ay)
                    curtxt = f" max={ay[max_idx]} @ {ax[max_idx]}"
                    xdata[k2 + "_0"] = curtxt

            pd2.update(plot_data)
            make_fitness_plots(title=title, data=pd2, fn=fn, x_label="size", extra_data=xdata)
        if not tbaseline:
            pass
        else:
            for k3 in tbaseline:
                if k3 in optimal:
                    diffs, rdiffs = diff_series(tbaseline[k3][:, 0], time_plot_data[k3][:, 0])
                    log(f"{k3} time deltas: ", diffs)
                    if rdiffs is None:
                        log("Zero divisors not calculating relative differences")
                    else:
                        log(f"{k3} rel times deltas: ", rdiffs)

            title = get_plot_title(pr_tag, ":Time_Performance-Tuned", f"size:{start}-{start + to_explore - 1}")
            log(f"{title}-tuned", {'plot_data': time_plot_data})
            log(f"optimal args=", optimal)
            fn_tag = pr_tag + '-' + DISCR + '-' + "time-complexity_tuned_"
            fn = get_plot_fn(tag=fn_tag)
            tpd2 = {k + "_0": v for k, v in tbaseline.items() if k in optimal}
            for k2 in tbaseline:
                if k2 in optimal:
                    ax = [x[1] for x in tbaseline[k2]]
                    ay = [x[0] for x in tbaseline[k2]]
                    curtxt = f" max={ay[-1]} @ {ax[-1]}"
                    txdata[k2 + "_0"] = curtxt

            tpd2.update(time_plot_data)
            make_fitness_plots(title=title, data=tpd2, fn=fn,
                x_label="size", y_label="time", extra_data=txdata
            )

    return plot_data, time_plot_data # will be used as baselines later

def run_alg_tuning(alg_tag=None, alg_param=None, dim=None):

    # Define initial state
    plot_data = {}
    xdata = {}
    pr_tag = COMPLEX_PLOTS[alg_tag]

    param = list(alg_param)[0]
    param_values = alg_param[param]
    print(f'tuning plot for {param}')
    pr_obj = get_prob_obj(fit_tag=pr_tag, dm_tag=DISCR, dim=dim)
    for v in param_values:
        print(f'value = {v}')
        pd = {param: v}
        kwa = opt_algos[alg_tag][AK](**pd)
        ret = opt_algos[alg_tag][FK](pr_obj, **kwa)
        cur_key = f"{param}:{v}"
        plot_data[cur_key] = {}
        plot_data[cur_key][BSK] = ret[0]
        plot_data[cur_key][BFK] = ret[1]
        plot_data[cur_key][BFCK] = ret[2]

        ax = [x[1] for x in ret[2]]
        ay = [x[0] for x in ret[2]]
        max_idx = np.argmax(ay)
        xdata[cur_key] = f"max={ay[max_idx]} @ {ax[max_idx]}"

    title= f"{alg_tag}:{param} on {pr_tag}:sz:{dim}"
    log(f"{title}", {'plot_data': plot_data})
    fn_tag = pr_tag + '-' + alg_tag + '-' + param + f"-d-{dim}-"
    fn = get_plot_fn(tag=fn_tag)
    pd2 = {k: v[BFCK] if BFCK in v else v for k, v in plot_data.items()}  # strip out unnecessary
    make_fitness_plots(title=title, data=pd2, fn=fn, trunc_converged=False, extra_data=xdata)

def run_all_opt(dim=None):

    for pr in all_problems:
        print(f'Optimization Problem: {pr}')
        pd = run_prob(exp_name=pr, alg_list=opt_algos, dim=dim)
        xdata = {k:f"{x[BFK]}, {x[BSK]}" for k, x in pd.items()}
        pd2 = {k: v[BFCK] if BFCK in v else v for k, v in pd.items()}  # strip out unnecessary
        title = get_plot_title(pr, DISCR, dim)
        fn_tag = pr + '-' + DISCR + '-' + f"d-{dim}" + '_'
        fn = get_plot_fn(tag=fn_tag)
        make_fitness_plots(title=title, data=pd2, fn=fn, extra_data=xdata)
        log(title, {'plot_data': pd})

def nn_run():
    nn_model1 = mlr.NeuralNetwork(
        phidden_nodes=[2], activation='relu',
        algorithm='random_hill_climb', max_iters=1000,
        bias=True, is_classifier=True, learning_rate=0.0001,
        early_stopping=True, clip_max=5, max_attempts=100,
        random_state=3
    )

# 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
prs = [2,3,5,7,11,13,17,19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
def get_max_pr(num):
    #thresh = .5*num
    thresh = num
    ret = (None, None)
    for i, x in enumerate(prs):
        # if x < thresh:
        if x*x < thresh :
            ret = (i, x)
        else:
            break

    if ret[1] == prs[-1]:
        print("hit max available prime!!!!")

    return ret

def get_repr(snum):
    if type(snum) is str:
        snum = int(snum)

    pr_max = get_max_pr(snum)
    cur_prs = prs[0:pr_max[0] + 1]
    n_repr= [snum%x for x in cur_prs]
    print(n_repr)
    return n_repr

def get_num_for_repr(p_repr="0113"):
    if type(p_repr) is str:
        p_repr = [int(x)%prs[i] for i, x in enumerate(p_repr)]

    max = prs[-1] * prs[-1]
    ret = None
    for x in range(max):
        if x >= 3:
            cur_repr = get_repr(x)
            if len(cur_repr) < len(p_repr):
                continue
            elif len(cur_repr) == len(p_repr):
                if cur_repr == p_repr:
                    ret = x
                    break
            else:
                assert (False)

    return ret


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

def main(args):
    to_explore = 12
    if 'exp' in args:
        # exploratory
        opt_dims = [15]
        if 'opt_dims' in args:
            idx = args.index('opt_dims')
            opt_dims = [int(x) for x in args[idx + 1:] if x.isnumeric()]

        Logs.set_log_file("exploratory_plots")

        for dim in opt_dims:
            run_all_opt(dim=dim)

    all_complexity_res = {}
    if 'complexity' in args:
        Logs.set_log_file("performance")

        for k in ALG_ORDER:
            pr_tag = COMPLEX_PLOTS[k]
            Logs.set_log_file("perform_" + pr_tag)
            pd_baseline, tpd_baseline = run_alg_vs_complexity(pr_tag=pr_tag, start=5, to_explore=to_explore, step=5)
            all_complexity_res[pr_tag] = {}
            all_complexity_res[pr_tag]['baseline'] = pd_baseline
            all_complexity_res[pr_tag]['time baseline'] = tpd_baseline

    if 'tuning' in args:
        Logs.set_log_file("tuning graphs")
        params = {
            SIMAK : [{'schedule':[mlr.ArithDecay, mlr.GeomDecay, mlr.ExpDecay]}],
            GAK : [{'pop_size':[50, 75, 100, 125, 150, 200 ]}, {'mutation_prob': [.05, .1, .15, .2, .25]}],
            MIMICK : [{'pop_size':[50, 100, 150, 200, 250]}, {'keep_pct':[0.1, .15, 0.2, .25, 0.3]}]
        }

        for dim in [20, 30]:
            for k in ALG_ORDER:
                for prm in params[k]:
                    run_alg_tuning(alg_tag=k, alg_param=prm, dim=dim)

    if 'opt' in args:
        Logs.set_log_file("opt-perform")

        for k in ALG_ORDER:
            pr_tag = COMPLEX_PLOTS[k]
            Logs.set_log_file("perform_tuned_" + pr_tag)
            opt_args = {
                GAK:{'pop_size':100, 'mutation_prob': .15},
                MIMICK:{'pop_size':250, 'keep_pct': .25}
            }
            pd_bl = all_complexity_res[pr_tag]['baseline']
            tpd_bl = all_complexity_res[pr_tag]['time baseline']
            run_alg_vs_complexity(
                pr_tag=pr_tag, start=5, to_explore=to_explore,
                step=5, optimal=opt_args, baseline=pd_bl, tbaseline= tpd_bl
            )

    if 'netw' in args:
        Logs.set_log_file("nnet-weights")
        #net_algos = ['random_hill_climb', 'gradient_descent']
        plot_data = {}
        nalgo_keys = [RHCK, SIMAK, GAK, GDK]

        for alg in nalgo_keys:
            print(f'running {alg} on neural net weights')
            ret = run_find_weights(net_algos[alg])
            if alg == GDK:
                ret.fitness_curve *= -1

            plot_data[alg] = ret.fitness_curve

        # def make_fitness_plots(
        #         title=None, data=None, fn=None, rows=1, extra_data=None,
        #         x_label="iterations", y_label="fitness score"
        # ):
        title = 'NN_weights' + ':fitness_curves'
        log(f"{title}", {'plot_data': plot_data})
        fn_tag = 'NN_weights' + '-' + "fitness_"
        fn = get_plot_fn(tag=fn_tag)
        make_fitness_plots(title='NN_weights', data=plot_data, fn=fn, trunc_converged=False)

    #get_repr(148)
    #print(get_num_for_repr(get_repr(148)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
    #get_repr("135")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

