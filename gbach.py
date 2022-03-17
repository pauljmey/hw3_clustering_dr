

# 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
prs = [2,3,5,7,11,13,17,19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
HALF_THRESHOLD = 0

def get_max_pr(num, test=HALF_THRESHOLD):
    #thresh = .5*num
    thresh = num
    ret = (None, None)
    for i, x in enumerate(prs):
        if test == HALF_THRESHOLD:
            if x <= int(thresh/2) :
                ret = (i, x)
            else:
                break
        else:
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

def main(args):
    #get_repr("135")
    get_repr(148)
    print(get_num_for_repr(get_repr(148)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
