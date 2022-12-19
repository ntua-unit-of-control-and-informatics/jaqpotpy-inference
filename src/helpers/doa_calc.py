import numpy as nump

def calc_doa(doa_matrix, new_data):
    doaAll = []
    for nd in new_data:
        d1 = nump.dot(nd, doa_matrix)
        ndt = nump.transpose(nd)
        d2 = nump.dot(d1, ndt)
        doa = {'DOA': d2}
        doaAll.append(doa)
    return doaAll