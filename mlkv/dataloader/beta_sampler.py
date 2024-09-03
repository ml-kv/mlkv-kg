from dgl._ffi.function import _init_api
import numpy as np
import math

def BetaPartition(n_entities, edges, n, has_importance=False):
    """This partitions a list of edges into n * n buckets

    Parameters
    ----------
    n_entities: int
        number of entities
    edges : (heads, rels, tails) triple
        Edge list to assign
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each buckets
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('partition {} entities into {} partitions'.format(n_entities, n))
    print('assign {} edges into {} buckets'.format(len(heads), n*n))

    part_size = int(math.ceil(n_entities / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), n_entities)
        parts.append((start, end))
        print('part {} has {} nodes'.format(i, end - start))

    import multiprocessing as mp
    queue = mp.Queue()
    num_assign_procs = 8
    assign_procs = []

    def assign_edges_to_buckets(queue, begin, end):
        sub_buckets = [[] for _ in range(n * n)]
        for i in range(begin, end):
            head_part = int(math.floor(heads[i] / part_size))
            tail_part = int(math.floor(tails[i] / part_size))
            sub_buckets[head_part * n + tail_part].append(i)
        queue.put(sub_buckets)

    for i in range(num_assign_procs):
        begin = int(i * len(heads) / num_assign_procs)
        end = int((i + 1) * len(heads) / num_assign_procs)
        proc = mp.Process(target=assign_edges_to_buckets, args=(queue, begin, end))
        assign_procs.append(proc)
        proc.start()

    buckets = [[] for _ in range(n * n)]
    for _ in range(num_assign_procs):
        sub_buckets = queue.get()
        for i in range(n * n):
            buckets[i] = buckets[i] + sub_buckets[i]

    buckets_ndarray = []
    for i in range(n * n):
        buckets_ndarray.append(np.array(buckets[i]))
        print('bucket {} has {} edges'.format(i, len(buckets_ndarray[i])))

    return buckets_ndarray, parts

def SetNegRangeUniformEdgeSample(sampler, lower, upper):
    _CAPI_SetNegRangeUniformEdgeSample(sampler, lower, upper)

_init_api('dgl.sampling', __name__)
