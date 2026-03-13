#!/usr/bin/env python

#  pHierCC.py
#  pipeline for Hierarchical Clustering of cgMLST
#
#  Author: Zhemin Zhou
#  Lisence: GPLv3
#
#
#  Example of a rofile file (tab delimited):
#  ST_id gene1 gene2
#  1 1 1
#  2 1 2
#  To execute type:
#  pHierCC --profile <path> \
#          --cpus <int> \
#          --clustering_method "single|complete" \
#          --allowed_missing <float> \
# Output files are named based on the linkage criterion used for clustering
# profile_{clustering_method}_linkage.HierCC.gz and \
# profile_{clustering_method}_linkage.HierCC.index


import sys, gzip, logging, click
import pandas as pd, numpy as np
from multiprocessing import Pool
from scipy.cluster.hierarchy import linkage
import os

try :
    from getDistance import getDistance
    from getDistance import Getsquareform
    from getDistance import dual_dist_single
    from getDistance import ExpandSquareform
    from getDistance import ExpandDistance
except :
    from .getDistance import getDistance
    from .getDistance import Getsquareform
    from .getDistance import dual_dist_single
    from .getDistance import ExpandSquareform
    from .getDistance import ExpandDistance

logging.basicConfig(format='%(asctime)s | %(message)s', stream=sys.stdout, level=logging.INFO)


def prep_index(file_to_index, every=10000):
    """
    Text file with information where data regarding clustering of a given ST starts
    in a gz file. The index contain information about every 10,000 ST
    @param file_to_index:
    @type file_to_index:
    @param every:
    @type every:
    @return:
    @rtype:
    """
    if os.path.exists(file_to_index):
        output_file=file_to_index.replace(".gz", "")
        with open(f'{output_file}.index', 'w') as f, gzip.open(file_to_index) as f2:
            i = 0
            length = 0
            for line in f2:
                if (i % every) == 0:
                    elementy = list(map(lambda x: x.decode('utf-8', errors='replace'), line.split()))
                    f.write(f'{elementy[0]}\t{length}\n')
                length += len(line)
                i += 1
    else:
        raise Exception('Provided file does not exist')
    return True

def prepare_mat(profile_file):
    """
    Creates a numpy array from a profile file in tab-separated format. First row is
    assumed to contain information regarding allele names (columns where allele name starts with # are omitted). If ST
    identifiers are numeric the value that are lower than 0 are ignored.
    Allele variants with values lower than 0 are converted to 0.
    Profile should not contain non-int values (e.g. "-")

    :param profile_file: str, path to file containing a file with profile
    :return: a 2-element list, first element contains a matrix (values are stored as np.int32, second array
    stores names of sequence types (de facto first column from a profile file but without the first row)
    """
    mat = pd.read_csv(profile_file, sep='\t', header=None, dtype=str).values
    allele_columns = np.array([i == 0 or (not h.startswith('#')) for i, h in enumerate(mat[0])])
    mat = mat[1:, allele_columns]
    try:
        mat = mat.astype(np.int32)
        mat = mat[mat.T[0] > 0]
        names = mat.T[0].copy()
    except:
        names = mat.T[0].copy()
        mat.T[0] = np.arange(1, mat.shape[0]+1)
        mat = mat.astype(np.int32)
    mat[mat < 0] = 0
    return mat, names


@click.command()
@click.option('-p', '--profile', help='[INPUT] name of a profile file consisting of a '
                                      'table of columns of the ST numbers and the allelic numbers, '
                                      'separated by tabs. Can be GZIPped.',
              required=True, type=click.Path())
@click.option('-a', '--profile_distance0', help='[INPUT; optional] The .npy output of a'
                                                ' previous pHierCC run with calculated distance 0 (Default: None).',
              default='', required=False, type=click.Path())
@click.option('-b', '--profile_distance1', help='[INPUT; optional] The .npy output of a '
                                           'previous pHierCC run with calculated distance 1 (Default: None).',
              default='', required=False, type=click.Path())
@click.option('-m', '--allowed_missing', help='[INPUT; optional] Allowed proportion of '
                                              'missing genes in pairwise comparisons (Default: 0.05). ',
              default=0.05, type=float)
@click.option('-n', '--n_proc', help='[INPUT; optional] Number of processes (CPUs) to use '
                                     '(Default: 4).', default=4, type=int)
@click.option('--clustering_method', help='[INPUT; optional] A linkage criterion for clustering '
                                          ' (Default: single).', default="single",
              type=click.Choice(['single', 'complete']))
def phierCC(profile, profile_distance0, profile_distance1, n_proc, clustering_method, allowed_missing):
    """
    pHierCC functions takes a file containing allelic profiles (as in https://pubmlst.org/data/), calculates
    distance between each profile (dual_dist function from getDistance) and performs
    hierarchical clustering of the full dataset based on a minimum-spanning (or macimum) tree.

    When ordering.npy and dist0.npy exist in the output directory (from a previous run)
    and --profile_distance0 is NOT provided, incremental mode is activated: old
    distances are reused and only pairs involving new STs are computed.
    """

    output_dir = os.path.dirname(profile)
    if not output_dir:
        output_dir = '.'
    profile_file = profile
    numpy_dist0_out = f'{output_dir}/dist0.npy'
    numpy_dist1_out = f'{output_dir}/dist1.npy'
    ordering_path = f'{output_dir}/ordering.npy'

    # Read profiles file
    mat, names = prepare_mat(profile_file)
    n_loci = mat.shape[1] - 1

    # --- Decide: incremental or full mode ---
    incremental = False
    old_n = 0
    if (not profile_distance0
            and os.path.exists(ordering_path)
            and os.path.exists(numpy_dist0_out)):

        old_ordering = np.load(ordering_path, allow_pickle=True)
        old_n = len(old_ordering)

        if old_n >= mat.shape[0]:
            logging.info('No new STs compared to previous run – falling back to full mode.')
        else:
            new_st_set = set(int(x) for x in mat.T[0])
            missing_sts = [st for st in old_ordering if int(st) not in new_st_set]

            if missing_sts:
                logging.warning(
                    f'{len(missing_sts)} old STs missing from new profile '
                    f'(e.g. {missing_sts[:5]}). Falling back to full mode.')
            else:
                expected_size = old_n * (old_n - 1) // 2
                probe = np.load(numpy_dist0_out, mmap_mode='r', allow_pickle=True)
                if probe.shape[0] != expected_size:
                    logging.warning(
                        f'Old dist0 size {probe.shape[0]} != expected {expected_size}. '
                        f'Falling back to full mode.')
                    del probe
                else:
                    del probe
                    incremental = True

    if incremental:
        n_new_sts = mat.shape[0] - old_n
        logging.info(
            f'Incremental mode: {old_n} existing STs + {n_new_sts} new STs')

        old_id_to_pos = {int(st): pos for pos, st in enumerate(old_ordering)}
        old_mask = np.array([int(row[0]) in old_id_to_pos for row in mat])
        new_mask = ~old_mask

        old_rows = mat[old_mask]
        old_sort = np.array([old_id_to_pos[int(st)] for st in old_rows.T[0]])
        old_rows = old_rows[np.argsort(old_sort)]

        new_rows = mat[new_mask]
        new_absence = np.sum(new_rows[:, 1:] <= 0, axis=1)
        new_rows = new_rows[np.argsort(new_absence, kind='mergesort')]

        mat = np.vstack([old_rows, new_rows])
    else:
        absence = np.sum(mat <= 0, 1)
        mat[:] = mat[np.argsort(absence, kind='mergesort')]

    np.save(ordering_path, mat.T[0].copy(), allow_pickle=True, fix_imports=True)

    logging.info(
        'Loaded in allelic profiles with dimension: {0} and {1}. '
        'The first column is assumed to be type id.'.format(*mat.shape))

    start = 0

    # ---- Distance matrix 0 (condensed / squareform) ----

    if profile_distance0:
        logging.info('Reading user-provided distance matrix 0')
        dist = np.load(profile_distance0, allow_pickle=True)
    elif incremental:
        logging.info('Expanding distance matrix 0 (incremental)')
        pool = Pool(n_proc)
        dist = ExpandSquareform(numpy_dist0_out, old_n, mat, pool,
                                output_dir, allowed_missing)
        logging.info(f'Saving distance matrix 0 to {numpy_dist0_out}')
        np.save(numpy_dist0_out, dist, allow_pickle=True, fix_imports=True)
        pool.close()
    else:
        logging.info('Calculate distance matrix 0 (full)')
        pool = Pool(n_proc)
        dist = Getsquareform(mat, 'dual_dist_squareform', pool,
                             output_dir, start, allowed_missing)
        logging.info(f'Saving distance matrix 0 to {numpy_dist0_out}')
        np.save(numpy_dist0_out, dist, allow_pickle=True, fix_imports=True)
        pool.close()

    # create object for an output matrix
    res = np.repeat(mat.T[0], int(mat.shape[1]) + 1).reshape(mat.shape[0], -1)
    res[res < 0] = np.max(mat.T[0]) + 100
    res.T[0] = mat.T[0]

    logging.info(f'Start {clustering_method} linkage clustering')
    slc = linkage(dist, method=f'{clustering_method}')
    del dist


    index = {s: i for i, s in enumerate(mat.T[0])}
    descendents = [[m] for m in mat.T[0]] + [None for _ in np.arange(mat.shape[0] - 1)]
    for idx, c in enumerate(slc.astype(int)):
        n_id = idx + mat.shape[0]
        d = sorted([int(c[0]), int(c[1])], key=lambda x: descendents[x][0])
        min_id = min(descendents[d[0]])
        descendents[n_id] = descendents[d[0]] + descendents[d[1]]
        for tgt in descendents[d[1]]:
            res[index[tgt], c[2] + 1:] = res[index[min_id], c[2] + 1:]


    # ---- Distance matrix 1 (full lower-triangular) ----

    if profile_distance1:
        logging.info('Reading user-provided distance matrix 1')
        dist = np.load(profile_distance1, allow_pickle=True, fix_imports=True)
    elif incremental and os.path.exists(numpy_dist1_out):
        logging.info('Expanding distance matrix 1 (incremental)')
        pool = Pool(n_proc)
        dist = ExpandDistance(numpy_dist1_out, old_n, mat, pool,
                              output_dir, allowed_missing, depth=1)
        logging.info(f'Saving distance matrix 1 to {numpy_dist1_out}')
        np.save(numpy_dist1_out, dist, allow_pickle=True, fix_imports=True)
        pool.close()
    else:
        pool = Pool(n_proc)
        logging.info('Calculate distance matrix 1 (full)')
        dist = getDistance(mat, 'dual_dist', pool, output_dir,
                           start, allowed_missing, depth=1)
        logging.info(f'Saving distance matrix 1 to {numpy_dist1_out}')
        np.save(numpy_dist1_out, dist, allow_pickle=True, fix_imports=True)
        pool.close()


    logging.info('Attach genomes onto the tree.')
    for id, (r, d) in enumerate(zip(res[start:], dist[:, :, 0])):
        if id + start > 0 :
            i = np.argmin(d[:id+start])
            min_d = d[i]
            if r[min_d + 1] > res[i, min_d + 1]:
                r[min_d + 1:] = res[i, min_d + 1:]

    logging.info('Saving data.')
    res.T[0] = mat.T[0]
    res = res[np.argsort(res.T[0])]

    with gzip.open(f'{output_dir}/profile_{clustering_method}_linkage.HierCC.gz', 'wt') as fout:
        fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci + 1)])))
        for n, r in zip(names, res):
            fout.write('\t'.join([str(n)] + [str(rr) for rr in r[1:]]) + '\n')
    prep_index(f'{output_dir}/profile_{clustering_method}_linkage.HierCC.gz')
    logging.info(f'Saving clustering results to profile_{clustering_method}_linkage.HierCC.gz')

    return True

if __name__ == '__main__':
    phierCC(sys.argv[1:])

