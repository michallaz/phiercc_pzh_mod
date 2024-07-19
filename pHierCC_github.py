#!/usr/bin/env python

# pHierCC.py
# pipeline for Hierarchical Clustering of cgMLST
#
# Author: Zhemin Zhou
# Lisence: GPLv3
#
# New assignment: pHierCC -p <allelic_profile> -o <output_prefix>
# Incremental assignment: pHierCC -p <allelic_profile> -o <output_prefix> -i <old_cluster_npz>
# Input format (tab delimited):
# ST_id gene1 gene2
# 1 1 1
# 2 1 2
# ...

import sys, gzip, logging, click
import pandas as pd, numpy as np
from multiprocessing import Pool #, set_start_method
from scipy.spatial import distance as ssd
from scipy.cluster.hierarchy import linkage
try :
    from getDistance import getDistance
    from getDistance import Getsquareform
    from getDistance import dual_dist_single
except :
    from .getDistance import getDistance
    from .getDistance import Getsquareform
    from .getDistance import dual_dist_single

logging.basicConfig(format='%(asctime)s | %(message)s', stream=sys.stdout, level=logging.INFO)

def prepare_mat(profile_file) :
    """
    Creates a numpy array from a profile file in tab-separated format. First row is
    assumed to contain information regarding allele names (columns where allele name starts with # are ommitted). If ST
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
    try :
        mat = mat.astype(np.int32)
        mat = mat[mat.T[0] > 0]
        names = mat.T[0].copy()
    except :
        names = mat.T[0].copy()
        mat.T[0] = np.arange(1, mat.shape[0]+1)
        mat = mat.astype(np.int32)
    mat[mat < 0] = 0
    return mat, names

@click.command()
@click.option('-p', '--profile', help='[INPUT] name of a profile file consisting of a table of columns of the ST numbers and the allelic numbers, separated by tabs. Can be GZIPped.',
                        required=True)
@click.option('-o', '--output',
                        help='[OUTPUT] Prefix for the output files consisting of a  NUMPY and a TEXT version of the clustering result. ',
                        required=True)
@click.option('-a', '--profile_dist', help='[INPUT; optional] The .npy output of a previous pHierCC run (Default: None). ',
                        default='')
@click.option('-z', '--new_profile_mat', help='[INPUT; optional] The new profile added to clustering of profile (Default: None). ',
                        default='')
@click.option('-m', '--allowed_missing', help='[INPUT; optional] Allowed proportion of missing genes in pairwise comparisons (Default: 0.05). ',
                        default=0.05, type=float)
@click.option('-n', '--n_proc', help='[INPUT; optional] Number of processes (CPUs) to use (Default: 4).', default=4, type=int)

def phierCC(profile, output, profile_dist, new_profile_mat, n_proc, allowed_missing):
    '''pHierCC takes a file containing allelic profiles (as in https://pubmlst.org/data/) and works
    out hierarchical clusters of the full dataset based on a minimum-spanning tree.'''

    if not profile_dist:
        pool = Pool(n_proc)

    # inicjalizacja nazw
    profile_file, cluster_file,  numpy_dist0_out, numpy_dist1_out = profile, output + '.npz', output + '_dist0.npy',  output + '_dist1.npy'

    mat, names = prepare_mat(profile_file)
    n_loci = mat.shape[1] - 1

    logging.info(
        'Loaded in allelic profiles with dimension: {0} and {1}. The first column is assumed to be type id.'.format(
            *mat.shape))
    logging.info('Start HierCC assignments')

    # Sorting profile matrix to push down ST with high number of missing alleles (with 0)
    absence = np.sum(mat <= 0, 1)
    mat[:] = mat[np.argsort(absence, kind='mergesort')]
    typed = {}

    # prepare matrix with profile
    if  profile_dist:
        # jest append wczytujemy nowa macierz rowniez oraz czytujemy dystans policzony dla 'bazowego_profilu'
        mat_new, names_new = prepare_mat(new_profile_mat)
        with open(profile_dist, 'rb') as f:
            dist = np.load(f)
        #od = np.load(old_cluster, allow_pickle=True)
        #cls = od['hierCC']
        #try :
        #    n = od['names']
        #except :
        #    n = cls.T[0]
        #typed = {c: id for id, c in enumerate(n)}

    # if len(typed) > 0:
    #     logging.info('Loaded in {0} old HierCC assignments.'.format(len(typed)))
    #     # mat_idx = np.array([t in typed for t in names])
    #     mat_idx = np.argsort([typed.get(t, len(typed)) for t in names])
    #     mat[:] = mat[mat_idx]
    #     names[:] = names[mat_idx]
    #     start = np.sum([t in typed for t in names])
    #     if names.dtype != np.int64 :
    #         mat.T[0] = np.arange(1, mat.shape[0]+1)
    # else :
    #     start = 0


    start = 0




    if profile_dist:
        logging.info('Calculate distance between new and base profile')
        # tutaj omijamy funkcje get distance wiec czesc parametrow definiujemy sami
        # funkcja w sumie ignoruje tez n
        tot_cmp = (mat.shape[0] * mat.shape[0] - start * start) / 1
        e = int(np.sqrt(start * start + tot_cmp))
        # pamietamy o ominieciu kolumny z nazwa !
        dist_new = dual_dist_single(mat[:, 1:], mat_new[:, 1:], start, e + 1, allowed_missing)

        # dolaczamy wynik do wczesniej wczytanego dystansu liczonego dla bazowego profilu
        # w tym celu dodajemy kolumne z 0 do poprzednio wygenerowanych wynikow
        # i wiersz z nowymi wynikami
        b = np.zeros((dist.shape[0], 1, 2), dtype=dist.dtype)
        tmp = np.concatenate([dist, b], axis=1)
        dist = np.concatenate((tmp, dist_new))

        # dodajemy rowniez nowy wpis do macierzy res, gdzie program trzyma wyniki
        #to_res = np.zeros(shape=(1, res.shape[1]),dtype = res.dtype )
        #to_res[:] = int(names_new[0])
        #res = np.concatenate([res, to_res])
    else:
        logging.info('Calculate distance matrix')
        # o dziwo oryginalna funkcja dual_dist_squareform jest szybsza
        # niz jej implementacja w numpy wiec uzywamy jej ale liczymy dalej przy jej uzyciu squareform'a
        dist = Getsquareform(mat, 'dual_dist_squareform_lessloop_optimized', pool, start, allowed_missing)


        #zapisujemy surowy output dist tdo pliku
        np.save(numpy_dist0_out, dist, allow_pickle=True, fix_imports=True)

        dist1 = '' # niszczymy zmienna dist1 bo do klastrowania nie jest potrzebna i zwolnimy w ten sposob 1/2 allokowanej pamieci
    # prepare existing tree



    if profile_dist:
        mat = np.concatenate((mat, mat_new))
        names = np.concatenate((names, names_new))
    # przygotujemy output czyli macierz res, niezalezenie czy jest to wersja append czy nie
    # tworzymy ja tak samo wystarczy do oryginalne macierzy appendowac macierz new i isc po starym kodzie

    res = np.repeat(mat.T[0], int(mat.shape[1]) + 1).reshape(mat.shape[0], -1)
    res[res < 0] = np.max(mat.T[0]) + 100
    res.T[0] = mat.T[0]

    # klastrowanie
    # to jest dodanie elementow z gornego trojkata
    #dist[:, :, 0] += dist[:, :, 0].T

    logging.info('Start Single linkage clustering')

    #slc = linkage(ssd.squareform(dist[:, :, 0]), method='single')
    slc = linkage(dist,  method='single')


    # przestajemy potrzebowac macierz dist
    del dist
    # if profile_dist:
    #     mat = np.concatenate((mat, mat_new))

    index = {s: i for i, s in enumerate(mat.T[0])}
    descendents = [[m] for m in mat.T[0]] + [None for _ in np.arange(mat.shape[0] - 1)]
    for idx, c in enumerate(slc.astype(int)):
        n_id = idx + mat.shape[0]
        d = sorted([int(c[0]), int(c[1])], key=lambda x: descendents[x][0])
        min_id = min(descendents[d[0]])
        descendents[n_id] = descendents[d[0]] + descendents[d[1]]
        for tgt in descendents[d[1]]:
            res[index[tgt], c[2] + 1:] = res[index[min_id], c[2] + 1:]
    # else:
    #     mat_index_updated = np.append(mat.T[0], names_new[0])
    #     index = { s:i for i, s in enumerate(mat_index_updated) }
    #     descendents = [ [m] for m in mat_index_updated ] + [None for _ in np.arange(mat.shape[0])]
    #     for idx, c in enumerate(slc.astype(int)) :
    #         n_id = idx + mat.shape[0] + 1
    #         d = sorted([int(c[0]), int(c[1])], key=lambda x:descendents[x][0])
    #         min_id = min(descendents[d[0]])
    #         descendents[n_id] = descendents[d[0]] + descendents[d[1]]
    #         for tgt in descendents[d[1]] :
    #             res[index[tgt], c[2]+1:] = res[index[min_id], c[2]+1:]

    # potrzebujemy macierz depth 1

    logging.info('Calculate second distance matrix')
    dist = getDistance(mat, 'dual_dist', pool, start, allowed_missing, depth=1)

    # zapisujemy surowy output dist tdo pliku
    logging.info('Saving second distance matrix')
    np.save(numpy_dist1_out, dist, allow_pickle=True, fix_imports=True)

    logging.info('Attach genomes onto the tree.')
    for id, (r, d) in enumerate(zip(res[start:], dist[:, :, 0])):
        if id + start > 0 :
            i = np.argmin(d[:id+start])
            min_d = d[i]
            if r[min_d + 1] > res[i, min_d + 1]:
                r[min_d + 1:] = res[i, min_d + 1:]

    # teraz sortujemy res, tak by pierwsza kolumna byla od wartosci najmniejszych do najwiekszych
    # w przypadku append chcemy po prostu miec na koncu res nasz wpis
    logging.info('Saving data.')
    if not profile_dist:
        res.T[0] = mat.T[0]
        res = res[np.argsort(res.T[0])]

        np.savez_compressed(cluster_file, hierCC=res, names=names)
        with gzip.open(output + '.HierCC.gz', 'wt') as fout:
            fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci+1)])))
            for n, r in zip(names, res):
                fout.write('\t'.join([str(n)] + [str(rr) for rr in r[1:]]) + '\n')

        logging.info('NPZ  clustering result (for production mode): {0}.npz'.format(output))
        logging.info('TEXT clustering result (for visual inspection and HCCeval): {0}.HierCC.gz'.format(output))
        pool.close()
        return True
    else:
        res.T[0] = mat.T[0]
        res = res[np.argsort(res.T[0])]

        #np.savez_compressed(cluster_file, hierCC=res, names=names)
        with open(output + '.HierCC.txt', 'wt') as fout:
            fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci + 1)])))
            fout.write('\t'.join([str(names[-1])] + [str(rr) for rr in res[-1,1:]]) + '\n')

        logging.info('Saved output to {0}.HierCC.txt'.format(output))
        #logging.info('NPZ  clustering result (for production mode): {0}.npz'.format(output))
        #logging.info('TEXT clustering result (for visual inspection and HCCeval): {0}.HierCC.gz'.format(output))
        return True

if __name__ == '__main__':
    phierCC(sys.argv[1:])

