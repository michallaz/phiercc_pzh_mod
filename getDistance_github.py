import numpy as np, numba as nb, os
from tempfile import NamedTemporaryFile
import SharedArray as sa
from scipy.special import binom
import logging

def getDistance(data, func_name, pool, output_dir, start=0, allowed_missing=0.0, depth=0):
    """
    This function creates a file that stores a distance matrix. For samples and runs a __parallel_dist function
    it is call by the pHierCC.py script
    """

    with NamedTemporaryFile(dir=output_dir, prefix='HCC_') as file:
        prefix = f'file://{file.name}'
        func = eval(func_name)
        mat_buf = '{0}.mat.sa'.format(prefix)
        mat = sa.create(mat_buf, shape=data.shape, dtype=data.dtype)
        mat[:] = data[:]
        dist_buf = '{0}.dist.sa'.format(prefix)
        dist = sa.create(dist_buf, shape=[mat.shape[0] - start, mat.shape[0], 1], dtype=np.int16)
        dist[:] = 0
        __parallel_dist(mat_buf, func, dist_buf, mat.shape, pool, start, allowed_missing, depth)
        # usun pliki posrednie pozostanie tylko plik dist zapisywany w glowny mskrypcie
        sa.delete(mat_buf)
        sa.delete(dist_buf)
    return dist

def __parallel_dist(mat_buf, func, dist_buf, mat_shape, pool, start=0, allowed_missing=0.0, depth=0) :
    n_pool = len(pool._pool)
    tot_cmp = (mat_shape[0] * mat_shape[0] - start * start)/n_pool
    s, indices = start, []
    for _ in np.arange(n_pool) :
        e = np.sqrt(s * s + tot_cmp)
        indices.append([s, e])
        s = e
    indices = (np.array(indices)+0.5).astype(int)
    for _ in pool.imap_unordered(__dist_wrapper, [[func, mat_buf, dist_buf, s, e, start, allowed_missing, depth] for s, e in indices ]) :
        pass
    return

def __dist_wrapper(data) :
    func, mat_buf, dist_buf, s, e, start, allowed_missing, depth = data
    mat = sa.attach(mat_buf)
    dist = sa.attach(dist_buf)
    if e > s :
        d = func(mat[:, 1:], s, e, allowed_missing, depth)
        dist[(s-start):(e-start)] = d
    del mat, dist

@nb.jit(nopython=True)
def dual_dist(mat, s, e, allowed_missing=0.05, depth=0):
    dist = np.zeros((e-s, mat.shape[0], 1), dtype=np.int16 )
    n_loci = mat.shape[1]
    for i in range(s, e) :
        ql = np.sum(mat[i] > 0)
        for j in range(i) :
            rl, ad, al = 0., 1e-4, 1e-4
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    rl += 1
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci

            if depth == 1:
                if ll2 > al :
                    ad += ll2 - al
                    al = ll2
                dist[i-s, j] = int(ad/al * n_loci + 0.5)


            if depth == 0:
                if ll > al :
                    ad += ll - al
                    al = ll
                dist[i-s, j] = int(ad/al * n_loci + 0.5)
    return dist

@nb.jit(nopython=True)
def p_dist(mat, s, e, allowed_missing=0.05):
    dist = np.zeros((e-s, mat.shape[0], 2), dtype=np.int16 )
    n_loci = mat.shape[1]
    for i in range(s, e) :
        for j in range(i) :
            ad, al = 0., 0.
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            dist[i-s, j, 0] = int( -np.log(1.-(ad+0.5)/(al+1.0)) * n_loci * 100. + 0.5)
    return dist


@nb.jit(nopython=True)
def dual_dist_single(mat, new_mat, s, e, allowed_missing=0.05):
    """
    Modification of original function
    :param mat: numpy_ndarray, a matrix with new profile
    :param new_mat: numpy_ndarray, a matrix with old profile
    :param s: int, not used in this functions, kept for historical reason
    :param e: int, number of rows in a matrix
    :param allowed_missing:
    :return:
    """
    # merge 2 profile into a new matrix,
    if not mat.shape[1] == new_mat.shape[1]:
        raise Exception("Provided profiles must have the same number of columns")
    mat = np.concatenate((mat, new_mat))

    # create output variable
    dist = np.zeros((1, mat.shape[0], 2), dtype=np.int16)
    n_loci = mat.shape[1]
    # in dual dist we loop through every row in matrix (i), than we only fill lower triangular (j)
    # and for that coordinates we iterate every column to determine distance. Here "i" can have only one value
    # (identical to mat shape).
    for i in range(e-1, e) :
        ql = np.sum(mat[i] > 0)
        for j in range(i) :
            rl, ad, al = 0., 1e-4, 1e-4
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    rl += 1
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci

            if ll2 > al :
                ad += ll2 - al
                al = ll2
            dist[0, j, 1] = int(ad/al * n_loci + 0.5)

            if ll > al :
                ad += ll - al
                al = ll
            dist[0, j, 0] = int(ad/al * n_loci + 0.5)
    return dist

def Getsquareform(data, func_name, pool, output_dir, start=0, allowed_missing=0.0):
    # przepisanie  oryginalnej funkcji getDistance
    # aby od razu zwracala squareform, ktore mozna wykorzystac do klastrowania
    with NamedTemporaryFile(dir=output_dir, prefix='HCC_') as file :
        prefix = f'file://{file.name}'
        func = eval(func_name)
        # Stworz obiekt mat bedacy macierza zawierajaca informacje o wszystkich profilach
        # Nie jest to obiekt bardzo duzy ale bedzie replikowany kilkadziesiat razy wiec musi pozostac
        # w pliku
        mat_buf = '{0}.mat.sa'.format(prefix)
        mat = sa.create(mat_buf, shape=data.shape, dtype=data.dtype)
        ## wypelnij obiek mat danymi
        mat[:] = data[:]

        # Stworz obieky w ktrym trzymamy dystanse
        # bedzie to wektor o dlugosci n * (n-1) /2
        # gdzie n jest rozmiarem macierzy z data i/lub mat

        dist_buf = '{0}.dist.sa'.format(prefix)
        wymiar = int(mat.shape[0] * (mat.shape[0] - 1) / 2)
        dist = sa.create(dist_buf, shape=wymiar, dtype=np.int16)
        ## wypelnij obiekt zerami
        dist[:] = 0

        # uzupelnij obiekt dist_buf odleglosciami
        # kod ponizej jest tylko czytelniejsza reimplementacja oryginalnej funkcji
        __parallel_squareform(mat_buf=mat_buf,
                              func=func,
                              dist_buf=dist_buf,
                              n=mat.shape[0],
                              pool=pool,
                              start=start,
                              output_dir=output_dir,
                              allowed_missing=allowed_missing)

        # usun sharedarray
        sa.delete(mat_buf)
        sa.delete(dist_buf)

    return dist

def __parallel_squareform(mat_buf, func, dist_buf, n, pool, output_dir, start=0, allowed_missing=0.05) :
    """
    Funkcja jest wrapperem ktory wywoluje worker-ow (tyle ile wynosi pool)
    w celu uzupelnienia wektora zapisanego w dist_buf
    Wektor te zawiera odleglosci miedzy wpisami w macierz z mat_buf
    :param mat_buf: link do pliku w ktorym przechowywana jest macierz dla ktorej wierszy liczymy odleglosc
    :param func: obeikt klasy funkcja zawierajacy funkcje przekazywana worker'om
    :param dist_buf: wektor ktory bedzie uzupelniany wartosciami obliczonymi przy pomocy funkcji z func na macierz z mat_buf
    :param n: ilosc elementow w macierzy mat_buf, macierz dystansow jaka by powstawala mialaby rozmiar n x n
    :param pool: liczba workerow
    :param start: indeks pierwszego elementu z mat_buf od ktorego liczymy odleglosc (de facto zawsze 0)
    :param allowed_missing: parametr z etoki ilosc alleli jaka moze miec wartosc < 0, de facto nie znalezionego tego allelu
    :return: chyba nic to po prostu wrapper do wywolania pool-a
    """

    n_pool = len(pool._pool)
    # calkowita liczba obliczen na pool-a
    # jako ze start to zwykle 0 oznacza to ze dzielimy liczbe wszystkich obliczen przez liczbe przydzielonych prockow
    tot_cmp = (n * n - start * start)/n_pool
    # indices to lista trzymajaca informacje jaka czesc macierzy dystansu liczymy
    # np [4,10] oznaczalovy ze liczymy odleglosci miedzy elementami od 4 do 10 a WSZYSTKIMI pozostalymi indeksami w macierzy

    s, indices = start, []
    # Podziel caly wektor dla danych na rowne odcinki
    wymiar_oczekiwany = int(n * (n - 1) / 2) / int(n_pool)

    # zrobic warunek by indeksy nie wyskoczyy poza macierz
    # upewnic sie ze podfunkcja zwraca wektoro o czekiwanej dlugosci

    while len(indices) < n_pool:
        if len(indices) == (n_pool - 1):
            # assume that indeces end at n
            indices.append([s, n])
        else:
            for e in range((s + 1), n):
                # policz ile wynosi wektor dystansow jaki trzeba policzyc dla macierzy miedzy indeksem s a e
                dlugosc_tymczasowa = int(binom(n, 2) - binom(n - (e-1), 2) + (n - (e -1) - 1) - (binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1)))
                # w koncu znajdujemy taki e (dla ustalonego s) ktore jest dluzsze niz wektor oczekiwany,
                # zapisz te indeksy i zacznij ponownie dla nowego s
                if dlugosc_tymczasowa > wymiar_oczekiwany:
                    indices.append([s, e])
                    s = e
    # profilaktycznie zapisz odcinki do pliku
    with open(f'{output_dir}/macierz_indeksy.txt', 'w') as f:
        for s,e in indices:
            f.write(f'Analizuje przypadki od {s} do {e}\n')

    # wywolanie funkcji z func z odpowiednimi parametrami
    # iterujemy po indeksach i podajemy "DUZEJ" funkcji poza wartoscami "stalymi" jak gdzie sa pliki z amcierz itd
    # wartosci s i e czyli pierwszy i ostatni element z macierzy mat dla ktorych mamy policzyc odleglosc
    # do wszystkich pozostalych elementow macierzy mat
    # macierz operuje
    for _ in pool.imap_unordered(__dist_wrapper_squareform, [[func, mat_buf, dist_buf, s, e, start, allowed_missing, n] for s, e in indices ]) :
        pass
    return

def __dist_wrapper_squareform(data) :
    """
    Teraz mamy wlasciwe wrapper ktory wyowluje funkcje
    :param data:  wektor elementy przyjmowane przez "wlasciwa" funkcje
    :return: sherad array-e mat czyli macierz z oryginalnymi danymi i dist, czyli wektor z policzonymi dystansami
    """
    func, mat_buf, dist_buf, s, e, start, allowed_missing, n = data
    # polacz lokalne mat i dist z shared arrayami
    mat = sa.attach(mat_buf)
    dist = sa.attach(dist_buf)
    if e > s :
        # jako ze ide po wierszach macierzy profili to wektor squareform bedzie zawsze liczony od "s" wiersza i s+1 kolumny
        # do "e -1" wiersza i ostatniej kolumny ktora ma rozmiar (n-1) gdzie n to rozmia macierzy
        # jako ze mmay policzyc wycinek wetora z dystanami

        # musimy okrelisc jako ten fragmentaryczny wektor bedzie mial dlugosc
        # oraz gdzie w "PELNYM" wektorze dla pelnego dist sie znajdzie


        pozycja_wektor_start = int(binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1))
        pozycja_wektor_end = int(binom(n, 2) - binom(n - (e - 1), 2) + (n - (e - 1) - 1))
        dlugosc_wektora = int(binom(n, 2) - binom(n - (e-1), 2) + (n - (e -1) - 1) - (binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1)))

        # pusty przelot dla numby co sugeruje chatgpt
        sample_matrix = np.random.randint(0, 2, size=(3, 10))
        func(sample_matrix, 0, 3, int(3 * (3 - 1) / 2))

        # wywolaj obliczenia funkcja, liczy dystans dla macierzy od indeksow s do e
        # ktory nastepnie wklejamy w duzy wektor
        d = func(mat[:, 1:], s, e, dlugosc_wektora, allowed_missing)

        dist[pozycja_wektor_start:pozycja_wektor_end] = d
    del mat, dist

@nb.jit(nopython=True)
def dual_dist_squareform(mat, s, e, dlugosc_wektora, allowed_missing=0.05):
    #  da funkcja produke macierz dystansu od razu w postaci
    #  squareform co oszczedza nam polowe RAM-u
    #  ale tylko do robienia klastrowania , dla tej macierzy [:,:,1] zostaje "poprzedni" kod

    #  pozycja w wektorzez V dla dystansu z macierzy kwadratowej o wymiarze n (liczac rozmiar od 0 jako macierz 1x1)
    #  pozycja miedzy elementami i-wiersz i j-ta kolumna (tez indeksowana od 0)
    #  wyraza wzor
    #  binom(n, 2) - binom(n-i, 2) + (j - i -1)
    #  tez indeksowany jak python chce od 0
    #  np dla n = 10 (macierz 10 na 10) tutaj indeksacja jest od 1
    #  i = 0 (pierwszy wiersz) indeksacja od 0
    #  j = 1 (druga kolumna) da wartosc
    #  0 (pierwszy element wktora) tez indeksowany od 0
    #  dla i=1 j = 2 mamy 9 element wektora (czyli 10 liczac od 1)

    #  tworze wektor ktory "przekaze" do glownego wektora
    dist = np.zeros(dlugosc_wektora, dtype=np.int16)
    n_loci = mat.shape[1]
    element = 0
    for i in range(s, e) :
        ql = np.sum(mat[i] > 0)
        # idziemy po kolumnach ale nie od o do i (dolny fragment) a od (i do konca)
        for j in range(i+1, mat.shape[0]) :
            #print(i,j)
            rl, ad, al = 0., 1e-4, 1e-4
            for k in range(n_loci) :
                if mat[j, k] > 0 :
                    rl += 1
                    if mat[i, k] > 0 :
                        al += 1
                        if mat[i, k] != mat[j, k] :
                            ad += 1
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci
            # we use this function only for "first" distnace matrix, ll2 is not used during its calculations
            if ll > al :
                ad += ll - al
                al = ll
            # dist[i - s, j] = int(ad / al * n_loci + 0.5)
            dist[element] = int(ad/al * n_loci + 0.5)
            element += 1
    return dist


@nb.jit(nopython=True)
def dual_dist_squareform_lessloop(mat, s, e, dlugosc_wektora, allowed_missing=0.05):
    """
    Modyfikacja funkcji dual_dist_squareform, wyjmuje czesc petli z obliczen
    :param mat:
    :param s:
    :param e:
    :param dlugosc_wektora:
    :param allowed_missing:
    :return:
    """


    dist = np.zeros(dlugosc_wektora, dtype=np.int16)
    n_loci = len(mat[0])

    element = 0
    for i in range(s, e) :
        wektor_x = mat[i]
        # ql ile mamy nie zerowych alleli w pierwszym analizaowanym ST
        ql = np.sum(wektor_x > 0)
        for j in range(i+1, mat.shape[0]) :
            # w oryginalnym kodzie zmienne ad i al inicjowane sa z malymi wartosciami co powoduje
            # roznice przy zaookragleniach ...
            rl, ad, al = 0., 1e-4, 1e-4
            wektor_y = mat[j]
            # rl ile mamy nie zerowych alleli w drugim analizowanym ST
            rl = np.sum(wektor_y > 0)
            # ile mamy WSPOLNYCH niezerowych alleli w obu ST
            common_non_zero = (wektor_x > 0) & (wektor_y > 0)
            al += np.sum(common_non_zero)
            # ile mamy NIEZEROWYCH alleli ktore przyjmuja ROZNA wartosc
            ad += np.sum(wektor_x[common_non_zero] != wektor_y[common_non_zero])
            # Wieksza z wartosci ql i rl pomniejszona o dopuszczalna liczbe zerowych alleli
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci

            #if ll2 > al:
            #    ad += ll2 - al
            #    al = ll2

            # Tutaj zapisany bylby drugi z dystansow nie uzywany do lastrowania

            if ll > al:
                ad += ll - al
                al = ll
            # oryginalna definicja dystansu uzywana do klastrowania
            dist[element] = np.int16(ad / al * n_loci + 0.5)
            element += 1


    return dist


@nb.jit(nopython=True)
def dual_dist_squareform_lessloop_optimized(mat, s, e, dlugosc_wektora, allowed_missing=0.05):
    """
    Modyfikacja funkcji dual_dist_squareform, wyjmuje czesc petli z obliczen
    :param mat:
    :param s:
    :param e:
    :param dlugosc_wektora:
    :param allowed_missing:
    :return:
    """


    dist = []
    n_loci = len(mat[0])

    element = 0
    for i in range(s, e) :
        wektor_x = mat[i]
        # ql ile mamy nie zerowych alleli w pierwszym analizaowanym ST
        ql = np.sum(wektor_x > 0)
        for j in range(i+1, mat.shape[0]) :
            # w oryginalnym kodzie zmienne ad i al inicjowane sa z malymi wartosciami co powoduje
            # roznice przy zaookragleniach ...
            rl, ad, al = 0., 1e-4, 1e-4
            wektor_y = mat[j]
            # rl ile mamy nie zerowych alleli w drugim analizowanym ST
            rl = np.sum(wektor_y > 0)
            # ile mamy WSPOLNYCH niezerowych alleli w obu ST
            common_non_zero = (wektor_x > 0) & (wektor_y > 0)
            al += np.sum(common_non_zero)
            # ile mamy NIEZEROWYCH alleli ktore przyjmuja ROZNA wartosc
            ad += np.sum(wektor_x[common_non_zero] != wektor_y[common_non_zero])
            # Wieksza z wartosci ql i rl pomniejszona o dopuszczalna liczbe zerowych alleli
            ll = max(ql, rl) - allowed_missing * n_loci
            ll2 = ql - allowed_missing * n_loci

            #if ll2 > al:
            #    ad += ll2 - al
            #    al = ll2

            # Tutaj zapisany bylby drugi z dystansow nie uzywany do lastrowania

            if ll > al:
                ad += ll - al
                al = ll
            # oryginalna definicja dystansu uzywana do klastrowania
            dist.append(np.int16(ad / al * n_loci + 0.5))


    dist = np.array(dist, dtype = np.int16)
    return dist


# ---------------------------------------------------------------------------
# Branchless v2: eliminate branch misprediction in the inner k-loop
# (benchmark showed no gain: LLVM already converts branches to cmov)
# ---------------------------------------------------------------------------

@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def _precompute_ql(mat):
    """Count positive alleles per row (avoids np.sum(mat[i] > 0) temporaries)."""
    n = mat.shape[0]
    n_loci = mat.shape[1]
    ql = np.empty(n, dtype=np.int32)
    for i in range(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql[i] = c
    return ql


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def dual_dist_squareform_v2(mat, s, e, dlugosc_wektora, allowed_missing=0.05):
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = _precompute_ql(mat)

    dist = np.zeros(dlugosc_wektora, dtype=np.int16)
    element = 0
    for i in range(s, e):
        qi = np.float64(ql_arr[i])
        for j in range(i + 1, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            dist[element] = np.int16(ad / al * n_loci + 0.5)
            element += 1
    return dist


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def dual_dist_v2(mat, s, e, allowed_missing=0.05, depth=0):
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = _precompute_ql(mat)

    dist = np.zeros((e - s, n, 1), dtype=np.int16)
    for i in range(s, e):
        qi = np.float64(ql_arr[i])
        for j in range(i):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4

            if depth == 1:
                ll2 = qi - allowed
                if ll2 > al:
                    ad += ll2 - al
                    al = ll2
                dist[i - s, j, 0] = np.int16(ad / al * n_loci + 0.5)

            if depth == 0:
                ll = max(qi, np.float64(ql_arr[j])) - allowed
                if ll > al:
                    ad += ll - al
                    al = ll
                dist[i - s, j, 0] = np.int16(ad / al * n_loci + 0.5)
    return dist


# ---------------------------------------------------------------------------
# Numba-parallel approach: replace multiprocessing Pool + SharedArray with
# Numba's built-in thread parallelism (prange).  Eliminates process fork,
# shared-memory attachment, and serialisation overhead entirely.
# ---------------------------------------------------------------------------

@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _squareform_numba_parallel(mat, allowed_missing=0.05):
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    size = n * (n - 1) // 2
    dist = np.zeros(size, dtype=np.int16)

    for i in nb.prange(n):
        qi = np.float64(ql_arr[i])
        for j in range(i + 1, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)

    return dist


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _dist1_numba_parallel(mat, start=0, allowed_missing=0.05, depth=0):
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    dist = np.zeros((n - start, n, 1), dtype=np.int16)

    for i in nb.prange(start, n):
        qi = np.float64(ql_arr[i])
        for j in range(i):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4

            if depth == 1:
                ll2 = qi - allowed
                if ll2 > al:
                    ad += ll2 - al
                    al = ll2
                dist[i - start, j, 0] = np.int16(ad / al * n_loci + 0.5)

            if depth == 0:
                ll = max(qi, np.float64(ql_arr[j])) - allowed
                if ll > al:
                    ad += ll - al
                    al = ll
                dist[i - start, j, 0] = np.int16(ad / al * n_loci + 0.5)

    return dist


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _squareform_append_numba_parallel(mat, n_old, dist, allowed_missing=0.05):
    """Compute only new-pair distances (prange), write into pre-allocated dist."""
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    for i in nb.prange(n):
        qi = np.float64(ql_arr[i])
        j_start = n_old if i < n_old else i + 1
        for j in range(j_start, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)


def GetSquareformParallel(data, n_threads, allowed_missing=0.0):
    """Compute condensed distance using Numba parallel threading (no Pool)."""
    nb.set_num_threads(n_threads)
    logging.info(f'Numba parallel: using {nb.get_num_threads()} threads')

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _squareform_numba_parallel(warmup, 0.05)

    dist = _squareform_numba_parallel(data[:, 1:], allowed_missing)
    return dist


def GetDistanceParallel(data, n_threads, start=0, allowed_missing=0.0, depth=0):
    """Compute full distance matrix using Numba parallel threading (no Pool)."""
    nb.set_num_threads(n_threads)

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _dist1_numba_parallel(warmup, 0, 0.05, depth)

    dist = _dist1_numba_parallel(data[:, 1:], start, allowed_missing, depth)
    return dist


def ExpandSquareformParallel(old_dist_path, old_n, new_mat, n_threads,
                              allowed_missing=0.0):
    """Expand condensed distance vector using Numba parallel (no Pool/SharedArray)."""
    nb.set_num_threads(n_threads)
    n_new = new_mat.shape[0]
    old_dist = np.load(old_dist_path, mmap_mode='r', allow_pickle=True)

    new_size = int(n_new * (n_new - 1) / 2)
    dist = np.zeros(new_size, dtype=np.int16)

    logging.info(f'Copying old condensed distances ({old_n} STs) into new vector ({n_new} STs)')
    for i in range(old_n):
        old_start = old_n * i - i * (i + 1) // 2
        new_start = n_new * i - i * (i + 1) // 2
        length = old_n - 1 - i
        if length > 0:
            dist[new_start:new_start + length] = old_dist[old_start:old_start + length]
    del old_dist

    n_new_sts = n_new - old_n
    total_new = old_n * n_new_sts + n_new_sts * (n_new_sts - 1) // 2
    logging.info(f'Computing {total_new} new pairwise distances ({n_new_sts} new STs)')

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    warmup_d = np.zeros(int(4 * 3 / 2), dtype=np.int16)
    _squareform_append_numba_parallel(warmup, 2, warmup_d, 0.05)

    _squareform_append_numba_parallel(new_mat[:, 1:], old_n, dist, allowed_missing)
    return dist


def ExpandDistanceParallel(old_dist_path, old_n, new_mat, n_threads,
                            allowed_missing=0.0, depth=0):
    """Expand full distance matrix using Numba parallel (no Pool/SharedArray)."""
    nb.set_num_threads(n_threads)
    n_new = new_mat.shape[0]

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _dist1_numba_parallel(warmup, 0, 0.05, depth)

    new_rows = _dist1_numba_parallel(new_mat[:, 1:], old_n, allowed_missing, depth)

    full_dist = np.zeros((n_new, n_new, 1), dtype=np.int16)
    old_dist = np.load(old_dist_path, mmap_mode='r', allow_pickle=True)
    full_dist[:old_n, :old_n, :] = old_dist[:, :, :]
    del old_dist

    full_dist[old_n:, :, :] = new_rows
    del new_rows

    return full_dist


# ---------------------------------------------------------------------------
# Incremental (append) mode: reuse old distances, compute only new ST pairs
# ---------------------------------------------------------------------------

@nb.jit(nopython=True)
def _dual_dist_squareform_append(mat, s, e, n_old, n, dist, allowed_missing=0.05):
    """
    Compute distances only for pairs involving at least one new ST and write
    them directly into the full condensed distance vector.

    For rows i in [s, e):
      - i < n_old  → compute d(i, j) for j in [n_old, n)   (old-vs-new)
      - i >= n_old → compute d(i, j) for j in [i+1, n)     (all new)

    The condensed-vector position for pair (i, j) with i < j is:
        pos = n*i - i*(i+1)//2 + (j - i - 1)
    """
    n_loci = mat.shape[1]
    for i in range(s, e):
        ql = np.sum(mat[i] > 0)
        j_start = n_old if i < n_old else i + 1
        for j in range(j_start, n):
            rl, ad, al = 0., 1e-4, 1e-4
            for k in range(n_loci):
                if mat[j, k] > 0:
                    rl += 1
                    if mat[i, k] > 0:
                        al += 1
                        if mat[i, k] != mat[j, k]:
                            ad += 1
            ll = max(ql, rl) - allowed_missing * n_loci
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)


def _dist_wrapper_squareform_append(data):
    mat_buf, dist_buf, s, e, n_old, n, allowed_missing = data
    mat = sa.attach(mat_buf)
    dist = sa.attach(dist_buf)
    if e > s:
        sample = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
        sample_d = np.zeros(int(4 * 3 / 2), dtype=np.int16)
        _dual_dist_squareform_append(sample, 0, 2, 0, 4, sample_d, 0.05)
        _dual_dist_squareform_append(mat[:, 1:], s, e, n_old, n, dist, allowed_missing)
    del mat, dist


def _parallel_squareform_append(mat_buf, dist_buf, n, n_old, pool, output_dir,
                                allowed_missing=0.05):
    n_pool = len(pool._pool)

    total_new_pairs = n_old * (n - n_old)
    for i in range(n_old, n):
        total_new_pairs += n - 1 - i
    target = total_new_pairs / n_pool

    indices = []
    s = 0
    cumulative = 0
    for i in range(n):
        cumulative += (n - n_old) if i < n_old else (n - 1 - i)
        if cumulative >= target and len(indices) < n_pool - 1:
            indices.append([s, i + 1])
            s = i + 1
            cumulative = 0
    indices.append([s, n])

    with open(f'{output_dir}/macierz_indeksy_append.txt', 'w') as f:
        for si, ei in indices:
            f.write(f'Append mode: rows {si} to {ei}\n')

    for _ in pool.imap_unordered(
        _dist_wrapper_squareform_append,
        [[mat_buf, dist_buf, si, ei, n_old, n, allowed_missing]
         for si, ei in indices]):
        pass


@nb.jit(nopython=True, fastmath=True, boundscheck=False)
def _dual_dist_squareform_append_v2(mat, s, e, n_old, n, dist, allowed_missing=0.05):
    """Branchless v2 of _dual_dist_squareform_append."""
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci
    ql_arr = _precompute_ql(mat)

    for i in range(s, e):
        qi = np.float64(ql_arr[i])
        j_start = n_old if i < n_old else i + 1
        for j in range(j_start, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)


def _dist_wrapper_squareform_append_v2(data):
    mat_buf, dist_buf, s, e, n_old, n, allowed_missing = data
    mat = sa.attach(mat_buf)
    dist = sa.attach(dist_buf)
    if e > s:
        sample = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
        sample_d = np.zeros(int(4 * 3 / 2), dtype=np.int16)
        _dual_dist_squareform_append_v2(sample, 0, 2, 0, 4, sample_d, 0.05)
        _dual_dist_squareform_append_v2(mat[:, 1:], s, e, n_old, n, dist, allowed_missing)
    del mat, dist


def _parallel_squareform_append_v2(mat_buf, dist_buf, n, n_old, pool, output_dir,
                                   allowed_missing=0.05):
    n_pool = len(pool._pool)

    total_new_pairs = n_old * (n - n_old)
    for i in range(n_old, n):
        total_new_pairs += n - 1 - i
    target = total_new_pairs / n_pool

    indices = []
    s = 0
    cumulative = 0
    for i in range(n):
        cumulative += (n - n_old) if i < n_old else (n - 1 - i)
        if cumulative >= target and len(indices) < n_pool - 1:
            indices.append([s, i + 1])
            s = i + 1
            cumulative = 0
    indices.append([s, n])

    with open(f'{output_dir}/macierz_indeksy_append.txt', 'w') as f:
        for si, ei in indices:
            f.write(f'Append mode (v2): rows {si} to {ei}\n')

    for _ in pool.imap_unordered(
        _dist_wrapper_squareform_append_v2,
        [[mat_buf, dist_buf, si, ei, n_old, n, allowed_missing]
         for si, ei in indices]):
        pass


def ExpandSquareform(old_dist_path, old_n, new_mat, pool, output_dir,
                     allowed_missing=0.0):
    """
    Expand an existing condensed distance vector with newly appended STs.

    Parameters
    ----------
    old_dist_path : str
        Path to the .npy file with the previous condensed distance vector.
    old_n : int
        Number of STs in the previous run (indices 0..old_n-1 in new_mat must
        be in the same order as last time).
    new_mat : ndarray, shape (n_new, n_loci+1), int32
        Full profile matrix – first old_n rows in old order, remaining rows are
        new STs.
    pool : multiprocessing.Pool
    output_dir : str
    allowed_missing : float

    Returns
    -------
    dist : ndarray, 1-D int16
        New condensed distance vector of length n_new*(n_new-1)/2.
    """
    n_new = new_mat.shape[0]
    old_dist = np.load(old_dist_path, mmap_mode='r', allow_pickle=True)

    with NamedTemporaryFile(dir=output_dir, prefix='HCC_') as file:
        prefix = f'file://{file.name}'

        mat_buf = f'{prefix}.mat.sa'
        mat = sa.create(mat_buf, shape=new_mat.shape, dtype=new_mat.dtype)
        mat[:] = new_mat[:]

        dist_buf = f'{prefix}.dist.sa'
        new_size = int(n_new * (n_new - 1) / 2)
        dist = sa.create(dist_buf, shape=new_size, dtype=np.int16)
        dist[:] = 0

        logging.info(f'Copying old condensed distances ({old_n} STs) into new vector ({n_new} STs)')
        for i in range(old_n):
            old_start = old_n * i - i * (i + 1) // 2
            new_start = n_new * i - i * (i + 1) // 2
            length = old_n - 1 - i
            if length > 0:
                dist[new_start:new_start + length] = old_dist[old_start:old_start + length]

        del old_dist

        n_new_sts = n_new - old_n
        total_new = old_n * n_new_sts + n_new_sts * (n_new_sts - 1) // 2
        logging.info(f'Computing {total_new} new pairwise distances ({n_new_sts} new STs)')

        _parallel_squareform_append(
            mat_buf=mat_buf,
            dist_buf=dist_buf,
            n=n_new,
            n_old=old_n,
            pool=pool,
            output_dir=output_dir,
            allowed_missing=allowed_missing)

        sa.delete(mat_buf)
        sa.delete(dist_buf)

    return dist


def ExpandDistance(old_dist_path, old_n, new_mat, pool, output_dir,
                  allowed_missing=0.0, depth=0):
    """
    Expand an existing full distance matrix (n_old, n_old, 1) with new STs.

    Old rows are copied from the previous matrix; new rows (indices
    old_n..n_new-1) are computed via getDistance with start=old_n.
    Uses mmap on the old file to avoid loading ~720 GB into RAM at once.

    Returns
    -------
    full_dist : ndarray, shape (n_new, n_new, 1), int16
    """
    n_new = new_mat.shape[0]

    new_rows = getDistance(new_mat, 'dual_dist', pool, output_dir,
                           start=old_n, allowed_missing=allowed_missing,
                           depth=depth)

    full_dist = np.zeros((n_new, n_new, 1), dtype=np.int16)

    old_dist = np.load(old_dist_path, mmap_mode='r', allow_pickle=True)
    full_dist[:old_n, :old_n, :] = old_dist[:, :, :]
    del old_dist

    full_dist[old_n:, :, :] = new_rows
    del new_rows

    return full_dist

