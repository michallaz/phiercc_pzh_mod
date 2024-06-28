import numpy as np, numba as nb, os
from tempfile import NamedTemporaryFile
import SharedArray as sa
from scipy.special import binom

def getDistance(data, func_name, pool, start=0, allowed_missing=0.0, depth=0):
    # przepisanie funkcji na RAM z pominieciem shared array-a
    # depth to wartosc 0 lub 1 i odpowiada 3-ciemu wymiarowi oryginalnej macierzy
    with NamedTemporaryFile(dir='.', prefix='HCC_') as file :
        prefix = 'file://{0}'.format(file.name)
        func = eval(func_name)
        mat_buf = '{0}.mat.sa'.format(prefix)
        mat = sa.create(mat_buf, shape = data.shape, dtype = data.dtype)
        mat[:] = data[:]
        dist_buf = '{0}.dist.sa'.format(prefix)
        #dist = sa.create(dist_buf, shape = [mat.shape[0] - start, mat.shape[0], 2], dtype = np.int32)
        # macierz bedzie "x na x na 1" a nie "x na x na 2"
        # ale liczenie bedziemy musieli zrobic dwa razy kosztem 2 mniejszego uzcia pamieci
        # unikniemy tez koniecznosci robienia tego po dysku
        dist = sa.create(dist_buf, shape=[mat.shape[0] - start, mat.shape[0], 1], dtype=np.int16)
        dist[:] = 0
        #dist = np.zeros(shape = (mat.shape[0] - start, mat.shape[0], 1), dtype = np.int16)

        __parallel_dist(mat_buf, func, dist_buf, mat.shape, pool, start, allowed_missing, depth)

        # usun pliki posrednie pozostanie tylko plik dist zapisywany w glowny mskrypcie
        sa.delete(mat_buf)
        sa.delete(dist_buf)
        #os.unlink(dist_buf[7:])
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

def Getsquareform(data, func_name, pool, start=0, allowed_missing=0.0):
    # przepisanie  oryginalnej funkcji getDistance
    # aby od razu zwracala squareform, ktore mozna wykorzystac do klastrowania
    with NamedTemporaryFile(dir='.', prefix='HCC_') as file :
        prefix = 'file://{0}'.format(file.name)
        func = eval(func_name)
        # Stworz obiekt mat bedacy macierza zawierajaca informacje o wszystkich profilach
        # Nie jest to obiekt bardzo duzy ale bedzie replikowany kilkadziesiat razy wiec musi pozostac
        # w pliku
        mat_buf = '{0}.mat.sa'.format(prefix)
        mat = sa.create(mat_buf, shape = data.shape, dtype = data.dtype)
        ## wypelnij obiek mat danymi
        mat[:] = data[:]

        # Stworz obieky w ktrym trzymamy dystanse
        # bedzie to wektor o dlugosci n * (n-1) /2
        # gdzie n jest rozmiarem macierzy z data i/lub mat

        dist_buf = '{0}.dist.sa'.format(prefix)
        wymiar = int(mat.shape[0] * (mat.shape[0] - 1) / 2)
        dist = sa.create(dist_buf, shape=wymiar, dtype=np.int64)
        ## wypelnij obiekt zerami
        dist[:] = 0

        # uzupelnij obiekt dist_buf odleglosciami
        __parallel_squareform(mat_buf = mat_buf,
                              func = func,
                              dist_buf = dist_buf,
                              n = mat.shape[0],
                              pool = pool,
                              start = start,
                              allowed_missing= allowed_missing)

        # usun sharedarray
        sa.delete(mat_buf)
        sa.delete(dist_buf)

    return dist

def __parallel_squareform(mat_buf, func, dist_buf, n, pool, start=0, allowed_missing=0.05) :
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
    wymiar_oczekiwany = int(n * (n - 1) / 2) / int(n_pool)

    # zrobic warunek by indeksy nie wyskoczyy poza macierz
    # upewnic sie ze podfunkcja zwraca wektoro o czekiwanej dlugosci

    while len(indices) < n_pool:
        if len(indices) == (n_pool - 1):
            indices.append([s,n])
        else:
            for e in range((s + 1),n):
                dlugosc_tymczasowa = int(binom(n, 2) - binom(n - (e-1), 2) + (n - (e -1) - 1) - (binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1)))
                if dlugosc_tymczasowa > wymiar_oczekiwany:
                    indices.append([s,e])
                    s = e

    #for _ in np.arange(n_pool) :
    #    e = np.sqrt(s * s + tot_cmp)
    #    indices.append([s, e])
    #    s = e
    # po skonczonych obliczeniach zaookraglamy w gore indeksy i upewanimy sie ze sa int-ami
    #indices = (np.array(indices)+0.5).astype(int)

    # wywolanie funkcji z func z odpowiednimi parametrami
    # iterujemy po indeksach i podajemy "DUZEJ" funkcji poza wartoscami "stalymi" jak gdzie sa pliki z amcierz itd
    # wartosci s i e czyli pierwszy i ostatni element z macierzy mat dla ktorych mamy policzyc odleglosc
    # do wszystkich pozostalych elementow macierzy mat
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
        # jako ze ide po wierszach macierzy profili to wektor squareform bedzie zawsze liczony od "s" wiersza i "0" kolumny
        # do "e -1" wiersza i ostatniej kolumny ktora ma rozmiar (n-1) gdzie n to rozmia macierzy
        # jako ze mmay policzyc wycinek wetora z dystanami

        # musimy okrelisc jako ten fragmentaryczny wektor bedzie mial dlugosc
        # oraz gdzie w "PELNYM" wektorze dla pelnego dist sie znajdzie


        pozycja_wektor_start = int(binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1))

        pozycja_wektor_end = int(binom(n, 2) - binom(n - (e - 1), 2) + (n - (e - 1) - 1))

        dlugosc_wektora = int(binom(n, 2) - binom(n - (e-1), 2) + (n - (e -1) - 1) - (binom(n, 2) - binom(n - s, 2) + (s + 1 - s - 1)))

        d = func(mat[:, 1:], s, e, dlugosc_wektora, allowed_missing)
        #print(f'Dla indeksow {s} i {e} musze policzyc wektor o dlugosci {dlugosc_wektora}')
        dist[pozycja_wektor_start:pozycja_wektor_end] = d
    del mat, dist

@nb.jit(nopython=True)
def dual_dist_squareform(mat, s, e, dlugosc_wektora, allowed_missing=0.05):
    # da funkcja produke macierz dystansu od razu w postaci
    # squareform co oszczedza nam polowe RAM-u
    # ale tylko do robienia klastrowania , dla tej macierzy [:,:,1] zostaje "poprzedni" kod

    # pozycja w wektorzez V dla dystansu z macierzy kwadratowej o wymiarze n (liczac rozmiar od 0 jako macierz 1x1)
    # pozycja miedzy elementami i-wiersz i j-ta kolumna (tez indeksowana od 0)
    # wyraza wzor
    # binom(n, 2) - binom(n-i, 2) + (j - i -1)
    # tez indeksowany jak python chce od 0
    # np dla n = 10 (macierz 10 na 10) tutaj indeksacja jest od 1
    # i = 0 (pierwszy wiersz) indeksacja od 0
    # j = 1 (druga kolumna) da wartosc
    # 0 (pierwszy element wktora) tez indeksowany od 0
    # dla i=1 j = 2 mamy 9 element wektora (czyli 10 liczac od 1)

    # tworze wektor ktory "przekaze" do glownego wektora
    # pewnie dlatego jako robie obliczenia to zajmuje tyle pamieci na tym etapie ile mam w docelowym wektorze


    dist = np.zeros(dlugosc_wektora, dtype=np.int64)


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



            if ll > al :
                ad += ll - al
                al = ll

            # dist[i - s, j] = int(ad / al * n_loci + 0.5)
            dist[element] = int(ad/al * n_loci + 0.5)
            element+=1

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


    dist = np.zeros(dlugosc_wektora, dtype=np.int64)


    element = 0
    for i in range(s, e) :
        wektor_x = mat[i]
        n_loci = len(wektor_x)
        # ql ile mamy nie zerowych alleli w pierwszym analizaowanym ST
        ql = np.sum(wektor_x > 0)
        for j in range(i+1, mat.shape[0]) :
            wektor_y = mat[j]
            # rl ile mamy nie zerowych alleli w drugim analizowanym ST
            rl = np.sum(wektor_y > 0)
            # ile mamy WSPOLNYCH niezerowych alleli w obu ST
            indeksy = np.where((wektor_x > 0) & (wektor_y > 0))
            al = len(indeksy[0])
            # ile mamy NIEZEROWYCH alleli ktore przyjmuja IDENTYCZNA wartosc
            ad = np.sum(~np.equal(wektor_x[indeksy[0]], wektor_y[indeksy[0]]))
            # Wieksza z wartosci ql i rl pomniejszona o dopuszczalna liczbe zerowych alleli
            ll = max(ql, rl) - allowed_missing * n_loci

            if ll > al:
                ad += ll - al
                al = ll
            # oryginalna definicja dystansu
            dist[element] = np.int16(ad / al * n_loci + 0.5)
            element += 1


    return dist

