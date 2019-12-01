CISD 2019 ROSEAU GALVEZ
=======================

Algèbre Linéaire HPC

Prérequis : CMake version 3.13.1 ou supérieure.

Compilation
-----------

Pour compiler le code depuis une console linux, vous pouvez exécuter les commandes suivantes depuis la racine :

- mkdir build
- cd build
- cmake ..
- make -j4

La bibliothèque partagée contenant les fonctions définies dans lib/my_lapack.h se trouvera dans build/lib/libmy_lapack_all.a

Executables
-----------

Après avoir compilé le code, 4 exécutables seront générés dans build/test pour chaque implémentation (séquentielle, OpenMP) :

- *driver_my_lapack_all* : comprend quelques tests affichant lesrésultats d'opérations (dgemm, ddot..)
- *test_valid_my_lapack_all* : contient quelques tests de validité
- *test_perf_my_lapack_all* : teste les performances du dgemm en sauvegardantles informations utiles dans *dgemm.csv*
- *test_algonum_my_lapack_all* : Lance les tests de M.Faverge sur les implémentations de dgemm et dgetrf

Bonus
-----------
Un utilitaire python permettant de tracer automatiquement des courbes depuis un fichier *.csv* est disponible dans perf/drawCurve.py

    python3 perf/drawCurve.py <path/to/data.csv>

Le format des données doit être le suivant :

    TITLE, XTITLE, YTITLE, XSCALE ("linear" or "log"), YSCALE
    CURVE LABEL
    x0_0, y0_0
    x0_1, y0_1
    ...
    CURVE LABEL
    x1_0, y1_0
    x1_1, y1_1
    x1_2, y1_2
    ...
