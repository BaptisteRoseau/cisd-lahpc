CISD 2019 ROSEAU GALVEZ

Algèbre Linéaire HPC

Prérequis : CMake version 3.13.1 ou supérieure.

Pour compiler le code depuis une console linux, vous pouvez exécuter les commandes suivantes depuis la racine :

- mkdir build
- cd build
- cmake ..
- make

Après avoir compilé le code, 3 exécutables seront générés dans build/test pour chaque implémentation (séquentielle, OpenMP) :

 - *driver_my* : comprend quelques tests affichant les résultats d'opérations (dgemm, ddot..)
 - *test_valid* : teste la validité des fonctions (vide pour l'instant)
 - *test_perf* : teste les performances des fonctions en sauvegardant les informations utiles sous format *.csv*

La bibliothèque partagée contenant les fonctions définies dans lib/my_lapack.h se trouvera dans build/lib/libmy_lapack.a
