CISD 2019 ROSEAU GALVEZ

Algèbre Linéaire HPC

Pour compiler le code, tapez les commandes suivantes depuis la racine :

mkdir build
cd build
cmake ..
make

Après avoir compilé le code, 3 exécutables seront générés dans build/test :

 - *driver* : comprends quelques test affichant les résultats d'opérations (dgemm, ddot..)
 - *test_valid* : teste la validité des fonctions (vide pour l'instant)
 - *test_perf* : teste les performances des fonctions en sauvegardant les informations utiles sous format *.csv*

La librairie partagée contenant les fonctions définies dans lib/my_lapack.h se trouvera dans build/lib/libmy_lapack.a
