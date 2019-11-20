//#include "algonum.h"
#include "Mat.h"
#include "util.h"
#include "cblas.h"

#include <iostream>

using namespace std;
using namespace my_lapack;

/*============ UTILS FOR TESTING PURPOSE =============== */


void print_test_result(int result, int *nb_success, int *nb_tests)
{
    if (result == EXIT_SUCCESS) {
        printf("\x1B[32mSUCCESS\x1B[0m\n");
        (*nb_success)++;
    } else {

        printf("\x1B[31mFAILED\x1B[0m\n");
    }

    (*nb_tests)++;
}

void print_test_summary(int nb_success, int nb_tests)
{
    if (nb_success == nb_tests)
        printf("TESTS SUMMARY: \t\x1B[32m%d\x1B[0m/%d\n", nb_success, nb_tests);
    else
        printf("TESTS SUMMARY: \t\x1B[31m%d\x1B[0m/%d\n", nb_success, nb_tests);
}


/*============ TESTS DEFINITION =============== */


int test_dgemm_square(){
    printf("%s ", __func__);

    return EXIT_SUCCESS;
}


/*============ MAIN CALL =============== */


int main(int argc, char** argv){
    printf("----------- TEST VALID -----------\n");

    //testall_dgemm( my_dgemm_scalaire );

    int nb_success = 0;
    int nb_tests = 0;

    print_test_result(test_dgemm_square(), &nb_success, &nb_tests);

    print_test_summary(nb_success, nb_tests);

    return EXIT_SUCCESS;
}