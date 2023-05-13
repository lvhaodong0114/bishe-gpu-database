#ifndef INFO_H
#define INFO_H

#include "MARCO_define.h"
#include <iostream>

void __INFO(){

    #ifdef OPEN_GPU
            printf("OPEN_GPU.\n");
            #ifdef OP_PARALLEL
                printf("open operation parallel.\n");
            #else
                printf("close operation parallel.\n");
            #endif
    #else
            printf("OPEN_CPU  THREAD_NUM:%d.\n",_THREAD_NUM);
    #endif



    #ifdef OPEN_DELETE 
            printf("SUPPORT TRANSCTION DELETE.\n");
    #endif

    return;
};

template<class DB>
void test_with_MARCO(DB& db){

    #ifdef  TEST_ONE_EPOCH
        #ifdef OPEN_GPU
            db.test_one_epoch();
        #else
            db.test_one_epoch_on_cpu();
        #endif
    #else
        db.test();
    #endif

    return;
};




#endif