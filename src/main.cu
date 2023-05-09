#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>




// #define OP_PARALLEL

#include "random.h"
#include  "test.cuh"
#include "hashTable.cuh"
#include "ycsb/db.h"
#include "Timer.h"
#include "cudarand.cuh"

std::default_random_engine M_Random::e;
bool  M_Random::is_init = false;
bool Timer::start = false;
bool Timer::end = false;
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::start_t;
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::end_t;





int main(){

    //将 buffer 指定为 NULL，关闭标准输出缓冲  防止输出到log文件中乱序
    setbuf(stdout,NULL);


    ycsb::DB db; 

    db.init_db();
    Timer::start_timer();
    db.init_table(0,0,50000);
    // db.search_all(0,200000);
    Timer::end_timer();
    Timer::show_during();


    db.get_kvs_num(0);

    // db.generate_transction(30);




    db.generate_transction(100);
    db.transction_manager_show();
    

    Timer::start_timer();
//    db.test_one_epoch();
    db.test();

    Timer::end_timer();
    Timer::show_during();


    #ifdef OP_PARALLEL
        printf("open operation parallel.\n");
    #else
        printf("close operation parallel.\n");
    #endif
    


    
    return 0;
};