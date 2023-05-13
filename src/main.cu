#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

//实验设定
#include "MARCO_define.h"

#include "random.h"
#include  "test.cuh"
#include "hashTable.cuh"
#include "ycsb/db.h"
#include "Timer.h"
#include "cudarand.cuh"
#include "info.h"

//全局初始化
std::default_random_engine M_Random::e;
bool  M_Random::is_init = false;
bool Timer::start = false;
bool Timer::end = false;
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::start_t;
std::chrono::time_point<std::chrono::high_resolution_clock> Timer::end_t;


int main(){

    //将 buffer 指定为 NULL，关闭标准输出缓冲  防止输出到log文件中乱序
    setbuf(stdout,NULL);


    //设置cuda设备堆大小
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * (1 << 20));
    CUDACHECKERROR();


    ycsb::DB db; 

    db.init_db();
    Timer::start_timer();
    db.init_table(0,0,200000);
    // db.search_all(0,200000);
    Timer::end_timer();
    Timer::show_during();


    db.get_kvs_num(0);

    // db.generate_transction(30);





    db.generate_transction(100);
    // db.generate_transction(50);
    // db.transction_manager_show();
    

    Timer::start_timer();

    test_with_MARCO(db);

    Timer::end_timer();
    Timer::show_during();

    __INFO();

    return 0;
};