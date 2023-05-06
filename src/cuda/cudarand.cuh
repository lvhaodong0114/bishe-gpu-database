#ifndef CUDARAND_CUH
#define CUDARAND_CUH


#include "gpuallocator.cuh"
#include "curand_kernel.h"

//提供给外部使用
//在设备端生成随机数和随机cstr
//使用前须在host段初始化随机数生成器

__device__ int cu_random(int low,int high,curandState* state){
    int res = curand(state)%(high-low) + low;
    return res;
};

__device__ bool cu_random_cstr(char* cstr,int size,curandState* state){
    for(int i=0;i<size;i++){
        char tmp = cu_random(33,126,state);
        cstr[i]=tmp;
    }
    return true;
};


/*在host段初始化随机数生成器*/

//生成n个 curandstate  返回第一个指针devstate
curandState * malloc_curand_state(Gpu_Allocator* allocator_ptr,int n){
    curandState *devState=nullptr;
    printf("malloc_curand_state %d.\n",sizeof(curandState)*n);

    allocator_ptr->_cudaMalloc((void**)&devState,sizeof(curandState)*n);
    printf("malloc_curand_state %d.\n",sizeof(curandState)*n);

    return devState;
};

bool free_curand_state(Gpu_Allocator* allocator_ptr,curandState *devState ){
    if(devState){
        allocator_ptr->_cudaFree(devState);
        return true;
    }
    return false;
};


//生成n个 curandstate  返回第一个指针devstate
//推荐调用方式 setup_rand_kernel<<<1,n>>>  n为state的个数
__global__ void setup_rand_kernel(curandState* state,int seed){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    //seed sequence offset *state
    curand_init(seed,idx,0,&state[idx]);
    return;
};


__global__ void test_rand_kernel(curandState* state){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    printf("%d threadid  random number %d  %d\n",idx,cu_random(0,999,&state[idx]),cu_random(0,999,&state[idx]));
};




void curand_cuh_test(){
    Gpu_Allocator allocator(GPUALLOCATOR_DEFAULTSIZE);

    curandState *devState=nullptr;

    int state_number=16;
    devState = malloc_curand_state(&allocator,state_number);
    setup_rand_kernel<<<1,state_number>>>(devState,0);
    // test_rand_kernel<<<2,16>>>(devState);
    test_rand_kernel<<<1,16>>>(devState);
    return;
};


#endif