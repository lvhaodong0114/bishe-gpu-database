#ifndef RWKEY_H
#define RWKEY_H

#include "stdint.h"

class RWKey_Base{
    public:
        __host__ __device__ virtual void set_key_ptr(void* dev_keyptr)=0;
        __host__ __device__ virtual void set_kv_ptr(void* dev_kvptr)=0;
        __host__ __device__ virtual void set_tid(uint64_t* tid)=0;

    public:

};

#endif
