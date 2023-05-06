#ifndef STORAGE_H
#define STORAGE_H

#include "kv.cuh"
#include "RWKey.h"
namespace ycsb{

    template<int N>
    class Storage{
        using KV =  kv<Key,Value>;
        public:
            KV _kvList[N];
    };


    class RWKey{ 
        public:

            __host__ __device__ void set_key_ptr(void* dev_keyptr){
                this->key_ptr = (Key*)dev_keyptr;
                return;
            };
            __host__ __device__ void set_kv_ptr(void* dev_kvptr){
                this->kv_ptr = (kv<Key,Value>*)dev_kvptr;
                return;
            };
            __host__ __device__ void set_tid(uint64_t* tid){
                this->tid = tid;
                return;
            };
        public:
            Key* key_ptr;
            kv<Key,Value>* kv_ptr;
            uint64_t  *tid;
    };
};

#endif