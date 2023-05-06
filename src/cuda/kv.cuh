#ifndef KV_CUH
#define KV_CUH

#include "random.h"
#include "cudarand.cuh"

template <class KeyType,class ValueType>
class kv{
    public:
        __host__ __device__ KeyType* getKey(){
            return &key;
        };
        __host__ __device__ ValueType* getValue(){
            return &value;
        };
        __host__ __device__ bool copy(kv* ptr){
            if(ptr){
                key.copy(&ptr->key);
                value.copy(&ptr->value);
                return true;
            }else{
                return false;
            }
        };
    public:
        KeyType key;
        ValueType value;        
};

namespace ycsb{
    struct Key{
        public:
            uint32_t k;
        public:
            __host__ __device__ Key(int t):k(t){};
            __host__ __device__    Key(){};
                __host__ __device__ void copy(Key * src_k){
                    this->k = src_k->k;
                    return;
                };

                __host__ __device__ void operator=(uint32_t n){
                    this->k=n;
                    return;
                };
    };
    
    struct Value{
        public:
            uint64_t metadata;
            char value[10][10];

        public:
            __host__ __device__ void copy(Value *src_v){
                this->metadata = src_v->metadata;
                for(int i=0;i<10;i++){
                    for(int j=0;j<10;j++){
                        this->value[i][j]=src_v->value[i][j];
                    }
                }
                return;
            };

            __host__ void generate(uint64_t metadata=0){
                this->metadata = metadata;
                for(int i=0;i<10;i++){
                    M_Random::random_cstr(this->value[i],10);
                }
                return;
            };

            __device__ void device_generate(curandState *devState,uint64_t metadata=0){
                this->metadata = metadata;
                for(int i=0;i<10;i++){
                    cu_random_cstr(this->value[i],10,devState);
                }
                return;
            };
    };
};






#endif