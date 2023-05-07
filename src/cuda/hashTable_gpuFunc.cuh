#ifndef HASHTABLE_GPUFUNC_CUH
#define HASHTABLE_GPUFUNC_CUH

#include "hashTable.cuh"
#include "kv.cuh"

template<class KeyType,class ValueType>
__device__ void insert_into_kvTable(kv<KeyType,ValueType>* table_ptr,uint32_t size,kv<KeyType,ValueType>* kvptr,uint32_t* insertCounter){   
    uint32_t key=kvptr->getKey()->k;
    if(key==KEY_INVALID){
        // atomicAdd(insertCounter,1);
        return;
    }
    uint32_t hash  = hashKey(key);
    uint32_t i = hash %size;
    for(;;i=(i+1)%size){
        /* atomicCAS  arg :: *address,compare,val  */
        /* old == compare ? val : old */
        uint32_t old = atomicCAS(&(table_ptr[i].getKey()->k),KEY_INVALID,key);
        if(old == KEY_INVALID || old==key){
            /* key was set previously */
            table_ptr[i].copy(kvptr);
            if(old == KEY_INVALID){
                atomicAdd(insertCounter,1);
            }
            break;
        }
        
    };
};


template<class KeyType,class ValueType>
__global__ void kernel_Reinsert(kv<KeyType,ValueType>* new_table_ptr,uint32_t new_size,kv<KeyType,ValueType>* old_table_ptr,bool* old_is_delete_flag,uint32_t old_size,uint32_t* insertCounter){

    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= old_size)
        return;
    if(old_is_delete_flag[idx]){
        // atomicAdd(insertCounter,1);
        old_is_delete_flag[idx]=false;
        return;
    }
    kv<KeyType,ValueType>* tmpptr  = &old_table_ptr[idx];
    insert_into_kvTable(new_table_ptr,new_size,tmpptr,insertCounter);
    return;
};


#endif