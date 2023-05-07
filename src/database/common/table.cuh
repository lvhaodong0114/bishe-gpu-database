#ifndef TABLE_H
#define TABLE_H

#include "stdint.h"
#include "kv.cuh"
#include <vector>
#include "hashTable.cuh"


class ITable{
    public:
        using MetaDataType = uint64_t;
        using KeyType = ycsb::Key;

    virtual ~ITable() = default ; 


    __host__ __device__ virtual void* search_value(const void *key) = 0;

    __host__ __device__ virtual MetaDataType* search_metadata(const void *key) = 0;


    __host__ __device__ virtual void insert(const void *key, const void *value)= 0;

    __host__ __device__ virtual void update(const void *key, const void *value)= 0;

    __host__ __device__ virtual void delete_(const void *key, const void *value)= 0;


    __host__ __device__ virtual int get_size() = 0;

                        virtual void* get_map_ptr() =0;

    // virtual void get_keys_num(void* nums_ptr)= 0;

    // virtual void get_key_by_index(const void* index)= 0;

    // virtual bool get_all_keys(std::vector<KeyType>& vec)= 0;  
};



template <class KeyType,class ValueType>
class Table : public ITable{

    using KV = kv<KeyType,ValueType>;
    using MetaDataType = uint64_t;

public:
    
    Table(int table_id , Gpu_Allocator* allocatorptr):table_id(table_id),map(allocatorptr){
        map.init_host(DEAULT_TABLE_SIZE);
    };
    
    __host__ void init_device(uint32_t mapsize){
        
    };

    __host__ __device__ void* search_value(const void *key){
        KV* kv_ptr=nullptr; 
        map.contain(*(KeyType*)key,&kv_ptr);

        if(kv_ptr){
            printf("in func:search_value. success!");
            return (void*)&(kv_ptr->value);
        }
        else{
            printf("in func:search_value. failed!");
            return nullptr;
        }
    };

    __host__ __device__ MetaDataType* search_metadata(const void *key){
        KV* kv_ptr=nullptr; 
        map.contain(*(KeyType *)key,&kv_ptr);

        if(kv_ptr){
            printf("in func:search_value. success!");
            return &(kv_ptr->value.metadata);
        }
        else{
            printf("in func:search_value. failed!");
            return nullptr;
        }
    };

    __host__ __device__ void insert(const void *key, const void *value){

        KV tmp;
        KeyType* tmp_key_ptr = tmp.getKey();
        ValueType* tmp_value_ptr = tmp.getValue();

        tmp_key_ptr->copy((KeyType*)key);
        tmp_value_ptr->copy((ValueType*)value);
        map.insert(*(KeyType*)key,&tmp);
        // printf("tableinsert\n");
        return;
    };

    __host__ __device__ void update(const void *key, const void *value){

    };

    __host__ __device__ void delete_(const void *key, const void *value){
        map._delete(*(KeyType*)key);
    };

    __host__ __device__ int get_size(){
        return map.get_itemnums();
    };

    void* get_map_ptr(){
        return (void*)&map;
    };
    
    private:
        HashTable<KeyType,ValueType> map;
        int table_id;
};




#endif