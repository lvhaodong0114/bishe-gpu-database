#ifndef TRANSCM_FUNC_CPU_H
#define TRANSCM_FUNC_CPU_H

#define _THREAD_NUM 16

#include <thread>
#include "metadatahelper.cuh"
#include "My_atomic.h"

namespace ycsb{
    namespace cpu{
template <class KeyType,class ValueType,int N>
    void execute_func(int tid,int transction_nums,HashTable<KeyType,ValueType>* map_ptr,Transction<N>* transction_ptr);
    

    template <class KeyType,class ValueType,int N>
    void execute_cpu(HashTable<KeyType,ValueType>* map_ptr,Transction<N>* transction_ptr,int transction_nums){
        std::thread t[_THREAD_NUM];

        for(int i=0;i<_THREAD_NUM;i++){
            t[i] = std::thread(execute_func<KeyType,ValueType,N>,i,transction_nums,map_ptr,transction_ptr);
        }

        for(int i=0;i<_THREAD_NUM;i++){
            t[i].join();
        }
        return;
    };


        template <class KeyType,class ValueType,int N>
    void execute_cpu(HashTable<KeyType,ValueType>* map_ptr,Transction<N>* transction_ptr){
        for(int i=0;i<transction_ptr->operation_numbers;i++){
            KeyType key = transction_ptr->key[i];
            bool update = transction_ptr->update[i];
            kv<KeyType,ValueType>* src_kv_ptr=nullptr;
            kv<KeyType,ValueType>* storage_kv_ptr=&((transction_ptr->storage_ptr)->_kvList[i]);
            bool contain = map_ptr->contain(key,&src_kv_ptr);
            // printf("in <cpu_Execute>  transaction_tid:%d  key:%d  contain:%d    kv_ptr:%p\n",transction_ptr->Tid,key.k,contain,src_kv_ptr);
        
            if(!contain){

                #ifdef OPEN_DELETE   

                    //fake insert
                    KeyType _key = transction_ptr->key[i];
                    kv<KeyType,ValueType> kv;

                    kv.key.copy(&_key);
                    kv.value.device_generate(devState);

                    insert_atomic(device_map_ptr,&kv,&src_kv_ptr);

                #else
                    continue;
                #endif    
            }
            storage_kv_ptr->copy(src_kv_ptr);

            if(update){
                // printf("update.\n");
                // //写操作，保存到storage中
                storage_kv_ptr->value.generate();

                // //初始化Rkey
                RWKey* _rwkey_ptr =  &(transction_ptr->read_key_list_head[transction_ptr->read_key_nums]);
                transction_ptr->read_key_nums++;
                _rwkey_ptr->set_key_ptr((void*)&(transction_ptr->key[i]));
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);

                // //初始化Wkey
                _rwkey_ptr = &(transction_ptr->write_key_list_head[transction_ptr->write_key_nums]);
                transction_ptr->write_key_nums++;
                _rwkey_ptr->set_key_ptr((void*)&transction_ptr->key[i]);
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);
            }else{
                // //初始化Rkey
                RWKey* _rwkey_ptr = &(transction_ptr->read_key_list_head[transction_ptr->read_key_nums]);
                transction_ptr->read_key_nums++;
                _rwkey_ptr->set_key_ptr((void*)&transction_ptr->key[i]);
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);
            }

           
        }
        printf("<TRANSCTION_MANAGER INFO>:   in <cpu_Execute>  transaction_tid:%d  finished. read_key_nums:%d write_key_nums:%d\n",transction_ptr->Tid,transction_ptr->read_key_nums,transction_ptr->write_key_nums);
        return;
    };



    template<class KeyType,class ValueType,int N>
    void reserve_cpu(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr){
        for(int i=0;i<transction_ptr->read_key_nums;i++){
            RWKey* read_key_ptr = &(transction_ptr->read_key_list_head[i]);
            kv<KeyType,ValueType>* src_kv_ptr = read_key_ptr->kv_ptr;

            //设置这个readkey 的tid指针指向src元数据
            read_key_ptr->set_tid(&((src_kv_ptr->value).metadata));
            reserve_read_cpu(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
        }

        //write reservation
        for(int i=0;i<transction_ptr->write_key_nums;i++){
            RWKey* write_key_ptr = &(transction_ptr->write_key_list_head[i]);
            kv<KeyType,ValueType>* src_kv_ptr = write_key_ptr->kv_ptr;

            //设置这个writekey 的tid指针指向src元数据
            write_key_ptr->set_tid(&((src_kv_ptr->value).metadata));
            reserve_write_cpu(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
        }

        printf("in <cpu_Reserve>  transaction_tid:%d  finished.\n",transction_ptr->Tid);
    };

    template <class KeyType,class ValueType,int N>
    void execute_func(int tid,int transction_nums,HashTable<KeyType,ValueType>* map_ptr,Transction<N>* transction_ptr){
        for(;tid<transction_nums;tid=tid+_THREAD_NUM){
            Transction<N>* ptr = &transction_ptr[tid];

            ptr->storage_ptr = new Storage<N>;
            ptr->read_key_list_head=new RWKey[N];
            ptr->write_key_list_head=new RWKey[N];

            ptr->read_key_nums=0;
            ptr->write_key_nums=0;

            execute_cpu(map_ptr,ptr);
            reserve_cpu(map_ptr,ptr);
        }
        return;

    };


    template<int N>
    void _analyze(int tid,int transction_nums,Transction<N>* transction_ptr){
        for(;tid<transction_nums;tid=tid+_THREAD_NUM){
            // printf("analyze %d\n",tid);
            Transction<N>* ptr = &transction_ptr[tid];

            RWKey* ReadkeySet = ptr->read_key_list_head;
            RWKey* WritekeySet = ptr->write_key_list_head;

            //分析raw
            for(int i=0;i<ptr->read_key_nums;i++){
                //读取这个readkey的metadata 里面有epoch和rts信息
                uint64_t metadata = *(ReadkeySet[i].tid);

                uint16_t epoch = MetadataHelper::get_epoch(metadata);
                uint16_t wts   = MetadataHelper::get_wts(metadata);

                if(epoch==ptr->epoch && wts<ptr->Tid && wts!=0){
                    printf("in func<CPU_analyze_dependency>  Transction:%d has raw dependency on key %d  is been Transction:%d write reserved.\n",ptr->Tid,(ReadkeySet[i].key_ptr)->k,wts);
                    ptr->raw = true;
                    // break;
                };
            };

            //分析waw 和 war
            for(int i=0;i<ptr->write_key_nums;i++){
                uint64_t metadata = *(WritekeySet[i].tid);

                uint16_t epoch = MetadataHelper::get_epoch(metadata);
                uint16_t wts   = MetadataHelper::get_wts(metadata);
                uint16_t rts   = MetadataHelper::get_rts(metadata);

                if(epoch==ptr->epoch && rts<ptr->Tid && rts!=0){
                    printf("in func<CPU_analyze_dependency>  Transction:%d has war dependency on key %d  is been Transction:%d read reserved.\n",ptr->Tid,(WritekeySet[i].key_ptr)->k,rts);
                    ptr->war = true;
                }

                if(epoch==ptr->epoch && wts<ptr->Tid && wts!=0){
                    printf("in func<CPU_analyze_dependency>  Transction:%d has waw dependency on key %d  is been Transction:%d read reserved.\n",ptr->Tid,(WritekeySet[i].key_ptr)->k,wts);
                    ptr->waw = true;
                }
            }

        }
        return;
    };


    template<int N>
    void analyze_cpu(Transction<N>* host_transction_ptr,int transction_nums){
        std::thread t[_THREAD_NUM];

        for(int i=0;i<_THREAD_NUM;i++){
            t[i] = std::thread(_analyze<N>,i,transction_nums,host_transction_ptr);
        }

        for(int i=0;i<_THREAD_NUM;i++){
            t[i].join();
        }
        return;
    };


    template <class KeyType,class ValueType,int N>
    void _install_without_reorder_opt(int tid,int transction_nums,HashTable<KeyType,ValueType>* map_ptr,Transction<N>* host_transction_ptr){
        for(;tid<transction_nums;tid=tid+_THREAD_NUM){
            auto ptr = &host_transction_ptr[tid];
            device_install_without_reorder_optmization(map_ptr,ptr);
        }
        return;
    };



    template <class KeyType,class ValueType,int N>
    void install_cpu(HashTable<KeyType,ValueType>* map_ptr,Transction<N>* host_transction_ptr,int transction_nums,bool reorder_optmization=false){
        std::thread t[_THREAD_NUM];

        for(int i=0;i<_THREAD_NUM;i++){
            if(reorder_optmization==false){
                t[i] = std::thread(_install_without_reorder_opt<KeyType,ValueType,N>,i,transction_nums,map_ptr,host_transction_ptr);
            }else{
                
            };
        }

        for(int i=0;i<_THREAD_NUM;i++){
            t[i].join();
        }
        return;
    };


    template <int N>
    void _collect(int tid,int transction_nums,Transction<N>* transction_ptr){
        for(;tid<transction_nums;tid=tid+_THREAD_NUM){
            Transction<N>*  ptr  = &transction_ptr[tid];

            delete ptr->storage_ptr;
            delete[] ptr->read_key_list_head;
            delete[] ptr->write_key_list_head;
            
            ptr->storage_ptr=nullptr;
            ptr->read_key_list_head=nullptr;
            ptr->write_key_list_head=nullptr;
            
            ptr->read_key_nums=0;
            ptr->write_key_nums=0;
        }
        return;
    }


    template <int N>
    void collect_cpu(Transction<N>* host_transction_ptr,int transction_nums){
        std::thread t[_THREAD_NUM];

        for(int i=0;i<_THREAD_NUM;i++){
            t[i] = std::thread( _collect<N>,i,transction_nums,host_transction_ptr);
        }

        for(int i=0;i<_THREAD_NUM;i++){
            t[i].join();
        }
        return;
    };
    
        
    };
};



#endif