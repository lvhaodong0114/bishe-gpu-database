#ifndef TRANCM_FUNC_CUH
#define TRANCM_FUNC_CUH

#include "metadatahelper.cuh"

namespace ycsb{


template<class KeyType,class ValueType>
__global__ void kernel_show_table(HashTable<KeyType,ValueType>* device_map_ptr){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<device_map_ptr->Size){
        kv<KeyType,ValueType>* table_ptr = device_map_ptr->TablePtr;
        printf("thread id:%d         key:%d\n",idx,table_ptr[idx].key.k);
    }
    return;
};

//在gpu上执行 一个事务的所有操作，将结果存储在stroge中  并初始化了rwkey
template<class KeyType,class ValueType,int N>
__device__ void device_execute(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
    //
    for(int i=0;i<transction_ptr->operation_numbers;i++){
        KeyType key = transction_ptr->key[i];
        bool update = transction_ptr->update[i];
        kv<KeyType,ValueType>* src_kv_ptr=nullptr;
        kv<KeyType,ValueType>* storage_kv_ptr=&((transction_ptr->storage_ptr)->_kvList[i]);
        bool contain = device_map_ptr->contain(key,&src_kv_ptr);
        printf("in <device_Execute>  transaction_tid:%d  key:%d  contain:%d    kv_ptr:%p\n",transction_ptr->Tid,key.k,contain,src_kv_ptr);

        if(!contain){
            continue;    
        }
        storage_kv_ptr->copy(src_kv_ptr);
        
        if(update){
            // printf("update.\n");
            // //写操作，保存到storage中
            storage_kv_ptr->value.device_generate(devState);

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
    printf("<TRANSCTION_MANAGER INFO>:   in <device_Execute>  transaction_tid:%d  finished.\n",transction_ptr->Tid);
    return;
};


//在gpu上执行保留阶段  在事务的RWkey中记录读写信息
//    在执行阶段，已将kv信息读入了该事务的对应操作的storage中(访问数据不需要在从哈希表中查找)
template<class KeyType,class ValueType,int N>
__device__ void device_reserve(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr){
    //保留阶段

    //初始化rwkey 在执行阶段一起做， 
    //此处设置tid

    //read reservation
    for(int i=0;i<transction_ptr->read_key_nums;i++){
        RWKey* read_key_ptr = &(transction_ptr->read_key_list_head[i]);
        kv<KeyType,ValueType>* src_kv_ptr = read_key_ptr->kv_ptr;

        //设置这个readkey 的tid指针指向src元数据
        read_key_ptr->set_tid(&((src_kv_ptr->value).metadata));
        MetadataHelper::reserve_read(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
    }

    //write reservation
    for(int i=0;i<transction_ptr->write_key_nums;i++){
        RWKey* write_key_ptr = &(transction_ptr->write_key_list_head[i]);
        kv<KeyType,ValueType>* src_kv_ptr = write_key_ptr->kv_ptr;

        //设置这个writekey 的tid指针指向src元数据
        write_key_ptr->set_tid(&((src_kv_ptr->value).metadata));
        MetadataHelper::reserve_write(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
    }

    printf("in <device_Reserve>  transaction_tid:%d  finished.\n",transction_ptr->Tid);
};

template<class KeyType,class ValueType,int N>
__global__ void kernel_execute(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* device_transction_ptr,curandState *devState,int transction_nums){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<transction_nums){
        //当前线程所要处理的事务指针
        Transction<N>* ptr = &device_transction_ptr[idx];
        //当前线程申请到的随机数生成器
        curandState *devState_now = &devState[idx];
        //申请执行时所用内存  
        //https://zhuanlan.zhihu.com/p/525597120

        ptr->storage_ptr =(Storage<N>*)malloc(sizeof(Storage<N>));
        ptr->read_key_list_head=(RWKey*)malloc(sizeof(RWKey)*N);
        ptr->write_key_list_head=(RWKey*)malloc(sizeof(RWKey)*N);
        ptr->read_key_nums=0;
        ptr->write_key_nums=0;

        device_execute(device_map_ptr,ptr,devState_now);
        device_reserve(device_map_ptr,ptr);
    }
    return;
}




template<int N>
__device__ void device_analyze_dependency(Transction<N>* transction_ptr){
    RWKey* ReadkeySet = transction_ptr->read_key_list_head;
    RWKey* WritekeySet = transction_ptr->write_key_list_head;
    
    //分析raw
    for(int i=0;i<transction_ptr->read_key_nums;i++){
        //读取这个readkey的metadata 里面有epoch和rts信息
        uint64_t metadata = *(ReadkeySet[i].tid);

        uint16_t epoch = MetadataHelper::get_epoch(metadata);
        uint16_t wts   = MetadataHelper::get_wts(metadata);

        if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
            printf("in func<device_analyze_dependency>  Transction:%d has raw dependency on key %d  is been Transction:%d write reserved.\n",transction_ptr->Tid,(ReadkeySet[i].key_ptr)->k,wts);
            transction_ptr->raw = true;
            //break;
        };
    };

    //分析waw 和 war
    for(int i=0;i<transction_ptr->write_key_nums;i++){
        uint64_t metadata = *(WritekeySet[i].tid);

        uint16_t epoch = MetadataHelper::get_epoch(metadata);
        uint16_t wts   = MetadataHelper::get_wts(metadata);
        uint16_t rts   = MetadataHelper::get_rts(metadata);

        if(epoch==transction_ptr->epoch && rts<transction_ptr->Tid && rts!=0){
            printf("in func<device_analyze_dependency>  Transction:%d has war dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[i].key_ptr)->k,rts);
            transction_ptr->war = true;
        }

        if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
            printf("in func<device_analyze_dependency>  Transction:%d has waw dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[i].key_ptr)->k,wts);
            transction_ptr->waw = true;
        }
    }
    return;
}




template<int N>
__global__ void kernel_commit(Transction<N>* device_transction_ptr,int transction_nums){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<transction_nums){
        Transction<N>* ptr = &device_transction_ptr[idx];
        device_analyze_dependency(ptr);
    };
    return;
};


template<int N>
__device__ void device_install_with_reorder_optmization(Transction<N>* transction_ptr){
    if(transction_ptr->waw == true){
        transction_ptr->state=TRANSCTION_STATE::ABORT;
        printf("transction:%d abort state%d.\n",transction_ptr->Tid,transction_ptr->state);
        return;
    }else if(transction_ptr->raw == false ||transction_ptr->war ==false){
        for(int i=0;i<transction_ptr->operation_numbers;i++){
            if(!transction_ptr->update[i]){
                //读操作 不需要写回
                continue;
            }
            auto storage_ptr = &(transction_ptr->storage_ptr->_kvList[i]);
            auto src_ptr = (transction_ptr->read_key_list_head[i].kv_ptr);

            src_ptr->copy(storage_ptr);
        }
        printf("transction:%d successful install!\n",transction_ptr->Tid);
    }else{
        transction_ptr->state=TRANSCTION_STATE::ABORT;
        printf("transction:%d abort state%d.\n",transction_ptr->Tid,transction_ptr->state);
        return;
    }    
    return;
}

template<int N>
__device__ void device_install_without_reorder_optmization(Transction<N>* transction_ptr){
    if(transction_ptr->waw == true || transction_ptr->raw == true){
        transction_ptr->state=TRANSCTION_STATE::ABORT;
        printf("transction:%d abort state%d.\n",transction_ptr->Tid,transction_ptr->state);
        return;
    }else{
        for(int i=0;i<transction_ptr->operation_numbers;i++){
            if(!transction_ptr->update[i]){
                //读操作 不需要写回
                continue;
            }
            auto storage_ptr = &(transction_ptr->storage_ptr->_kvList[i]);
            auto src_ptr = (transction_ptr->read_key_list_head[i]).kv_ptr;

            src_ptr->copy(storage_ptr);
        }
        printf("transction:%d successful install!\n",transction_ptr->Tid);
    }
    return;
}


template<int N>
__global__ void kernel_install_with_reorder_optmization(Transction<N>* device_transction_ptr,int transction_nums){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<transction_nums){
        Transction<N>* ptr = &device_transction_ptr[idx];
        device_install_with_reorder_optmization(ptr);
    };
    return;
};

template<int N>
__global__ void kernel_install_without_reorder_optmization(Transction<N>* device_transction_ptr,int transction_nums){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<transction_nums){
        Transction<N>* ptr = &device_transction_ptr[idx];
        device_install_without_reorder_optmization(ptr);
    };
    return;
};



template<int N>
__global__ void kernel_collect(Transction<N>* device_transction_ptr,int transction_nums){
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx<transction_nums){
        //当前线程所要处理的事务指针
        Transction<N>* ptr = &device_transction_ptr[idx];
                
        //释放执行时所用内存  
        free(ptr->storage_ptr);
        free(ptr->read_key_list_head);
        free(ptr->write_key_list_head);

        ptr->storage_ptr=nullptr;
        ptr->read_key_list_head=nullptr;
        ptr->write_key_list_head=nullptr;

        ptr->read_key_nums=0;
        ptr->write_key_nums=0;
    }
    return;
}

};


#endif