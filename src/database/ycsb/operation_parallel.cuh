#ifndef OPREATION_PARALLEL_CUH 
#define OPREATION_PARALLEL_CUH

#include "key_op_chain.cuh"
#include "hashTable_gpuFunc.cuh"

namespace ycsb{
namespace opreation_parallel{


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

            // malloc style
            // ptr->storage_ptr =(Storage<N>*)malloc(sizeof(Storage<N+1>));
            // ptr->read_key_list_head=(RWKey*)malloc(sizeof(RWKey)*(N+1));
            // ptr->write_key_list_head=(RWKey*)malloc(sizeof(RWKey)*(N+1));

            //new style
            ptr->storage_ptr = new Storage<N>;
            ptr->read_key_list_head=new RWKey[N];
            ptr->write_key_list_head=new RWKey[N];
            ptr->read_key_nums=0;
            ptr->write_key_nums=0;

            //预定初始化
            device_reserve(device_map_ptr,ptr,devState_now);
            //预执行+差错检测+统一RWKEY
            exec(device_map_ptr,ptr,devState);
            //todo 对无差错的事务进行预定
            do_reserve(ptr);

        }
        return;
    };


    //执行到缓存中并实装
    template<class KeyType,class ValueType,int N>
    __device__ void exec(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        Key_Op_Chain<32>* chain_ptr = (Key_Op_Chain<32>*)malloc(sizeof(Key_Op_Chain<32>));
        chain_ptr->init();
        // printf("exec %p\n",chain_ptr);
        for(int i=0;i<transction_ptr->operation_numbers;i++){
            chain_ptr->insert(transction_ptr->key[i].k,i);
        }
        // chain_ptr->show(transction_ptr->Tid);
        chain_exec<<<32,1>>>(chain_ptr,transction_ptr,devState,device_map_ptr);
        
        return;
    };
    

    //在gpu上执行 一个事务 所有rwkey的预定
    template<class KeyType,class ValueType,int N>
    __device__ void device_reserve(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        int opnumbers = transction_ptr->operation_numbers;
        //预定初始化
        kernel_operation_reserve<<<1,opnumbers>>>(device_map_ptr,transction_ptr,devState);
        cudaDeviceSynchronize();

        return;
    };


    template<class KeyType,class ValueType,int N>
    __global__ void kernel_operation_reserve(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        if(idx<transction_ptr->operation_numbers){
            KeyType key = transction_ptr->key[idx];
            bool update = transction_ptr->update[idx];
            kv<KeyType,ValueType>* storage_kv_ptr=&((transction_ptr->storage_ptr)->_kvList[idx]);
            kv<KeyType,ValueType>* src_kv_ptr=nullptr;

            bool contain = device_map_ptr->contain(key,&src_kv_ptr);
            printf("in <kernel_operation_reserve>  transaction_tid:%d  key:%d  contain:%d    kv_ptr:%p  \n",transction_ptr->Tid,key.k,contain,src_kv_ptr);

            if(!contain){
                //for test
                //先插入并标记已删除，但数据库中已有metadata 可供reserve阶段读取metadata，最后满足install的再实装入数据库

                KeyType _key = transction_ptr->key[idx];
                kv<KeyType,ValueType> kv;

                kv.key.copy(&_key);
                kv.value.device_generate(devState);

                insert_atomic(device_map_ptr,&kv,&src_kv_ptr);

                // device_map_ptr->insert(_key,&kv);

                // bool contain = device_map_ptr->contain(key,&src_kv_ptr);
                printf("!contain key:%d fake insert pos:%p\n",_key,src_kv_ptr);
            }

            // storage_kv_ptr->copy(src_kv_ptr);

            if(update){
                //预定初始化
                RWKey* _rwkey_ptr =  &(transction_ptr->read_key_list_head[idx]);
                transction_ptr->read_key_nums++;
                _rwkey_ptr->set_key_ptr((void*)&(transction_ptr->key[idx]));
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);

                // //预定
                // _rwkey_ptr->set_tid(&((src_kv_ptr->value).metadata));
                // MetadataHelper::reserve_read(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);

                // //初始化Wkey  直接将写键预定到操作对应的位置，可以防止同时写入错误  由于读写键在后续处理时不具备顺序，所以不必顺序访问
                _rwkey_ptr =  &(transction_ptr->write_key_list_head[idx]);
                _rwkey_ptr->set_key_ptr((void*)&(transction_ptr->key[idx]));
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);
                _rwkey_ptr->set_tid(&((src_kv_ptr->value).metadata));

                // MetadataHelper::reserve_write(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
            }else{
                RWKey* _rwkey_ptr =  &(transction_ptr->read_key_list_head[idx]);
                transction_ptr->read_key_nums++;
                _rwkey_ptr->set_key_ptr((void*)&(transction_ptr->key[idx]));
                _rwkey_ptr->set_kv_ptr((void*)src_kv_ptr);

                _rwkey_ptr->set_tid(&((src_kv_ptr->value).metadata));
                // MetadataHelper::reserve_read(&((src_kv_ptr->value).metadata), transction_ptr->epoch, transction_ptr->Tid);
            }
        
        }
    };



    template<int N>
    __device__ void _device_analyze_dependency(Transction<N>* transction_ptr){
        RWKey* ReadkeySet = transction_ptr->read_key_list_head;
        RWKey* WritekeySet = transction_ptr->write_key_list_head;
        
        //分析raw
        for(int i=0;i<transction_ptr->operation_numbers;i++){
            //读取这个readkey的metadata 里面有epoch和rts信息
            uint64_t metadata = *(ReadkeySet[i].tid);
    
            uint16_t epoch = MetadataHelper::get_epoch(metadata);
            uint16_t wts   = MetadataHelper::get_wts(metadata);
    
            // printf("in func<device_analyze_dependency>  Transction:%d has raw dependency on key %d  Transction:%d write reserved.\n",transction_ptr->Tid,(ReadkeySet[i].key_ptr)->k,wts);
            // printf("check tid:%d  opid:%d.\n",transction_ptr->Tid,i);

            if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
                printf("in func<device_analyze_dependency>  Transction:%d has raw dependency on key %d  is been Transction:%d write reserved.\n",transction_ptr->Tid,(ReadkeySet[i].key_ptr)->k,wts);
                transction_ptr->raw = true;
                break;
            };
        };
    
        //分析waw 和 war
        for(int i=0;transction_ptr->update[i]&&i<transction_ptr->operation_numbers;i++){
            // printf("in func<device_analyze_dependency>  Transction:%d op:%d.\n",transction_ptr->Tid,i);
            uint64_t metadata = *(WritekeySet[i].tid);
    
            uint16_t epoch = MetadataHelper::get_epoch(metadata);
            uint16_t wts   = MetadataHelper::get_wts(metadata);
            uint16_t rts   = MetadataHelper::get_rts(metadata);
    
            if(epoch==transction_ptr->epoch && rts<transction_ptr->Tid && rts!=0 && !transction_ptr->war){
                printf("in func<device_analyze_dependency>  Transction:%d has war dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[i].key_ptr)->k,rts);
                transction_ptr->war = true;
            }
    
            if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
                printf("in func<device_analyze_dependency>  Transction:%d has waw dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[i].key_ptr)->k,wts);
                transction_ptr->waw = true;
            }
            // printf("in func<device_analyze_dependency>  Transction:%d op:%dfinied.\n",transction_ptr->Tid,i);
        }
        return;
    }

    template<int N>
    __global__ void _device_analyze_dependency_operation_parallel(Transction<N>* transction_ptr){
        RWKey* ReadkeySet = transction_ptr->read_key_list_head;
        RWKey* WritekeySet = transction_ptr->write_key_list_head;

        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;


        if(transction_ptr->update[idx]){
            uint64_t metadata = *(WritekeySet[idx].tid);
            uint16_t epoch = MetadataHelper::get_epoch(metadata);
            uint16_t wts   = MetadataHelper::get_wts(metadata);
            uint16_t rts   = MetadataHelper::get_rts(metadata);
    
            // printf("tid:%d,idx%d\n",transction_ptr->Tid,idx);

            if(epoch==transction_ptr->epoch && rts<transction_ptr->Tid && rts!=0){
                printf("in func<device_analyze_dependency>  Transction:%d has war dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[idx].key_ptr)->k,rts);
                transction_ptr->war = true;
            }
    
            if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
                printf("in func<device_analyze_dependency>  Transction:%d has waw dependency on key %d  is been Transction:%d read reserved.\n",transction_ptr->Tid,(WritekeySet[idx].key_ptr)->k,wts);
                transction_ptr->waw = true;
            }
            return;

        }

        if(!transction_ptr->update[idx] && !transction_ptr->raw){

            // printf("tid:%d,idx%d,tid_ptr:%p\n",transction_ptr->Tid,idx,ReadkeySet[idx].tid);


            uint64_t metadata = *(ReadkeySet[idx].tid);
            uint16_t epoch = MetadataHelper::get_epoch(metadata);
            uint16_t wts   = MetadataHelper::get_wts(metadata);


            if(epoch==transction_ptr->epoch && wts<transction_ptr->Tid && wts!=0){
                printf("in func<device_analyze_dependency>  Transction:%d has raw dependency on key %d  is been Transction:%d write reserved.\n",transction_ptr->Tid,(ReadkeySet[idx].key_ptr)->k,wts);
                transction_ptr->raw = true;
            };

        }

        return;
    };

    template<int N>
    __device__ void do_reserve(Transction<N>* ptr){
        for(int i=0;i<ptr->operation_numbers;i++){
            if(ptr->update[i]){
                auto _rwkey_ptr = ptr->write_key_list_head[i];
                auto src_kv_ptr = _rwkey_ptr.kv_ptr;
                MetadataHelper::reserve_write(&((src_kv_ptr->value).metadata), ptr->epoch, ptr->Tid);
            }else{
                auto _rwkey_ptr = ptr->read_key_list_head[i];
                auto src_kv_ptr = _rwkey_ptr.kv_ptr;
                MetadataHelper::reserve_read(&((src_kv_ptr->value).metadata), ptr->epoch, ptr->Tid);
            };
        }
    }

    template<int N>
    __global__ void kernel_analyze_dependency(Transction<N>* device_transction_ptr,int transction_nums){
        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        if(idx<transction_nums){
            Transction<N>* ptr = &device_transction_ptr[idx];

            if(device_transction_ptr[idx].state!=TRANSCTION_STATE::WRONG){
                // //串行分析+提前中止
                // _device_analyze_dependency(ptr);
                // //并行分析

                int opnumbers = ptr->operation_numbers;
                _device_analyze_dependency_operation_parallel<<<1,opnumbers>>>(ptr);
                cudaDeviceSynchronize();
            }
        };
        return;
    };



    template<class KeyType,class ValueType,int N>
    __device__ void install(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        int operation_numbers=transction_ptr->operation_numbers;
        kernel_install<<<1,operation_numbers>>>(device_map_ptr,transction_ptr,devState);
    }

    template<class KeyType,class ValueType,int N>
    __global__ void kernel_install(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        bool update = transction_ptr->update[idx];
        if(update){
            auto storage_kv_ptr = &((transction_ptr->storage_ptr)->_kvList[idx]);
            auto rwkey_ptr_now =  &(transction_ptr->write_key_list_head[idx]);

            rwkey_ptr_now->kv_ptr->copy(storage_kv_ptr);
            device_map_ptr->is_delete_flag[ rwkey_ptr_now->kv_ptr - device_map_ptr->TablePtr] = transction_ptr->_delete[idx];
        }

    }

    template<class KeyType,class ValueType,int N>
    __device__ void _device_install_without_reorder_optmization(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,curandState *devState){
        if(transction_ptr->waw == true || transction_ptr->raw == true){
            transction_ptr->state=TRANSCTION_STATE::ABORT;
            printf("transction:%d abort state%d.\n",transction_ptr->Tid,transction_ptr->state);
            return;
        }else{
            install(device_map_ptr,transction_ptr,devState);
            printf("transction:%d successful install!\n",transction_ptr->Tid);
        }
        return;
    };


    template<class KeyType,class ValueType,int N>
    __global__ void _kernel_install_without_reorder_optmization(HashTable<KeyType,ValueType>* device_map_ptr,Transction<N>* transction_ptr,int transction_nums,curandState *devState){
        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        
        if(idx<transction_nums && transction_ptr[idx].state!=TRANSCTION_STATE::WRONG){
            Transction<N>* ptr = &transction_ptr[idx];
            _device_install_without_reorder_optmization(device_map_ptr,ptr,devState);
        };
        return;
    };




};
};



#endif