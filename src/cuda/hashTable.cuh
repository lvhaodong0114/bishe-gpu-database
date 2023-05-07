#ifndef HASHTABLE_CUH
#define HASHTABLE_CUH

#include "kv.cuh"
#include "stdint.h"
#include "gpuallocator.cuh"
#include "test.cuh"
#include "hashTable_gpuFunc.cuh"



#define	KEY_INVALID		0
#define DEAULT_TABLE_SIZE 10000


/* Load factors are declared as integers to avoid floating point operations */
#define MIN_LOAD_FACTOR 65
#define MAX_LOAD_FACTOR 80

/**
 * Function hashKey
 * Returns hash of "key"
 */
 __device__ __host__ uint32_t hashKey(uint32_t key)
 {
     /* both a and b are prime numbers */
     return (32069ULL * key) % 694847539;
 };

enum class HASHTABLE_STATE{
    UNKNOWN = 0,
    ON_DEVICE = 1,
    ON_HOST = 2
};

template <typename KeyType,typename ValueType>
class HashTable{
    using KV = kv<KeyType,ValueType>;

    public:
        HashTable(Gpu_Allocator* allocatorptr,int size=DEAULT_TABLE_SIZE):Size(size),TablePtr(nullptr),is_delete_flag(nullptr),ItemNums(0),AllocatorPtr(allocatorptr),state(HASHTABLE_STATE::UNKNOWN){};
        HashTable() = default;
        ~HashTable() {
            if(TablePtr){
                if(state==HASHTABLE_STATE::ON_HOST){
                    free(TablePtr);
                    free(is_delete_flag);                    
                }
                else if(state==HASHTABLE_STATE::ON_DEVICE){
                    AllocatorPtr->_cudaFree(TablePtr);
                    AllocatorPtr->_cudaFree(is_delete_flag);
                }
            };
        };

        bool set_allocatorptr(Gpu_Allocator* allocatorptr){
            if(!this->AllocatorPtr){
                this->AllocatorPtr = allocatorptr;
                return true;
            }else{
                return false;
            }
        };

        void init_device(uint32_t size){
            if(size>0){
                Size=size;
                cudaError_t err;
                err = AllocatorPtr->_cudaMalloc((void**)&TablePtr,sizeof(KV)*size);
                err = AllocatorPtr->_cudaMalloc((void**)&is_delete_flag,sizeof(bool)*size);
                printf("Malloc successfull.\n");

                cudaCheckError(err);
        
                /* set everything to KEY_INVALID */
                err =cudaMemset(TablePtr, KEY_INVALID, sizeof(KV)*size);
                err =cudaMemset(is_delete_flag, 0, sizeof(bool)*size);
                printf("Malloc successfull.\n");
                state = HASHTABLE_STATE::ON_DEVICE;
            }else{
                printf("size is <= zero.\n");
            }
        };

        void init_host(uint32_t size){
            if(size>0){
                Size=size;
                TablePtr=(KV*)malloc(sizeof(KV)*size);
                is_delete_flag  = (bool*)malloc(sizeof(bool)*size);

                memset(TablePtr,KEY_INVALID,sizeof(KV)*size);
                memset(is_delete_flag,0,sizeof(bool)*size);
                printf("Malloc successfull.\n");
                state = HASHTABLE_STATE::ON_HOST;
            }else{
                printf("size is <= zero.\n");
            }
        };


        //check if key is in table,need a kv* to recive result pointer 
        __host__ __device__ bool contain(KeyType key, KV** ptr=nullptr){
            uint32_t hash=hashKey(key.k);
            // printf(" in func:contain. %d  %d\n",hash,key.k);
            for(uint32_t i = hash % Size;(TablePtr[i].getKey())->k!=KEY_INVALID;i=(i+1)%Size){
                // printf("pos %d  tablekey %d  key %d\n",i,(TablePtr[i].getKey())->k,key.k);
                if((TablePtr[i].getKey())->k == key.k){
                    // printf("<HASHTABLE__INFO>:            in func:contain. Table contain key %d\n",key.k);
                    if(ptr){
                        *ptr = &(TablePtr[i]);
                    }
                    //key相同  但已被删除
                    if(is_delete_flag[i]){
                        printf("key %d has been deleted.\n",key.k);
                        return false;
                    }
                    return true;
                };
            }
            // printf("<HASHTABLE__INFO>:           in func:contain. Table do not contain key %d  now table size is %d itemnums:%d\n",key.k,Size,ItemNums);
            return false;
        };

        __host__ __device__ bool _delete(KeyType key){
            KV* tmp_ptr;
            //查找待删除的键
            bool b = contain(key,&tmp_ptr);
            //存在未被删除的该键
            if(b){
                is_delete_flag[tmp_ptr-TablePtr]=1;
                ItemNums--;
                // printf("delete %d successful! pos:%d  key:%d\n",key.k,tmp_ptr-TablePtr,TablePtr[tmp_ptr-TablePtr].key.k);
                return true;
            }
            // printf("delete %d failed! pos:%d  \n",key.k,tmp_ptr-TablePtr);
            return false;
        };

        __host__ __device__ int get_load_factor(){
            return (ItemNums*100)/Size;
        }

        __host__ __device__ bool insert(KeyType key, KV* src_ptr){
            if(this->get_load_factor()<MAX_LOAD_FACTOR){
                uint32_t hash=hashKey(key.k);
                uint32_t i = hash % Size;
                //存有键值并且未被删除 则继续
                for(;((TablePtr[i].getKey())->k!=KEY_INVALID)&&(!is_delete_flag[i]);i=(i+1)%Size){
                };
                KV* des_ptr= &TablePtr[i];
                is_delete_flag[i]=false;
                des_ptr->copy(src_ptr);
                ItemNums++;
                //printf("insert key %d into hashtable pos %d successfull Size %d!\n",key.k,i,Size);
                return true;    
            }else if(state==HASHTABLE_STATE::ON_DEVICE){
                printf("the hashtable is ON_DEVICE\n");
                return false;
            }else if(state==HASHTABLE_STATE::ON_HOST){
                printf("<HASHTABLE__INFO>:           hashtable insert %d calls reshape!\n",key.k);
                uint32_t hash=hashKey(key.k);
                uint32_t i = hash % Size;
                for(;((TablePtr[i].getKey())->k!=KEY_INVALID)&&(!is_delete_flag[i]);i=(i+1)%Size){
                };
                KV* des_ptr= &TablePtr[i];
                des_ptr->copy(src_ptr);
                ItemNums++;
                // printf("insert key %d into hashtable pos %d successfull Size %d!\n",key.k,i,Size);

                int new_size = ItemNums*100/MIN_LOAD_FACTOR;
                printf("<HASHTABLE__INFO>:           hashtable reshape successfull. oldsize:%d new_Size %d!\n",Size,new_size);
                this->reshape_on_host(new_size);

                return true;
            }else{
                printf("<HASHTABLE__INFO>:           the hashtable state is UNKNOWN!\n");
                return false;
            }
        };


        __host__ __device__ int get_itemnums(){
            return ItemNums;
        };
        
        __host__ __device__ int get_size(){
            return Size;
        };

        __host__ __device__ void move_to_device(){
            if(state!=HASHTABLE_STATE::ON_HOST){
                printf("<HASHTABLE__INFO>:           in func move_to_device error  state is not ON_HOST.\n");
                return;
            }
            cudaError_t err;
            KV*  devtable_ptr;
            bool *dev_is_delete_flag;
            AllocatorPtr->_cudaMalloc((void**) &devtable_ptr,sizeof(KV)*Size);
            AllocatorPtr->_cudaMalloc((void**) &dev_is_delete_flag,sizeof(bool)*Size);
            err = cudaMemcpy(devtable_ptr, TablePtr, sizeof(KV)*Size, cudaMemcpyHostToDevice);
            err = cudaMemcpy(dev_is_delete_flag, is_delete_flag, sizeof(bool)*Size, cudaMemcpyHostToDevice);
            cudaCheckError(err);

            free(TablePtr);
            free(is_delete_flag);
            TablePtr=devtable_ptr;
            is_delete_flag=dev_is_delete_flag;
            state=HASHTABLE_STATE::ON_DEVICE;
            printf("<HASHTABLE__INFO>:           Successful move table to device. Size:%d\n",Size);
            return;
        };

        __host__ __device__ void move_to_host(){
            if(state!=HASHTABLE_STATE::ON_DEVICE){
                printf("<HASHTABLE__INFO>:           in func move_to_host error  state is not ON_DEVICE.\n");
                return;
            }
            cudaError_t err;
            KV*  hosttable_ptr;
            bool* host_is_delete_flag;
            hosttable_ptr =  (KV*)malloc(sizeof(KV)*Size);
            host_is_delete_flag = (bool*)malloc(sizeof(bool)*Size);
            err = cudaMemcpy(hosttable_ptr,TablePtr,sizeof(KV)*Size,cudaMemcpyDeviceToHost);
            err = cudaMemcpy(host_is_delete_flag,is_delete_flag,sizeof(bool)*Size,cudaMemcpyDeviceToHost);
            // cudaCheckError(err);
            AllocatorPtr->_cudaFree(TablePtr);
            AllocatorPtr->_cudaFree(is_delete_flag);

            TablePtr = hosttable_ptr;
            is_delete_flag = host_is_delete_flag;
            state=HASHTABLE_STATE::ON_HOST;

            printf("<HASHTABLE__INFO>:           successful move table to host. Size:%d \n",Size);
            return;
        };

        __host__ __device__ void show_all_table(){
            for(uint32_t i=0;i<Size;i++){
                printf("pos %d   key %d\n",i,(TablePtr[i].getKey())->k);
            }
        }

        __host__ __device__ void reshape_on_host(uint32_t new_size){
            if(state != HASHTABLE_STATE::ON_HOST){
                printf("<HASHTABLE__INFO>:           in func: reshape_on_host  TABLE IS NOT ON HOST!\n");
                return;
            }else{
                this->move_to_device();

                cudaError_t err;
                KV* new_table_ptr;
                bool* new_is_delete_flag;
                err = AllocatorPtr->_cudaMalloc((void**)&new_table_ptr,sizeof(KV)*new_size);
                err = AllocatorPtr->_cudaMalloc((void**)&new_is_delete_flag,sizeof(bool)*new_size);
                err =cudaMemset(new_table_ptr, KEY_INVALID, sizeof(KV)*new_size);
                err =cudaMemset(new_is_delete_flag, 0, sizeof(bool)*new_size);
                cudaCheckError(err);        
                
                uint32_t* insertCounter;
                AllocatorPtr->_cudaMalloc((void **)&insertCounter,sizeof(uint32_t));
                err=cudaMemset(insertCounter, 0, sizeof(uint32_t));
                cudaCheckError(err);

                int threadblocknums = (Size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
                kernel_Reinsert<<<threadblocknums,THREADS_PER_BLOCK>>>(new_table_ptr,new_size,TablePtr,is_delete_flag,Size,insertCounter);
                cudaDeviceSynchronize();

                uint32_t insertnum;
                cudaMemcpy(&insertnum,insertCounter,sizeof(uint32_t),cudaMemcpyDeviceToHost);
                ItemNums=insertnum;
                printf("<HASHTABLE__INFO>:           reinsert %d keys\n",insertnum);

                Size = new_size;
                err = AllocatorPtr->_cudaFree(TablePtr);
                err = AllocatorPtr->_cudaFree(is_delete_flag);
                cudaCheckError(err);      
                
                TablePtr=new_table_ptr;
                is_delete_flag=new_is_delete_flag;
                this->move_to_host();
                printf("<HASHTABLE__INFO>:           in func: reshape_on_host successful reshappe hashtable new size is %d\n",new_size);
            }
        };
                

    public:
        KV*  TablePtr;
        bool* is_delete_flag;
        Gpu_Allocator* AllocatorPtr;

        uint32_t ItemNums;
        uint32_t Size;
        HASHTABLE_STATE state;
    };



template <typename KeyType,typename ValueType>
__global__ void kernel_show_all_table(HashTable<KeyType,ValueType>* map_ptr){
    map_ptr->show_all_table();
};

#endif