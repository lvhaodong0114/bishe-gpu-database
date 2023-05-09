

#ifndef TRANSCTIONMANAGER_CUH
#define TRANSCTIONMANAGER_CUH

#include "gpuallocator.cuh"
#include "cudarand.cuh"
#include "Transction_y.h"
#include "TrancM_func.cuh"
#include "operation_parallel.cuh"


namespace ycsb{

    template<int N=200>
    class Transction_Manager{
        public:
            Gpu_Allocator* allocator_ptr;
            Transction<N>* host_transction_ptr;
            Transction<N>* device_transction_ptr;
            Transction_Context context;
            int transction_nums;
        public:
            Transction_Manager(){
                this->init_manager();
            };
            ~Transction_Manager(){
                if(host_transction_ptr){
                    delete []host_transction_ptr;
                }
                if(device_transction_ptr){
                    allocator_ptr->_cudaFree(device_transction_ptr);
                }
            };

            void set_context(Transction_Context context){
                this->context = context;
                M_Random::Init(context.random_seed);
                return;
            };

            void set_allocator_ptr(Gpu_Allocator* allocator_ptr){
                if(!allocator_ptr){
                    printf("<TRANSCTION_MANAGER INFO>:   transction manager's alloctor is not nullptr.\n");
                    return;
                }
                this->allocator_ptr = allocator_ptr;
                return;
            };

            void init_manager(){
                allocator_ptr=nullptr;
                host_transction_ptr=nullptr;
                device_transction_ptr=nullptr;
                transction_nums=0;
                return;
            };

            void generate(int nums){
                if(context.max_operations_numbers>N){
                    printf("<TRANSCTION_MANAGER INFO>:   max_operations_numbers %d is larger than Manager's capcity %d.\n",context.max_operations_numbers,N);
                    return;
                };
                if(!host_transction_ptr){
                    host_transction_ptr = new Transction<N>[N];
                    transction_nums=0;
                    //printf("host_transction_ptr %p  N:%d   bytes:%d  ptr%d.\n",host_transction_ptr,N,sizeof(Transction<N>) * N,sizeof(host_transction_ptr));
                }
                if(!device_transction_ptr && allocator_ptr){
                    allocator_ptr->_cudaMalloc((void**)&device_transction_ptr,sizeof(Transction<N>) * N );
                    cudaMemcpy(device_transction_ptr,host_transction_ptr,sizeof(Transction<N>) * N,cudaMemcpyHostToDevice);
                }
                printf("<TRANSCTION_MANAGER INFO>:   now have %d transctions.\n",transction_nums);

                int i=0;
                for(;transction_nums<N && i<nums;i++){
                    host_transction_ptr[transction_nums].init(transction_nums+1);
                    host_transction_ptr[transction_nums].generate(&context);
                    transction_nums++;
                }

                cudaMemcpy(device_transction_ptr,host_transction_ptr,sizeof(Transction<N>) * N,cudaMemcpyHostToDevice);
                printf("<TRANSCTION_MANAGER INFO>:   generate %d tranctions.\n",i);
                return;
            };

            void show_transction_now(){
                for(int i=0;i<transction_nums;i++){
                    printf("<TRANSCTION_MANAGER INFO>:   transction:%d   has operation:%d\n",host_transction_ptr[i].Tid,host_transction_ptr[i].operation_numbers);
                    for(int j=0;j<host_transction_ptr[i].operation_numbers;j++){
                        printf("op:%d      update:%d\n",host_transction_ptr[i].key[j].k,host_transction_ptr[i].update[j]);
                    };
                }
                return;
            };

            void serch_all_keys(int mode,int keynum,HashTable<Key,Value>* map_ptr){
                if(mode==0){
                    map_ptr->show_all_table();
                }else if(mode==1){
                    kernel_show_all_table<<<1,1>>>(map_ptr);
                    cudaDeviceSynchronize();
                }
                return;
            };

            void show_all_keys_on_device(HashTable<Key,Value>* map_ptr,HashTable<Key,Value>* device_map_ptr){
                int blocknum=(map_ptr->Size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
                kernel_show_table<<<blocknum,THREADS_PER_BLOCK>>>(device_map_ptr);
                cudaDeviceSynchronize();

                return;
            }

            void Execute(HashTable<Key,Value>* device_map_ptr){
                static bool set=false;
                if(set==false){
                    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
                    set=true;
                }

                int blocknum=(transction_nums+context.thread_per_block-1)/context.thread_per_block;
                //申请device端的random state
                curandState *devState=nullptr;
                devState = malloc_curand_state(allocator_ptr,blocknum*context.thread_per_block);
                setup_rand_kernel<<<1,blocknum*context.thread_per_block>>>(devState,context.random_seed);

                #ifndef OP_PARALLEL
                kernel_execute<<<blocknum,context.thread_per_block>>>(device_map_ptr,device_transction_ptr,devState,transction_nums);
                #else
                opreation_parallel::kernel_execute<<<blocknum,context.thread_per_block>>>(device_map_ptr,device_transction_ptr,devState,transction_nums);
                #endif
                cudaDeviceSynchronize();
                CUDACHECKERROR();

                free_curand_state(allocator_ptr,devState);
                cudaDeviceSynchronize();
                CUDACHECKERROR();
                return;
            };

            void Commit(){
                CUDACHECKERROR();

                int blocknum=(transction_nums+context.thread_per_block-1)/context.thread_per_block;
                
                #ifndef OP_PARALLEL
                kernel_commit<<<blocknum,context.thread_per_block>>>(device_transction_ptr,transction_nums);
                #else
                opreation_parallel::kernel_analyze_dependency<<<blocknum,context.thread_per_block>>>(device_transction_ptr,transction_nums);
                #endif

                cudaDeviceSynchronize();
                CUDACHECKERROR();
            };

            void Install(HashTable<Key,Value>* device_map_ptr){
                CUDACHECKERROR();
                int blocknum;
                
                #ifdef OP_PARALLEL
                blocknum=(transction_nums+context.thread_per_block-1)/context.thread_per_block;
                
                //申请device端的random state
                curandState *devState=nullptr;
                devState = malloc_curand_state(allocator_ptr,blocknum*context.thread_per_block);
                setup_rand_kernel<<<1,blocknum*context.thread_per_block>>>(devState,context.random_seed);
                #endif

                blocknum=(transction_nums+context.thread_per_block-1)/context.thread_per_block;     
                if(context.reorder_optmization){
                    kernel_install_with_reorder_optmization<<<blocknum,context.thread_per_block>>>(device_transction_ptr,transction_nums);
                }else{
                    #ifndef OP_PARALLEL
                    kernel_install_without_reorder_optmization<<<blocknum,context.thread_per_block>>>(device_transction_ptr,transction_nums);
                    #else
                    opreation_parallel::_kernel_install_without_reorder_optmization<<<blocknum,context.thread_per_block>>>(device_map_ptr,device_transction_ptr,transction_nums,devState);
                    #endif
                }
                cudaDeviceSynchronize();
                CUDACHECKERROR();

                #ifdef OP_PARALLEL
                free_curand_state(allocator_ptr,devState);
                cudaDeviceSynchronize();
                CUDACHECKERROR();
                #endif


                return;
            };

            void Collect(){
                context.epoch_now++;
                context.random_seed++;

                int blocknum=(transction_nums+context.thread_per_block-1)/context.thread_per_block;
                kernel_collect<<<blocknum,context.thread_per_block>>>(device_transction_ptr,transction_nums);
                cudaDeviceSynchronize();
                CUDACHECKERROR();

                cudaMemcpy(host_transction_ptr,device_transction_ptr,sizeof(Transction<N>) * N,cudaMemcpyDeviceToHost);

                uint16_t new_pos=0;
                for(int i=0;i<transction_nums;i++){
                    if(host_transction_ptr[i].state == TRANSCTION_STATE::ABORT){
                        memcpy(host_transction_ptr+new_pos,host_transction_ptr+i,sizeof(Transction<N>));
                        host_transction_ptr[new_pos].reset(new_pos+1);
                        new_pos++;
                    }
                }
                transction_nums=new_pos;
            };
    };

};

#endif