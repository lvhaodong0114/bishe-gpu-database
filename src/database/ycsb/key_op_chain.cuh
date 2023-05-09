#ifndef KEY_OP_CHAIN_CUH
#define KEY_OP_CHAIN_CUH

namespace koc{
    __host__ __device__ int hash(int k){
        return k%100;
    };
};

namespace ycsb{
    class op_list{
    public:
        op_list* next;
        int op;
    
        __host__ __device__ op_list(){
            next=nullptr;
            op=-1;
        };
    
        __host__ __device__ void init(){
            next=nullptr;
            op=-1;
        };
    
        __device__ void insert(int op){
            op_list* p =this;
            ////printf("%p %p\n",p,next);
            for(;p->op<op;){
                op_list* next = p->next;
                ////printf("%p %p\n",p,next);
                if(next!=nullptr){
                    p=next;
                    continue;
                }
                ////printf("insert%d\n",op);
    
                op_list* tmp_ptr = (op_list*)malloc(sizeof(op_list)); 
                tmp_ptr->init();
    
                tmp_ptr->next=nullptr;
                tmp_ptr->op=op;
                p->next=tmp_ptr;
                return;
            }
        };
    };
    
    class __node{
    public:
        __node* next;
        int k;
        op_list* oplist_head;
    
        __device__ __node(){
            ////printf("init __node\n");
            next=nullptr;
            k=-1;
            oplist_head=nullptr;
        };
    
    
        __device__ void init(){
            // ////printf("init __node\n");
            next=nullptr;
            k=-1;
            oplist_head=nullptr;
        };
        
    };
    
    template<int N=20>
    class Key_Op_Chain{
    public:
        __node* node_ptr_chain[N];
    
         __device__ Key_Op_Chain(){
            for(int i=0;i<N;i++){
                node_ptr_chain[i]=(__node*)malloc(sizeof(__node));
                node_ptr_chain[i]->init();
            }
        };
    
        __device__ void init(){
            for(int i=0;i<N;i++){
                node_ptr_chain[i]=(__node*)malloc(sizeof(__node));
                node_ptr_chain[i]->init();
            }
        };

        __device__ void insert(int k,int op){
            int pos=koc::hash(k)%N;
            __node* node_ptr=node_ptr_chain[pos];
            __node* old_ptr = nullptr;
    
            ////printf("in insert node pos%d k%d  key %d op %d\n",pos,node_ptr_chain[pos]->k,k,op);
    
            for(;;){
                if(node_ptr==nullptr){
                    ////printf("in nullptr82 node k%p  key %d op %d\n",old_ptr,k,op);
                    __node* tmp_ptr = (__node*)malloc(sizeof(__node));
                    tmp_ptr->next=nullptr;
                    tmp_ptr->k=k;   
                    ////printf("in nullptr86\n");
                    op_list* tmp_o_ptr = (op_list*)malloc(sizeof(op_list));
                    tmp_o_ptr->init();
                    tmp_ptr->oplist_head=tmp_o_ptr;
    
                    tmp_ptr->oplist_head->insert(op);
                    ////printf("in nullptr88\n");
                    old_ptr->next=tmp_ptr;
                    ////printf("in nullptr04\n");
                    return;
                }else if(node_ptr->k==-1){
                    ////printf("in node_ptr->k==-1\n");
                    node_ptr->k=k;
    
                    op_list* tmp_o_ptr = (op_list*)malloc(sizeof(op_list));
                    tmp_o_ptr->init();
                    node_ptr->oplist_head=tmp_o_ptr;
                    node_ptr->oplist_head->insert(op);
                    return;
                }else if(node_ptr->k==k){
                    ////printf("in node_ptr->k==k\n");
                    node_ptr->oplist_head->insert(op);
                    return;
                }else{
                    ////printf("in else node k%d  key %d op %d\n",node_ptr->k,k,op);
                    old_ptr=node_ptr;
                    node_ptr=node_ptr->next;
                }
            }
            return;
        };
    
        __device__ void show(int tid){
            for(int i=0;i<N;i++){
                __node* node_ptr=node_ptr_chain[i];
                if(node_ptr->k!=-1){
                    for(;node_ptr;){
                        printf("key :%d  tid :%d\n",node_ptr->k,tid);
                        for(op_list* p=node_ptr->oplist_head;p!=nullptr;p=p->next){
                            // if(p->op!=-1){
                                printf("op:%d tid :%d\n",p->op,tid);
                            // }
                        }
                        printf("\n");
                        node_ptr=node_ptr->next;
                    }
                }
            }
        };
    
        template<int t_n,typename HASHTABLE>
        __device__ void exec(Transction<t_n>* transction_ptr,curandState *devState,HASHTABLE* hashtable_ptr){
            //printf("chain_ptr %p\n",this);
            chain_exec<<<1,N>>>(this,transction_ptr,devState);
        };
    };
    

    //带有差错检测的预执行
    //将执行结果存入缓存中 并检测事务的正确性
    template<int Chain_N,int N,typename HASHTABLE>
    __global__ void chain_exec(Key_Op_Chain<Chain_N>* chain_ptr,Transction<N>* transction_ptr,curandState *devState,HASHTABLE* hashtable_ptr){
        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx<N){
            __node* node_ptr = (chain_ptr->node_ptr_chain)[idx];
            //printf("tid:%d  chain:%d %p\n",transction_ptr->Tid,idx,node_ptr);

            if(node_ptr->k!=-1){
                for(;node_ptr;node_ptr=node_ptr->next){
                    int first_op = node_ptr->oplist_head->next->op;
                    int previous_op = first_op;

                    RWKey* _rwkey_ptr =  &(transction_ptr->read_key_list_head[first_op]);
                    auto src_kv_ptr = _rwkey_ptr->kv_ptr;
                    auto storage_kv_ptr = &((transction_ptr->storage_ptr)->_kvList[first_op]);
                    storage_kv_ptr->copy(src_kv_ptr);

                    //第一个操作时，该键是否存在
                    bool is_delete = hashtable_ptr->is_delete_flag[src_kv_ptr-hashtable_ptr->TablePtr];
                    if(is_delete){
                        transction_ptr->state = TRANSCTION_STATE::WRONG;

                        // //for test
                        // auto tmp_ptr = src_kv_ptr;
                        // hashtable_ptr->contain(node_ptr->k,&tmp_ptr);
                        // int tt;
                        // if(hashtable_ptr->is_delete_flag[tmp_ptr-hashtable_ptr->TablePtr]){
                        //     tt = 1;
                        // }else{
                        //     tt = 0;
                        // }
                        // printf("key:%d  pos:%d  delete:%d is_delete:%d\n",node_ptr->k,tmp_ptr,tt,is_delete?1:0);
                        // src_kv_ptr = _rwkey_ptr->kv_ptr;

                        printf("first transction:%d is wrong. op:%d   key:%d\n",transction_ptr->Tid,first_op,node_ptr->k);
                        return;
                    }

                    for(op_list* p=node_ptr->oplist_head;p!=nullptr;p=p->next){
                        if(p->op>first_op){
                            int op = p->op;
                            bool update = transction_ptr->update[op];



                            if(update){
                                storage_kv_ptr->value.device_generate(devState);   
                                is_delete = transction_ptr->_delete[op];

                                //将所有写键的tid统一到哈希表中同一个位置
                                RWKey* _rwkey_ptr_now =  &(transction_ptr->write_key_list_head[op]);
                                _rwkey_ptr_now->set_tid(&((src_kv_ptr->value).metadata));
                                _rwkey_ptr_now->set_kv_ptr(src_kv_ptr);

                                
                            }else{
                                if(is_delete){
                                    transction_ptr->state = TRANSCTION_STATE::WRONG;
                                    printf("transction:%d is wrong. op:%d\n",transction_ptr->Tid,op);

                                    return;
                                }
                                //将所有读键的tid统一到哈希表中同一个位置
                                RWKey* _rwkey_ptr_now =  &(transction_ptr->read_key_list_head[op]);
                                _rwkey_ptr_now->set_tid(&((src_kv_ptr->value).metadata));
                                _rwkey_ptr_now->set_kv_ptr(src_kv_ptr);

                                //该操作的缓存
                                storage_kv_ptr = &((transction_ptr->storage_ptr)->_kvList[op]);
                                //该操作上次结果
                                auto previous_storage_kv_ptr =  &((transction_ptr->storage_ptr)->_kvList[previous_op]);
                                storage_kv_ptr->copy(previous_storage_kv_ptr);
                            }
                            previous_op = op;
                        }
                    }
                }

                // for(;node_ptr;node_ptr=node_ptr->next){
                //     int first_op = node_ptr->oplist_head->next->op;
                //     int previous_op = first_op;
                //     bool is_delete = false;
                //     for(op_list* p=node_ptr->oplist_head;p!=nullptr;p=p->next){
                //         if(p->op!=-1){
                //             int op=p->op;
                //             RWKey* _rwkey_ptr =  &(transction_ptr->read_key_list_head[first_op]);
                //             bool update = transction_ptr->update[op];

                            // //当前缓存位置
                            // auto storage_kv_ptr=&((transction_ptr->storage_ptr)->_kvList[op]);
                            // auto src_kv_ptr=_rwkey_ptr->kv_ptr;
        
                //             auto previous_storage_kv_ptr =&((transction_ptr->storage_ptr)->_kvList[previous_op]);
                //             storage_kv_ptr->copy(previous_storage_kv_ptr);
                //             //printf("transction id:%d chain:%d op:%d done.\n",transction_ptr->Tid,idx,op);


                //             if(update){
                //                 storage_kv_ptr->value.device_generate(devState);   
                //                 is_delete = transction_ptr->_delete[op];
                //                 // src_kv_ptr->copy(storage_kv_ptr);
                //                 // hashtable_ptr->is_delete_flag[src_kv_ptr-(hashtable_ptr->TablePtr)]=transction_ptr->_delete[op];
                //             }
                //             previous_op = op;
                //         }
                //     }

                    
                
                
                // }
            }
        }
    }
};


#endif