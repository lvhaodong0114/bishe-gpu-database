#ifndef DB_H
#define DB_H

#include <iostream>
#include "kv.cuh"
#include "database.h"
#include "table.cuh"
#include "TransctionManager.cuh"
#include "hashTable.cuh"

namespace ycsb{

class DB  :public database{
    public:
        DB()=default;
        ~DB(){
            if(transction_manager_ptr){
                delete transction_manager_ptr;
            };
            if(alloctor_ptr){
                delete alloctor_ptr;
            };
            if(table_vec[0][0]){
                delete table_vec[0][0];
            };
        }

        void init_db(){
            this->alloctor_ptr = new Gpu_Allocator(ALLOCATOR_DEFAULT_SIZE);

            this->table_vec.resize(1);
            ITable* table_ptr = new Table<Key,Value>(0,alloctor_ptr);
            this->table_vec[0].resize(1);
            this->table_vec[0][0]=table_ptr;

            this->transction_manager_ptr = new Transction_Manager<>;
            this->transction_manager_ptr->set_allocator_ptr(this->alloctor_ptr);

            this->device_map_ptr=nullptr;
        };

        void init_table(int table_id,int mode,int item_nums){
            ITable* table_ptr = this->table_vec[0][table_id];
            if(mode == 0){
                for(int i=1;i<item_nums;i++){
                    Key k(i);
                    Value v;
                    v.generate();
                    table_ptr->insert((void*)&k,(void*)&v);
                    // if(i%10==0){
                    //     table_ptr->delete_((void*)&k,(void*)&v);
                    // }
                }
            }
            printf("<DB INFO>:                   init successful!\n");
        };

        void set_transction_generate_config(Transction_Context cfg){
            transction_manager_ptr->set_context(cfg);
            return;
        };

        void generate_transction(int nums){
            if(!transction_manager_ptr){
                printf("<DB INFO>:                   transction manager is not init.\n");
                return;
            };
            transction_manager_ptr->generate(nums);
        };

        void transction_manager_show(){
            transction_manager_ptr->show_transction_now();
            return;
        };

        void get_kvs_num(int table_id){
            ITable* table_ptr = this->table_vec[0][table_id];
            printf("<DB INFO>:                   table %d has %d kvs  %d bytes,\n",table_id,table_ptr->get_size(),table_ptr->get_size()*sizeof(kv<Key,Value>));
        };

        void get_map_ptr(){
            map_ptr = (HashTable<Key,Value>*)this->table_vec[0][0]->get_map_ptr();
            map_ptr->move_to_device();
            alloctor_ptr->_cudaMalloc((void**)&device_map_ptr,sizeof(HashTable<Key,Value>));
            cudaMemcpy(device_map_ptr, map_ptr, sizeof(HashTable<Key,Value>), cudaMemcpyHostToDevice);
        };

        void free_map_ptr(){
            map_ptr->move_to_host();
            alloctor_ptr->_cudaFree(device_map_ptr);

            map_ptr=nullptr;
            device_map_ptr=nullptr;
        };

        void search_all(int mode,int keynum){
            if(mode==0){
                map_ptr = (HashTable<Key,Value>*)this->table_vec[0][0]->get_map_ptr();
                transction_manager_ptr->serch_all_keys(mode,keynum,map_ptr);
            }else if(mode==1){
                get_map_ptr();
                transction_manager_ptr->serch_all_keys(mode,keynum,device_map_ptr);
                free_map_ptr();
            };
        };

        void test(){
            while(transction_manager_ptr->transction_nums!=0){
                printf("<DB INFO>:                   remaining %d transctions ,start %d.\n",transction_manager_ptr->transction_nums,transction_manager_ptr->context.epoch_now);
                test_one_epoch();
            };
        }

        void test_one_epoch(){
            if(device_map_ptr==nullptr){
                get_map_ptr();
                printf("<DB INFO>:                   successful get map ptr.\n");
                // transction_manager_ptr->show_all_keys_on_device(map_ptr,device_map_ptr);                
                transction_manager_ptr->Execute(device_map_ptr);
                transction_manager_ptr->Commit();
                transction_manager_ptr->Install(device_map_ptr);
                transction_manager_ptr->Collect();
                free_map_ptr();
                return;
            }
            printf("<DB INFO>:                   device_map_ptr is not nullptr.\n");
            return;
        };

    private:
        Transction_Manager<>* transction_manager_ptr;
        HashTable<Key,Value>* device_map_ptr;
        HashTable<Key,Value>* map_ptr;


};




};




#endif