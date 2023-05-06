
#ifndef YCSB_TRANSCTION_H
#define YCSB_TRANSCTION_H

#include "gpuallocator.cuh"
#include "Transction.h"
#include "storage_y.h"
#include "random.h"

namespace ycsb{
    template<int N>
    class Transction :public Transction_Base{
        public:
            Transction()=default;
            ~Transction()=default;

            void generate(Transction_Context* context){
                operation_numbers = M_Random::random_unsigned(0,context->max_operations_numbers);
                this->epoch=context->epoch_now;
                for(int i=0;i<operation_numbers;i++){
                    key[i]=M_Random::random_unsigned(1,context->keys_max);
                    update[i]=M_Random::random_bool(context->WR_rate);
                }
                state=TRANSCTION_STATE::READY;
                return;
            };

            void reset(uint16_t tid){
                Tid=tid;
                raw=false;
                waw=false;
                war=false;
                
                state = TRANSCTION_STATE::READY;
            };

            void init(uint16_t tid){
                Tid=tid;
                raw=false;
                waw=false;
                war=false;

                storage_ptr=nullptr;
                read_key_list_head=nullptr;
                write_key_list_head=nullptr;
                read_key_nums=0;
                write_key_nums=0;
                state = TRANSCTION_STATE::UNKNOWN;
            };  
        public:
            /*base info*/
            Key key[N];
            bool update[N];
            uint16_t epoch;
            uint16_t Tid;
            int operation_numbers;
            bool raw,waw,war;

            /*execute info*/
            //for cache 
            Storage<N>* storage_ptr;

            //for build read/write set
            RWKey* read_key_list_head;
            int read_key_nums;
            RWKey* write_key_list_head;
            int write_key_nums;
    };



};





#endif