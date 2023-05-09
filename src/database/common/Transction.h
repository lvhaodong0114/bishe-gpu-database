#ifndef TRANSCTION_H
#define TRANSCTION_H


enum class TRANSCTION_STATE{
    UNKNOWN = 0,
    READY = 1,
    EXECUTE = 2,
    COMMIT = 3,
    ABORT = 4,
    WRONG = 5
};

class Transction_Context{
    public:
        double WR_rate;
        int max_operations_numbers;
        int random_seed;
        int keys_max;
        bool reorder_optmization;
        uint16_t epoch_now;
        uint16_t epoch_max;
        int thread_per_block;
    public:
        Transction_Context(){
            this->WR_rate = 0.4;
            this->max_operations_numbers = 30;
            this->random_seed = 0;
            this->keys_max = 20000;
            this->reorder_optmization = false;
            this->epoch_now = 0;
            this->epoch_max = 5;
            this->thread_per_block =16;
        };
};

class Transction_Base{
    public:
        virtual void generate(Transction_Context* context)=0;
        virtual void reset(uint16_t tid)=0;
        virtual void init(uint16_t tid)=0;
    public:
        TRANSCTION_STATE state;
};




#endif