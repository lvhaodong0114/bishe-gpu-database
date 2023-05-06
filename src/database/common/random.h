#ifndef RANDOM_H
#define RANDOM_H


#include <random>
#include <ctime>

class M_Random{

    public:
        static std::default_random_engine e;
        static bool is_init;
    public:
        
        static void Init(int seed){
            if(!is_init){
                is_init=true;
                e.seed(seed);
                printf("init random.\n");
            }
        };
        
        static unsigned random_unsigned(int min,int max){
            std::uniform_int_distribution<unsigned> u(min,max);
            return u(e);
        };

        static bool random_bool(float rate){
            std::uniform_real_distribution<float> u(0,1);
            return u(e)<rate;
        };

        static bool random_cstr(char* cstr,int size){
            for(int i=0;i<size;i++){
                char tmp = random_unsigned(33,126);
                cstr[i]=tmp;
            }
            return true;
        };

};


#endif