#ifndef    TIME_H
#define    TIME_H

#include <chrono>
#include <iostream>


class Timer{
    public:

    static void start_timer(){
        start_t = std::chrono::high_resolution_clock::now();
        start = true;
    };

    static void end_timer(){
        end_t =  std::chrono::high_resolution_clock::now();
        end =  true;
    };

    static void show_during(){
        if(start && end){
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t);
            std::cout << "execution time:" << duration.count()/1000 << "ms" << std::endl; 
            start = false;
            end =false;
        }
    };

    public:
    static std::chrono::time_point<std::chrono::high_resolution_clock> start_t;
    static bool start;
    static std::chrono::time_point<std::chrono::high_resolution_clock> end_t;
    static bool end;

};


#endif