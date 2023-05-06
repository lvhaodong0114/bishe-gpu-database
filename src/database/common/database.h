#ifndef  DATABASE_H
#define  DATABASE_H


#define ALLOCATOR_DEFAULT_SIZE (500*1024*1024)

#include "gpuallocator.cuh"
#include "table.cuh"
#include <vector>
class database{
    public:
        database() = default;
        virtual ~database() = default;

        virtual void init_db()=0;

        // virtual void generate()=0;

    public:
        std::vector<std::vector<ITable* >> table_vec;
        Gpu_Allocator* alloctor_ptr;
};





#endif