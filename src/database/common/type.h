#ifndef TYPE_H
#define TYPE_H

#include <iostream>
#include "stdint.h"
namespace GPU_DATABASE{


enum TypeId{
    INVALID=0,
    BOOLEAN,
    INTEGER,
    DECIMAL,
    VARCHAR,
    TIMESTAMP
};


class Value;

//some type may be not comparable
enum class CmpBool{
    CmpFalse = 0,
    CmpTrue =1,
    CmuNull =2
};

class Type{
    public:
        Type(TypeId type_id):_type_id(type_id){};
        virtual ~Type() = default;

        //get the size of this data type in bytes
        static uint64_t GetTypeSize(const TypeId type_id);

        static const char* TypeToCharArray(const TypeId type_id);

        //return null value in this type
        static Value Null(const TypeId type_id);

        TypeId GetTypeId() {return _type_id;};

        static Type* GetInstance(const TypeId type_id) {return k_types[type_id];};



    private:
        TypeId _type_id;
        static Type* k_types[20];
};

uint64_t Type::GetTypeSize(const TypeId type_id){
    switch (type_id){
    case TypeId::BOOLEAN:
        //
    case TypeId::INTEGER:
        return 4;
    case TypeId::DECIMAL:
        //
    case TypeId::VARCHAR:
        //use GetLength
        return 0;
    case TypeId::TIMESTAMP:
        return 8;
    default:
        break;
    }
    printf("GetTypeSize::Unknown type");
};

const char* Type::TypeToCharArray(const TypeId type_id){
    switch (type_id){
    case TypeId::INVALID:
        return "INVALID";
    case TypeId::BOOLEAN:
        return "BOLLEAN";
    case TypeId::INTEGER:
        return "INTEGER";
    case TypeId::DECIMAL:
        return "DECIMAL";
    case TypeId::VARCHAR:
        return "VARCHAR";
    case TypeId::TIMESTAMP:
        return "TIMESTAMP";
    default:
        return "INVALID";
    }
};























}










#endif