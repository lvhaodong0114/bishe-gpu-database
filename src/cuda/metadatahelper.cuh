
#ifndef  METADATAHELPER
#define  METADATAHELPER

#include "stdint.h"

#include <atomic>

class MetadataHelper{


public:
    __device__ __host__ static uint64_t set_epoch(uint64_t value, const uint64_t epoch){
        return (value & (~(EPOCH_MASK << EPOCH_OFFSET))) | (epoch << EPOCH_OFFSET);
    };

    __device__ __host__ static uint64_t get_epoch(uint64_t value){   
        return  (value >> EPOCH_OFFSET) & EPOCH_MASK;
    };

    __device__ __host__ static uint64_t set_rts(uint64_t value , const uint64_t rts){
        return (value & (~(RTS_MASK << RTS_OFFSET))) | (rts << RTS_OFFSET);
    };

    __device__ __host__ static uint64_t get_rts(uint64_t value){
        return (value >> RTS_OFFSET) & RTS_MASK;
    };

    __device__ __host__ static uint64_t set_wts(uint64_t value,const  uint64_t wts){
        return (value & (~(WTS_MASK << WTS_OFFSET))) | (wts << WTS_OFFSET);
    };

    __device__ __host__ static uint64_t get_wts(uint64_t value){
        return (value >> WTS_OFFSET) & WTS_MASK;
    };

    __device__ static bool reserve_read(uint64_t* metadata_ptr,const uint16_t epoch,const uint16_t Tid){
        uint64_t oldvalue,newvalue;
        do{
            oldvalue=*metadata_ptr;

            uint64_t old_epoch = get_epoch(oldvalue);
            uint64_t old_rts = get_rts(oldvalue);

            if(epoch > old_epoch){
                //当前事务的轮次大于之前预定的轮次
                newvalue = set_epoch(0,epoch);
                newvalue = set_rts(newvalue,Tid);
            }else if(epoch == old_epoch){
                //当前要做读预定的事务和上一个预定是同一轮次
                //比较事务tid
                if(old_rts < Tid && old_rts!=0){
                    printf("in device function:reserve_read epoch:%d  Tid:%d  failed reserve read! old rts  is %d\n",epoch,Tid,old_rts);

                    //旧的预定id较小 则这次预定失败
                    return false;
                }
                newvalue = oldvalue;
                newvalue = set_rts(newvalue,Tid);
            }
        }while(!(atomicCAS(metadata_ptr,oldvalue,newvalue)==oldvalue));
        printf("in device function:reserve_read epoch:%d  Tid:%d  successful reserve read!\n",epoch,Tid);
        return true;
    };

    __device__ static  bool reserve_write(uint64_t* metadata_ptr,const int epoch,const int Tid){
        uint64_t oldvalue,newvalue;
        do{    
            oldvalue=*metadata_ptr;

            uint64_t old_epoch = get_epoch(oldvalue);
            uint64_t old_wts = get_wts(oldvalue);

            if(epoch > old_epoch){
                //当前事务的轮次大于之前预定的轮次
                newvalue = set_epoch(0,epoch);
                newvalue = set_wts(newvalue,Tid);
            }else if(epoch == old_epoch){
                //当前要做读预定的事务和上一个预定是同一轮次
                //比较事务tid
                if(old_wts < Tid && old_wts!=0){
                    printf("in device function:reserve_write epoch:%d  Tid:%d  failed reserve write! old wts is %d\n",epoch,Tid,old_wts);
                    //旧的预定id较小 则这次预定失败
                    return false;
                }
                newvalue = oldvalue;
                newvalue = set_wts(newvalue,Tid);
            }
        }while(!(atomicCAS(metadata_ptr,oldvalue,newvalue)==oldvalue));
        printf("in device function:reserve_write epoch:%d  Tid:%d  successful reserve write!\n",epoch,Tid);
        return true;
    };

    

private:
    /*
   * [epoch (16) | read-rts  (16) | write-wts (16)]
   *
   */
    static constexpr int EPOCH_OFFSET = 32;
    static constexpr uint64_t EPOCH_MASK = 0xffffull;
  
    static constexpr int RTS_OFFSET = 16;
    static constexpr uint64_t RTS_MASK = 0xffffull;
  
    static constexpr int WTS_OFFSET = 0;
    static constexpr uint64_t WTS_MASK = 0xffffull;

};



#endif