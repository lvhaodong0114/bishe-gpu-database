#include <atomic>
#include "metadatahelper.cuh"
#include <iostream>

#include <mutex>

template <typename T>
bool CAS(T* addr, T oldval, T newval) {
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    if (*addr == oldval) {
        *addr = newval;
        return true;
    }
    return false;
}

bool reserve_read_cpu(uint64_t* metadata_ptr,const uint16_t epoch,const uint16_t Tid){
        uint64_t oldvalue,newvalue;


        //转换一个原子变量 最终仍然操作同一份数据
        // std::atomic_ref<uint64_t> metadata_ref(*metadata_ptr,std::memory_order_relaxed);

        do{
            oldvalue=*metadata_ptr;

            uint64_t old_epoch = MetadataHelper::get_epoch(oldvalue);
            uint64_t old_rts = MetadataHelper::get_rts(oldvalue);

            if(epoch > old_epoch){
                //当前事务的轮次大于之前预定的轮次
                newvalue = MetadataHelper::set_epoch(0,epoch);
                newvalue = MetadataHelper::set_rts(newvalue,Tid);
            }else if(epoch == old_epoch){
                //当前要做读预定的事务和上一个预定是同一轮次
                //比较事务tid
                if(old_rts < Tid && old_rts!=0){
                    printf("in device function:reserve_read epoch:%d  Tid:%d  failed reserve read! old rts  is %d\n",epoch,Tid,old_rts);
                    //旧的预定id较小 则这次预定失败
                    return false;
                }
                newvalue = oldvalue;
                newvalue = MetadataHelper::set_rts(newvalue,Tid);
            }
        }while(!CAS(metadata_ptr,oldvalue,newvalue));
        printf("in cpu function:reserve_read epoch:%d  Tid:%d  successful reserve read!\n",epoch,Tid);
        return true;
    };

bool reserve_write_cpu(uint64_t* metadata_ptr,const int epoch,const int Tid){
        uint64_t oldvalue,newvalue;

        // std::atomic_ref<uint64_t> metadata_ref(*metadata_ptr,std::memory_order_relaxed);

        do{    
            oldvalue=*metadata_ptr;

            uint64_t old_epoch = MetadataHelper::get_epoch(oldvalue);
            uint64_t old_wts = MetadataHelper::get_wts(oldvalue);

            if(epoch > old_epoch){
                //当前事务的轮次大于之前预定的轮次
                newvalue = MetadataHelper::set_epoch(0,epoch);
                newvalue = MetadataHelper::set_wts(newvalue,Tid);
            }else if(epoch == old_epoch){
                //当前要做读预定的事务和上一个预定是同一轮次
                //比较事务tid
                if(old_wts < Tid && old_wts!=0){
                    printf("in device function:reserve_write epoch:%d  Tid:%d  failed reserve write! old wts is %d\n",epoch,Tid,old_wts);
                    //旧的预定id较小 则这次预定失败
                    return false;
                }
                newvalue = oldvalue;
                newvalue = MetadataHelper::set_wts(newvalue,Tid);
            }
        }while(!CAS(metadata_ptr,oldvalue,newvalue));
        printf("in cpu function:reserve_write epoch:%d  Tid:%d  successful reserve write!\n",epoch,Tid);
        return true;
    };
