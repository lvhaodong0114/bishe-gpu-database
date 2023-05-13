#ifndef MARCO_DEFINE_H
#define MARCO_DEFINE_H


//1. 轮次设置

//开启该宏时  只进行一轮测试 关闭时执行直至事务处理完毕
//默认不开启  
// #define TEST_ONE_EPOCH


//2.  设备设置

// 用于对比实验  开启该宏时，会将事务处理模块运行在GPU上 不定义时会运行在CPU上
#define OPEN_GPU

// 开启该宏时  在GPU上的事务处理模块开启操作级并行
// #define OP_PARALLEL



//3.  参数设置

// 开启该宏时  支持生成带删除的事务  并用于测试
// #define OPEN_DELETE     


#endif