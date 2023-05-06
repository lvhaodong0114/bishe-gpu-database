#ifndef GPUALLOCATOR_CUH
#define GPUALLOCATOR_CUH

#include <unordered_map>

#define GPUALLOCATOR_DEFAULTSIZE 999999999


#define	KEY_INVALID		0
#define THREADS_PER_BLOCK 256

/* Load factors are declared as integers to avoid floating point operations */
#define MIN_LOAD_FACTOR 65
#define MAX_LOAD_FACTOR 80

#define INFO(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
	}	\
} while (0)


/* std::cout cannot use in device function */
// #define cudaCheckError(e) { \
// 	if(e!=cudaSuccess) { \
// 		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(e) << " (" << e << ")" << std::endl; \
// 		exit(0); \
// 	 }\
// }


#define cudaCheckError(e) do {                         \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(1);                             \
    }                                                 \
  } while(0)


#define CUDACHECKERROR() do {                         \
cudaError_t err = cudaGetLastError();               \
if( err != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(err));   \
    exit(1);                             \
}                                                 \
} while(0)



class Gpu_Allocator{

    public:
        uint64_t allocMax;
        uint64_t allocCurrent;
        std::unordered_map<void* , size_t> allocMap;


    public:
        Gpu_Allocator(uint64_t allocMax):allocMax(allocMax),allocCurrent(0){
        };

        cudaError_t _cudaMalloc( void** devPtr, size_t size ){
            //printf("in func _cudamalloc!\n");
            if((this->allocCurrent +size) <= this->allocMax){
                cudaError_t rt = cudaMalloc(devPtr,size);
                this->allocCurrent = this->allocCurrent +size;
                this->allocMap[*devPtr] = size;
                INFO(false, "cudaMalloc successful!" );
                return rt;
            }else{
                INFO(true, "cudaMalloc would exceed allowed max alloc size" );
            }
        };

        cudaError_t _cudaMallocManaged( void** devPtr, size_t size ){
            if ( (this->allocCurrent + size) <= this->allocMax ) {
                cudaError_t rt = cudaMallocManaged(devPtr, size);
                this->allocCurrent += size;
                this->allocMap[ *devPtr ] = size;
                CUDACHECKERROR();
                return rt;
            } else {
                INFO(true, "cudaMallocManaged would exceed allowed max alloc size" );
            }
        };
        cudaError_t _cudaFree( void* devPtr ){
            if (this->allocMap.find(devPtr) == this->allocMap.end()) {
                CUDACHECKERROR();
                return cudaErrorInvalidValue;
            } else {
                this->allocCurrent -= this->allocMap.find(devPtr)->second;
                this->allocMap.erase(devPtr);
                return cudaFree(devPtr);
            }
        };

        float _loadFactor();

        uint64_t _used(){
            return this->allocCurrent;
        };
};


#endif