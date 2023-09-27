#ifndef GPU_CUH
#define GPU_CUH
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void kernel_DrugSimulation(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                      double *time, double *states, double *out_dt,  double *cai_result, 
                                      double *ina, double *inal, 
                                      double *ical, double *ito,
                                      double *ikr, double *iks,
                                      double *ik1,
                                      unsigned int sample_size,
                                      param_t *p_param
                                      );

__device__ void kernel_DoDrugSim(double *d_ic50, double *d_CONSTANTS, double *d_STATES, double *d_RATES, double *d_ALGEBRAIC, 
                                       double *time, double *states, double *out_dt,  double *cai_result, 
                                       double *ina, double *inal,
                                       double *ical, double *ito,
                                       double *ikr, double *iks, 
                                       double *ik1,
                                       double *tcurr, double *dt, unsigned short sample_id, unsigned int sample_size,
                                       param_t *p_param
                                       );


#endif