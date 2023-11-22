#ifndef OHARA_RUDY_2011_HPP
#define OHARA_RUDY_2011_HPP

#include "enums/enum_Ohara_Rudy_2011.hpp"
#include <cuda_runtime.h>

  	// void initConsts();
	// void initConsts(double type);
	// void initConsts(bool is_dutta);
	__device__ void initConsts(double *CONSTANTS, double *STATES, double type, double conc, double *ic50, double *cvar, bool is_dutta, bool is_cvar,  int offset);
	__device__ void computeRates(double TIME, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC, int offset);
	__device__ void solveAnalytical(double *CONSTANTS, double *STATES, double *ALGEBRAIC, double *RATES, double dt, int offset);
	__device__ double set_time_step(double TIME,double time_point,double max_time_step,
  double* CONSTANTS,
  double* RATES,
  double* STATES,
  double* ALGEBRAIC,
  int offset);
    __device__ void applyDrugEffect(double *CONSTANTS, double conc, double *ic50, double epsilon, int offset);
	__device__ void ___applyDutta(double *CONSTANTS, int offset);
	__device__ void ___applyCvar(double *CONSTANTS, double *cvar, int offset);
	// void ___initConsts(double *CONSTANTS, double *STATES, double type, int offset);

#endif