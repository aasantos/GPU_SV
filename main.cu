#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include "io.cuh"
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"
#include "ar1model.cuh"
#include "dlm.cuh"
#include "sv.cuh"
#include "runfunc.cuh"
//
//
//
int main()
{
  printf("Hello from GPU_DLM\n");
  //
  //run_1();
  //
  //run_2();
  //
  //run_dlm_gpu();
  //
  //run_sv_gpu();
  //
  estimate_sv_sp500();
  //
  return 0;
}
