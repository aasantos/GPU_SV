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
int main(int argc,char **argv)
{
  printf("Starting .... \n");
  //
  const char* file = argv[1]; 
  //
  int flag = atoi(argv[2]);
  if(flag == 0) estimate_sv(file,-0.5,0.95,0.2);
  if(flag == 1) estimate_svt(file,-0.5,0.95,0.2,20);
  if(flag == 2) estimate_svl(file,-0.5,0.95,0.2,-0.5);
  if(flag == 3) estimate_svtl(file,-0.5,0.95,0.2,-0.5,20);
  if(flag == 4) estimate_sv_gpu(file);
  if(flag == 5) estimate_svt_gpu(file);
  if(flag == 6) estimate_svl_gpu(file);
  if(flag == 7) estimate_svtl_gpu(file);
  //
  //
  printf("Done ... \n");
  return 0;
}
