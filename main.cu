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
  printf("Start .... \n");
  //
  estimate_sv("sp500_ret_80_87.txt",-0.5,0.95,0.2);
  estimate_svt("sp500_ret_80_87.txt",-0.5,0.95,0.2,20);
  estimate_svl("sp500_ret_80_87.txt",-0.5,0.95,0.2,-0.5);
  estimate_svtl("sp500_ret_80_87.txt",-0.5,0.95,0.2,-0.5,20);
  //
  estimate_sv_gpu("sp500_ret_80_87.txt");
  estimate_svt_gpu("sp500_ret_80_87.txt");
  estimate_svl_gpu("sp500_ret_80_87.txt");
  estimate_svtl_gpu("sp500_ret_80_87.txt");

  /*
  int flag = atoi(argv[1]);
  if(flag == 0) estimate_sv_gpu("sp500_ret_80_87.txt");
  if(flag == 1) estimate_svl_gpu("sp500_ret_80_87.txt");
  if(flag == 2) estimate_svtl_gpu("sp500_ret_80_87.txt");
  if(flag == 3) estimate_svt_gpu("sp500_ret_80_87.txt");
  */
  //
  /*
  double time_elapsed = 0.0;
  clock_t begin = clock();
  estimate_svl_test("sp500_ret_80_87.txt");
  clock_t end = clock();
  time_elapsed += (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Running time: %.3f\n",time_elapsed);
  */
  //
  /*
  double time_elapsed = 0.0;
  clock_t begin = clock();
  estimate_svt_test("sp500_ret_80_87.txt");
  clock_t end = clock();
  time_elapsed += (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Running time: %.3f\n",time_elapsed);
  */
  //
  printf("Done ... \n");
  return 0;
}
