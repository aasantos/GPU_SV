#ifndef runfunc.cuh
#define runfunc.cuh

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


void simulate_dlm(float *x,int n,float sigmav,float mu,float phi,float sigma)
{
  //int n = 1000;
  //float sigmav = 0.15;
  //float mu = -0.5;
  //float phi = 0.97;
  //float sigma = 0.2;
  //
  //
  Random<float> *random = new Random<float>(8687);
  float a = 0.0;
  //
  for(int i=0;i<100;i++) a = mu + phi*(a - mu) + sigma*random->normal();
  //float *x = new float[n];
  for(int i=0;i<n;i++)
  {
    a = mu + phi*(a - mu) + sigma*random->normal();
    x[i] = a + sigmav*random->normal();
  }
  delete random;
}


void run_1()
{
  //
  int n = 1000;
  float sigmav = 0.15;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  float *x = new float[n];
  //
  simulate_dlm(x,n,sigmav,mu,phi,sigma);
  //
  DLMModel<float> *dlm = new DLMModel<float>(x,n,sigmav,mu,phi,sigma);
  
  
  delete dlm;
  //
  delete[] x;
}


#endif
