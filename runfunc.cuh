#ifndef runfunc_cuh
#define runfunc_cuh

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


void simulate_dlm(float *x,int n,float sigmav,float mu,float phi,float sigma)
{
  //int n = 1000;
  //float sigmav = 0.15;
  //float mu = -0.5;
  //float phi = 0.97;
  //float sigma = 0.2;
  //
  //
  srand(time(NULL));
  Random<float> *random = new Random<float>(rand());
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


void simulate_sv(float *x,int n,float mu,float phi,float sigma)
{
  //int n = 1000;
  //float sigmav = 0.15;
  //float mu = -0.5;
  //float phi = 0.97;
  //float sigma = 0.2;
  //
  //
  srand(time(NULL));
  Random<float> *random = new Random<float>(rand());
  float a = 0.0;
  //
  for(int i=0;i<100;i++) a = mu + phi*(a - mu) + sigma*random->normal();
  //float *x = new float[n];
  for(int i=0;i<n;i++)
  {
    a = mu + phi*(a - mu) + sigma*random->normal();
    x[i] = exp(0.5*a)*random->normal();
  }
  delete random;
}

void run_1()
{
  //
  printf("Start running ... \n");
  //
  int n = 5000;
  float sigmav = 0.15;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  int nwarmup = 1000;
  int niter = 10000;
  //
  for(int k=0;k<20;k++){
  //
  float *x = new float[n];
  simulate_dlm(x,n,sigmav,mu,phi,sigma);
  //
  DLMModel<float> *dlm = new DLMModel<float>(x,n,sigmav,mu,phi,sigma);
  //
  for(int i=0;i<500;i++){
    dlm->simulatestates();
  }
  // warmup
  for(int i=0;i<nwarmup;i++){ 
    dlm->simulatestates();
    dlm->simulatesigmav();
    dlm->simulatemu();
    dlm->simulatephi();
    dlm->simulatesigma();
  }
  //  
  float *sigmavsimul = new float[niter];
  float *musimul = new float[niter];
  float *phisimul = new float[niter];
  float *sigmasimul = new float[niter];
  //
  for(int i=0;i<niter;i++){
    dlm->simulatestates();
    sigmavsimul[i] = dlm->simulatesigmav();
    musimul[i] = dlm->simulatemu();
    phisimul[i] = dlm->simulatephi();
    sigmasimul[i] = dlm->simulatesigma();
  }
  //
  float msigmav = Vector<float>(sigmavsimul,niter).mean();
  float mmu = Vector<float>(musimul,niter).mean();
  float mphi = Vector<float>(phisimul,niter).mean();
  float msigma = Vector<float>(sigmasimul,niter).mean();
  //
  printf("sigmav: %.3f; mu: %.3f; phi: %.3f; sigma: %.3f\n",msigmav,mmu,mphi,msigma);
  //
  //memory free zone
  delete dlm;
  delete[] x;
  delete[] sigmavsimul;
  delete[] musimul;
  delete[] phisimul;
  delete[] sigmasimul;
  //
  }
  
  //
  printf("Done ... \n");
  //
}


void run_2()
{
  //
  printf("Start running ... \n");
  //
  int n = 5000;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  int nwarmup = 1000;
  int niter = 10000;
  //
  for(int k=0;k<20;k++){
  //
  float *x = new float[n];
  simulate_sv(x,n,mu,phi,sigma);
  //
  SVModel<float> *model = new SVModel<float>(x,n,mu,phi,sigma);
  //
  for(int i=0;i<500;i++){
    model->simulatestates();
  }
  // warmup
  for(int i=0;i<nwarmup;i++){ 
    model->simulatestates();
    model->simulatemu();
    model->simulatephi();
    model->simulatesigma();
  }
  //  
  float *musimul = new float[niter];
  float *phisimul = new float[niter];
  float *sigmasimul = new float[niter];
  //
  for(int i=0;i<niter;i++){
    model->simulatestates();
    musimul[i] = model->simulatemu();
    phisimul[i] = model->simulatephi();
    sigmasimul[i] = model->simulatesigma();
  }
  //
  float mmu = Vector<float>(musimul,niter).mean();
  float mphi = Vector<float>(phisimul,niter).mean();
  float msigma = Vector<float>(sigmasimul,niter).mean();
  //
  printf("mu: %.3f; phi: %.3f; sigma: %.3f\n",mmu,mphi,msigma);
  //
  //memory free zone
  delete model;
  delete[] x;
  delete[] musimul;
  delete[] phisimul;
  delete[] sigmasimul;
  //
  }
  
  //
  printf("Done ... \n");
  //
}

__global__ void kernel_dlm(float *x,int n,unsigned int *seed,float *sigmavs,float *mus,float *phis,float *sigmas,int niter)
{
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < niter)
  {
    int nwarmup = 1000;
    float sigmav = 0.2;
    float mu = 0.0;
    float phi = 0.95;
    float sigma = 0.2;
    //
    DLMModel<float> *model = new DLMModel<float>(x,n,sigmav,mu,phi,sigma);
    //
    for(int i=0;i<100;i++){
      model->simulatestates();
    }
    // warmup
    for(int i=0;i<nwarmup;i++){ 
      model->simulatestates();
      model->simulatesigmav();
      model->simulatemu();
      model->simulatephi();
      model->simulatesigma();
    }
    //
    sigmavs[idx] = model->simulatesigmasv();
    mus[idx] = model->simulatemu();
    phis[idx] = model->simulatephi();
    sigmas[idx] = model->simulatesigma();
    //
    delete model;
  }
  
}

void run_dlm_gpu()
{
  int n = 5000;
  float sigmav = 0.15;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  float *x;
  cudaMallocManaged(&x,n*sizeof(float));
  simulate_dlm(x,n,sigmav,mu,phi,sigma);
  //
  //
  int niter = 5000;
  
  
  cudaFree(x);
}
  

#endif
