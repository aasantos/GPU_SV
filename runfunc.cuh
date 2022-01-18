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
  printf("Start running ... \n");
  //
  int n = 5000;
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
  //
  //
  for(int i=0;i<500;i++){
    dlm->simulatestates();
  }
  //
  // warmup
  int nwarmup = 1000;
  for(int i=0;i<nwarmup;i++){ 
    dlm->simulatestates();
    dlm->simulatesigmav();
    dlm->simulatemu();
    dlm->simulatephi();
    dlm->simulatesigma();
  }
  //
  int niter = 10000;
  float *sigmavsimul = new float[niter];
  float *musimul = new float[niter];
  float *phisimul = new float[niter];
  float *sigmasimul = new float[niter];
  for(int i=0;i<niter;i++){
    dlm->simulatestates();
    sigmavsimul[i] = dlm->simulatesigmav();
    musimul[i] = dlm->simulatemu();
    phisimul[i] = dlm->simulatephi();
    sigmasimul[i] = dlm->simulatesigma();
  }
  //
  //
  float msigmav = Vector<float>(sigmavsimul,niter).mean();
  float mmu = Vector<float>(musimul,niter).mean();
  float mphi = Vector<float>(phisimul,niter).mean();
  float msigma = Vector<float>(sigmasimul,niter).mean();
  //
  //
  printf("sigmav: %.3f; mu: %.3f; phi: %.3f; sigma: %.3f\n",msigmav,mmu,mphi,msigma);
  //
  // memory free zone
  delete[] sigmavsimul;
  delete[] musimul;
  delete[] phisimul;
  delete[] sigmasimul;
  //
  delete dlm;
  //
  delete[] x;
  //
  printf("Done ... \n");
  //
  
}


#endif
