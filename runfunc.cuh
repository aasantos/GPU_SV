#ifndef runfunc_cuh
#define runfunc_cuh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "io.cuh"
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"
#include "ar1model.cuh"
#include "dlm.cuh"
#include "sv.cuh"
#include "svt.cuh"
#include "svl.cuh"
#include "svtl.cuh"
#include "kernelfunc.cuh"

void estimate_gpu_sv_sp500()
{
  printf("Start estimating .... \n");
  int n;
  float *xi = readArray<float>("sp500_ret_80_87.txt",&n);
  //
    float *x;
    cudaMallocManaged(&x,n*sizeof(float));
    for(int i=0;i<n;i++) x[i] = xi[i];
    //
    int niter = 5000;
    float *musimul;
    float *phisimul;
    float *sigmasimul;
    unsigned int *seed;
    //
    cudaMallocManaged(&musimul,niter*sizeof(float));
    cudaMallocManaged(&phisimul,niter*sizeof(float));
    cudaMallocManaged(&sigmasimul,niter*sizeof(float));
    cudaMallocManaged(&seed,niter*sizeof(unsigned int));
    //
    srand(time(NULL));
    for(int i=0;i<niter;i++) seed[i] = rand();
    //
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,524288000L);
    kernel_sv<<<512,128>>>(x,n,seed,musimul,phisimul,sigmasimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    //
    mumean[k] = mmu;
    phimean[k] = mphi;
    sigmamean[k] = msigma;
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f\n",mmu,mphi,msigma);
    //
    cudaFree(musimul);
    cudaFree(phisimul);
    cudaFree(sigmasimul);
    cudaFree(seed);
    cudaFree(x);
    cudaDeviceReset();
  //
  free(xi);
  printf("Done ... \n");
}

void estimate_sv_sp500()
{
  int n;
  float *x = readArray<float>("sp500y.txt",&n);
  printf("Number of observations: %d\n",n);
    //
    float mut = -0.5;
    float phit = 0.97;
    float sigmat = 0.2;
    float rhot = -0.7;
    //
    int nwarmup = 5000;
    int niter = 20000;
    //
    SVLModel<float> *model = new SVLModel<float>(x,n,mut,phit,sigmat,rhot);
    //
    for(int i=0;i<500;i++){
        model->simulatestates();
    }
    //
    // warmup
    for(int i=0;i<nwarmup;i++){
        if(i%100 == 0){
            printf("Warmup Iteration: %d/%d\n",i,nwarmup);
        }
        model->simulatestates();
        model->simulatemu();
        model->simulatephi();
        model->simulatesigmarho();
    }
    //
    //
    float *musimul = new float[niter];
    float *phisimul = new float[niter];
    float *sigmasimul = new float[niter];
    float *rhosimul = new float[niter];
    //
    //
    for(int i=0;i<niter;i++){
        if(i%100 == 0){
            printf("Iteration: %d/%d\n",i,niter);
        }
        model->simulatestates();
        float mt = model->simulatemu();
        float pt = model->simulatephi();
        model->simulatesigmarho();
        float st = model->getsigma();
        float rt = model->getrho();
        musimul[i] = mt;
        phisimul[i] = pt;
        sigmasimul[i] = st;
        rhosimul[i] = rt;
    }
    //
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    float mrho = Vector<float>(rhosimul,niter).mean();
    //
    printf("mu: %.3f; phi: %.3f; sigma: %.3f; rho: %.3f\n",mmu,mphi,msigma,mrho);
    //
    //
    delete[] musimul;
    delete[] phisimul;
    delete[] sigmasimul;
    delete[] rhosimul;
    delete model;
  free(x);
}


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
    model->setseed(seed[idx]);
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
    sigmavs[idx] = model->simulatesigmav();
    mus[idx] = model->simulatemu();
    phis[idx] = model->simulatephi();
    sigmas[idx] = model->simulatesigma();
    //
    delete model;
  }
}


void run_dlm_gpu()
{
  printf("Starting .. \n");
  int n = 1000;
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
  int niter = 1000;
  float *sigmavsimul;
  float *musimul;
  float *phisimul;
  float *sigmasimul;
  unsigned int *seed;
  //
  cudaMallocManaged(&sigmavsimul,niter*sizeof(float));
  cudaMallocManaged(&musimul,niter*sizeof(float));
  cudaMallocManaged(&phisimul,niter*sizeof(float));
  cudaMallocManaged(&sigmasimul,niter*sizeof(float));
  cudaMallocManaged(&seed,niter*sizeof(unsigned int));
  //
  srand(time(NULL));
  for(int i=0;i<niter;i++) seed[i] = rand();
  //
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,524288000L);
  kernel_dlm<<<512,128>>>(x,n,seed,sigmavsimul,musimul,phisimul,sigmasimul,niter);
  cudaDeviceSynchronize();
  //
  writeArray(sigmavsimul,"sigmavsimul.txt",niter);
  writeArray(musimul,"musimul.txt",niter);
  writeArray(phisimul,"phisimul.txt",niter);
  writeArray(sigmasimul,"sigmasimul.txt",niter);
  //
  cudaFree(sigmavsimul);
  cudaFree(musimul);
  cudaFree(phisimul);
  cudaFree(sigmasimul);
  cudaFree(seed);
  cudaFree(x);
  printf("Done .. \n");
}

void run_sv_gpu()
{
  printf("Starting .. \n");
  //
  int n = 2000;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  int m = 100;
  float *mumean = new float[m];
  float *phimean = new float[m];
  float *sigmamean = new float[m];
  //
  float *yy = new float[n*m];
  int kiter = -1;
  //
  //
  //
  for(int k=0;k<m;k++){
    //
    printf("Iteration: %d\n",k);
    float *x;
    cudaMallocManaged(&x,n*sizeof(float));
    simulate_sv(x,n,mu,phi,sigma);
    for(int j=0;j<n;j++){
      kiter++;
      yy[kiter] = x[j];
    }
    //
    int niter = 5000;
    float *musimul;
    float *phisimul;
    float *sigmasimul;
    unsigned int *seed;
    //
    cudaMallocManaged(&musimul,niter*sizeof(float));
    cudaMallocManaged(&phisimul,niter*sizeof(float));
    cudaMallocManaged(&sigmasimul,niter*sizeof(float));
    cudaMallocManaged(&seed,niter*sizeof(unsigned int));
    //
    srand(time(NULL));
    for(int i=0;i<niter;i++) seed[i] = rand();
    //
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,524288000L);
    kernel_sv<<<512,128>>>(x,n,seed,musimul,phisimul,sigmasimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    //
    mumean[k] = mmu;
    phimean[k] = mphi;
    sigmamean[k] = msigma;
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f\n",mmu,mphi,msigma);
    //
    cudaFree(musimul);
    cudaFree(phisimul);
    cudaFree(sigmasimul);
    cudaFree(seed);
    cudaFree(x);
    cudaDeviceReset();
    //
    sleep(10);
   }
   //
   writeArray<float>(yy,"yy.txt",n*m);
   writeArray<float>(mumean,"mumean.txt",m);
   writeArray<float>(phimean,"phimean.txt",m);
   writeArray<float>(sigmamean,"sigmamean.txt",m);
   //
   delete[] yy;
   delete[] mumean;
   delete[] phimean;
   delete[] sigmamean;
   //
   printf("Done .. \n");
}

#endif
