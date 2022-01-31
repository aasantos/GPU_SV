#ifndef kernelfunc_cuh
#define kernelfunc_cuh


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

__global__ void estimate_sv_sp500_gpu(float *x,int n,float *musimul,float *phisimul,float *sigmasimul,float *rhosimul,unsigned int *seed,int niter)
{
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < niter){
    //
    float mut = -0.5;
    float phit = 0.97;
    float sigmat = 0.2;
    float rhot = -0.7;
    //
    int nwarmup = 1000;
    //
    SVLModel<float> *model = new SVLModel<float>(x,n,mut,phit,sigmat,rhot);
    model->setseed(seed[idx]);
    //
    for(int i=0;i<100;i++){
        model->simulatestates();
    }
    //
    // 
    for(int i=0;i<nwarmup;i++){
        model->simulatestates();
        model->simulatemu();
        model->simulatephi();
        model->simulatesigmarho();
    }
    //
    musimul[idx] = model->simulatemu();
    phisimul[idx] = model->simulatephi();
    model->simulatesigmarho();
    sigmasimul[idx] = model->getsigma();
    rhosimul[idx] = model->getrho();
    //
    delete model;
    //
  }// if(idx < niter)
}



#endif
