#ifndef runfunc_cuh
#define runfunc_cuh
//
//
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
//
//
//   Simulate from models
//
//
void simulate_dlm(float *x,int n,float sigmav,float mu,float phi,float sigma)
{
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
//
//
void simulate_sv(float *x,int n,float mu,float phi,float sigma)
{
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
//
//
//estimate SV models
//
//
//
void estimate_sv(const char *file,float mu,float phi,float sigma)
{
    int n;
    float *x = readArray<float>(file,&n);
    printf("Number of observations: %d\n",n);
    //
    int nwarmup = 5000;
    int niter = 20000;
    //
    SVModel<float> *model = new SVModel<float>(x,n,mu,phi,sigma);
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
        model->simulatesigma();
    }
    //
    //
    float *musimul = new float[niter];
    float *phisimul = new float[niter];
    float *sigmasimul = new float[niter];
    //
    //
    for(int i=0;i<niter;i++){
        if(i%100 == 0){
            printf("Iteration: %d/%d\n",i,niter);
        }
        model->simulatestates();
        musimul[i] = model->simulatemu();
        phisimul[i] = model->simulatephi();
        sigmasimul[i] = model->simulatesigma();
    }
    //
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    //
    printf("mu: %.3f; phi: %.3f; sigma: %.3f\n",mmu,mphi,msigma);
    //
    //
    //
    FILE *fp;
    fp = fopen("svestim.txt", "wa");
    fprintf(fp,"mu phi sigma\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f\n",musimul[i],phisimul[i],sigmasimul[i]);
    fclose(fp);
    //
    delete[] musimul;
    delete[] phisimul;
    delete[] sigmasimul;
    delete model;
  free(x);
}
//
//
//
void estimate_svt(const char *file,float mu,float phi,float sigma,int nu)
{
    int n;
    float *x = readArray<float>(file,&n);
    printf("Number of observations: %d\n",n);
    //
    int nwarmup = 5000;
    int niter = 20000;
    //
    SVTModel<float> *model = new SVTModel<float>(x,n,mu,phi,sigma,nu);
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
        model->simulatesigma();
        model->simulatenu();
    }
    //
    //
    float *musimul = new float[niter];
    float *phisimul = new float[niter];
    float *sigmasimul = new float[niter];
    int *nusimul = new int[niter];
    //
    //
    for(int i=0;i<niter;i++){
        if(i%100 == 0){
            printf("Iteration: %d/%d\n",i,niter);
        }
        model->simulatestates();
        musimul[i] = model->simulatemu();
        phisimul[i] = model->simulatephi();
        sigmasimul[i] = model->simulatesigma();
        nusimul[i] = model->simulatenu();
    }
    //
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    float mnu = Vector<int>(nusimul,niter).mean();
    //
    printf("mu: %.3f; phi: %.3f; sigma: %.3f; nu: %.3f\n",mmu,mphi,msigma,mnu);
    //
    //
    FILE *fp;
    fp = fopen("svtestim.txt", "wa");
    fprintf(fp,"mu phi sigma nu\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %d\n",musimul[i],phisimul[i],sigmasimul[i],nusimul[i]);
    fclose(fp);
    //
    delete[] musimul;
    delete[] phisimul;
    delete[] sigmasimul;
    delete[] nusimul;
    //
    delete model;
  free(x);
}
//
//
//
void estimate_svl(const char *file,float mu,float phi,float sigma,float rho)
{
    int n;
    float *x = readArray<float>(file,&n);
    printf("Number of observations: %d\n",n);
    //
    int nwarmup = 5000;
    int niter = 20000;
    //
    SVLModel<float> *model = new SVLModel<float>(x,n,mu,phi,sigma,rho);
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
    FILE *fp;
    fp = fopen("svlestim.txt", "wa");
    fprintf(fp,"mu phi sigma rho\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %.4f\n",musimul[i],phisimul[i],sigmasimul[i],rhosimul[i]);
    fclose(fp);
    //
    //
    delete[] musimul;
    delete[] phisimul;
    delete[] sigmasimul;
    delete[] rhosimul;
    delete model;
  free(x);
}
//
//
//
void estimate_svtl(const char *file,float mu,float phi,float sigma,float rho,int nu)
{
    int n;
    float *x = readArray<float>(file,&n);
    printf("Number of observations: %d\n",n);
    //
    int nwarmup = 5000;
    int niter = 20000;
    //
    SVTLModel<float> *model = new SVTLModel<float>(x,n,mu,phi,sigma,rho,nu);
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
        model->simulatenu();
    }
    //
    //
    float *musimul = new float[niter];
    float *phisimul = new float[niter];
    float *sigmasimul = new float[niter];
    float *rhosimul = new float[niter];
    int *nusimul = new int[niter];
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
        nusimul[i] = model->simulatenu();
    }
    //
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    float mrho = Vector<float>(rhosimul,niter).mean();
    float mnu = Vector<int>(nusimul,niter).mean();
    //
    printf("mu: %.3f; phi: %.3f; sigma: %.3f; rho: %.3f; nu: %.3f\n",mmu,mphi,msigma,mrho,mnu);
    //
    //
    //
    FILE *fp;
    fp = fopen("svtlestim.txt", "wa");
    fprintf(fp,"mu phi sigma rho nu\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %.4f %d\n",musimul[i],phisimul[i],sigmasimul[i],rhosimul[i],nusimul[i]);
    fclose(fp);
    //
    //
    delete[] musimul;
    delete[] phisimul;
    delete[] sigmasimul;
    delete[] rhosimul;
    delete[] nusimul;
    delete model;
  free(x);
}
//
//
//
void estimate_sv_gpu(const char *file)
{
    printf("Start estimating SV .... \n");
    int n;
    float *xi = readArray<float>(file,&n);
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
    double time_spent = 0.0;
    clock_t begin = clock();
    //
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,2097152000L);
    kernel_sv<<<1024,8>>>(x,n,seed,musimul,phisimul,sigmasimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    //
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC; 
    printf("The elapsed time is %f seconds\n", time_spent);
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f\n",mmu,mphi,msigma);
    //
    FILE *fp;
    fp = fopen("svestimgpu.txt", "wa");
    fprintf(fp,"mu phi sigma\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f\n",musimul[i],phisimul[i],sigmasimul[i]);
    fclose(fp);
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
//
//
//
void estimate_svl_gpu(const char *file)
{
    printf("Start estimating SVL .... \n");
    int n;
    float *xi = readArray<float>(file,&n);
    //
    float *x;
    cudaMallocManaged(&x,n*sizeof(float));
    for(int i=0;i<n;i++) x[i] = xi[i];
    //
    int niter = 5000;
    float *musimul;
    float *phisimul;
    float *sigmasimul;
    float *rhosimul;
    unsigned int *seed;
    //
    cudaMallocManaged(&musimul,niter*sizeof(float));
    cudaMallocManaged(&phisimul,niter*sizeof(float));
    cudaMallocManaged(&sigmasimul,niter*sizeof(float));
    cudaMallocManaged(&rhosimul,niter*sizeof(float));
    cudaMallocManaged(&seed,niter*sizeof(unsigned int));
    //
    srand(time(NULL));
    for(int i=0;i<niter;i++) seed[i] = rand();
    //
    double time_spent = 0.0;
    clock_t begin = clock();
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,2097152000L);
    kernel_svl<<<1024,8>>>(x,n,seed,musimul,phisimul,sigmasimul,rhosimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    float mrho = Vector<float>(rhosimul,niter).mean();
    //
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC; 
    printf("The elapsed time is %f seconds\n", time_spent);
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f; rho: %.4f\n",mmu,mphi,msigma,mrho);
    //
    FILE *fp;
    fp = fopen("svlestimgpu.txt", "wa");
    fprintf(fp,"mu phi sigma rho\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %.4f\n",musimul[i],phisimul[i],sigmasimul[i],rhosimul[i]);
    fclose(fp);
    //
    cudaFree(musimul);
    cudaFree(phisimul);
    cudaFree(sigmasimul);
    cudaFree(rhosimul);
    cudaFree(seed);
    cudaFree(x);
    cudaDeviceReset();
    //
    free(xi);
    printf("Done ... \n");
}
//
//
void estimate_svtl_gpu(const char *file)
{
    printf("Start estimating SVTL .... \n");
    int n;
    float *xi = readArray<float>(file,&n);
    //
    float *x;
    cudaMallocManaged(&x,n*sizeof(float));
    for(int i=0;i<n;i++) x[i] = xi[i];
    //
    int niter = 5000;
    float *musimul;
    float *phisimul;
    float *sigmasimul;
    float *rhosimul;
    int *nusimul;
    unsigned int *seed;
    //
    cudaMallocManaged(&musimul,niter*sizeof(float));
    cudaMallocManaged(&phisimul,niter*sizeof(float));
    cudaMallocManaged(&sigmasimul,niter*sizeof(float));
    cudaMallocManaged(&rhosimul,niter*sizeof(float));
    cudaMallocManaged(&nusimul,niter*sizeof(int));
    cudaMallocManaged(&seed,niter*sizeof(unsigned int));
    //
    srand(time(NULL));
    for(int i=0;i<niter;i++) seed[i] = rand();
    //
    double time_spent = 0.0;
    clock_t begin = clock();
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,10485760000L);
    kernel_svtl<<<1024,8>>>(x,n,seed,musimul,phisimul,sigmasimul,rhosimul,nusimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    float mrho = Vector<float>(rhosimul,niter).mean();
    //
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC; 
    printf("The elapsed time is %f seconds\n", time_spent);
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f; rho: %.4f\n",mmu,mphi,msigma,mrho);
    //
    FILE *fp;
    fp = fopen("svtlestimgpu.txt", "wa");
    fprintf(fp,"mu phi sigma rho nu\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %.4f %d\n",musimul[i],phisimul[i],sigmasimul[i],rhosimul[i],nusimul[i]);
    fclose(fp);
    //
    cudaFree(musimul);
    cudaFree(phisimul);
    cudaFree(sigmasimul);
    cudaFree(rhosimul);
    cudaFree(nusimul);
    cudaFree(seed);
    cudaFree(x);
    cudaDeviceReset();
    //
    free(xi);
    printf("Done ... \n");
}
//
void estimate_svt_gpu(const char *file)
{
    printf("Start estimating SVT .... \n");
    int n;
    float *xi = readArray<float>(file,&n);
    //
    float *x;
    cudaMallocManaged(&x,n*sizeof(float));
    for(int i=0;i<n;i++) x[i] = xi[i];
    //
    int niter = 5000;
    float *musimul;
    float *phisimul;
    float *sigmasimul;
    int *nusimul;
    unsigned int *seed;
    //
    cudaMallocManaged(&musimul,niter*sizeof(float));
    cudaMallocManaged(&phisimul,niter*sizeof(float));
    cudaMallocManaged(&sigmasimul,niter*sizeof(float));
    cudaMallocManaged(&nusimul,niter*sizeof(int));
    cudaMallocManaged(&seed,niter*sizeof(unsigned int));
    //
    srand(time(NULL));
    for(int i=0;i<niter;i++) seed[i] = rand();
    //
    double time_spent = 0.0;
    clock_t begin = clock();
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,10485760000L);
    kernel_svt<<<1024,8>>>(x,n,seed,musimul,phisimul,sigmasimul,nusimul,niter);
    cudaDeviceSynchronize();
    //
    float mmu = Vector<float>(musimul,niter).mean();
    float mphi = Vector<float>(phisimul,niter).mean();
    float msigma = Vector<float>(sigmasimul,niter).mean();
    //
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC; 
    printf("The elapsed time is %f seconds\n", time_spent);
    //
    printf("mu: %.4f; phi: %.4f; sigma: %.4f\n",mmu,mphi,msigma);
    //
    FILE *fp;
    fp = fopen("svtestimgpu.txt", "wa");
    fprintf(fp,"mu phi sigma rho nu\n");
    for(int i=0;i<niter;i++) fprintf(fp,"%.4f %.4f %.4f %d\n",musimul[i],phisimul[i],sigmasimul[i],nusimul[i]);
    fclose(fp);
    //
    cudaFree(musimul);
    cudaFree(phisimul);
    cudaFree(sigmasimul);
    cudaFree(nusimul);
    cudaFree(seed);
    cudaFree(x);
    cudaDeviceReset();
    //
    free(xi);
    printf("Done ... \n");
}
//
#endif
