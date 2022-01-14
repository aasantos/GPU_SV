#ifndef dlmmodel_cuh
#define dlmmodel_cuh


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"
#include "ar1model.cuh"


/*
model 
y(t) = alpha(t) + sigmav*err
alpha(t+1) = mu + phi*(alpha(t) - mu) + sigma+err
*/

template <typename T>
class DLMModel{
protected:
    //
    //
    T *x;
    unsigned int n;
    T *alpha;
    T sigmav;
    T mu;
    T phi;
    T sigma;
    T a0;
    T a1;
    //
    //
    bool sigmavdiffuse;
    bool mudiffuse;
    bool phidiffuse;
    bool sigmadiffuse;
    //
    int phipriortype; // 0 normal; 1 beta
    //
    T sigmavprior[2];
    T muprior[2];
    T phiprior[2];
    T sigmaprior[2];
    //
    unsigned int seed;
    Random<T> *random;
    //
    //
public:
    __host__ __device__ DLMModel(T *x,unsigned int n,T sigmav,T mu,T phi,T sigma)
    {
        this->x = x;
        this->n = n;
        this->sigmav = sigmav;
        this->mu = mu;
        this->phi = phi;
        this->sigma = sigma;
        this->sigmavdiffuse = false;
        this->mudiffuse = false;
        this->phidiffuse = false;
        this->sigmadiffuse = false;
        this->phipriortype = 1;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->muprior[0] = 0.0; this->muprior[1] = 10.0;
        this->phiprior[0] = 25.0; this->phiprior[1] = 2.5;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->seed = 9796969;
        this-> alpha = new T[n];
        for(int i=0;i<this->n;i++) this->alpha[i] = 0.0;
        this->a0 = 0.0;
        this->a1 = 0.0;
        this->random = new Random<T>(this->seed);
    }
    //
    __host__ __device__ ~DLMModel()
    {
        if(random){
            delete random;
        }
        if(alpha){
            delete[] alpha;
        }
    }
    //
    //
    __host__ __device__ void simulatestates()
    {
        T t1 = this->sigmav*this->sigmav*(1.0 + this->phi*this->phi) +
        this->sigma*this->sigma;
        T sx = this->sigmav*this->sigma/sqrt(t1);
        for(int i=0;i<this->n;i++){
            if(i==0){
                this->a0 = random->normal(this->mu,this->sigma/(1.0 - this->phi));
                this->a1 = this->alpha[i+1];
                T t2 = this->sigmav*this->sigmav*(this->phi*(this->a0 + this->a1) +
                             this->mu*(1.0 -  this->phi)*(1.0 - this->phi))
                             + this->sigma*this->sigma*this->x[i];
                T mx = t2/t1;
                this->alpha[i] = random->normal(mx, sx);
            }else if(i==(n-1)){
                    this->a0 = this->alpha[i-1];
                    this->a1 = this->mu + this->phi*(this->alpha[i] - this->phi)
                                + random->normal(0.0,this->sigma);
                    T t2 = this->sigmav*this->sigmav*(this->phi*(this->a0 + this->a1) +
                             this->mu*(1.0 -  this->phi)*(1.0 - this->phi))
                             + this->sigma*this->sigma*this->x[i];
                    T mx = t2/t1;
                    this->alpha[i] = random->normal(mx, sx);
            }else{
                this->a0 = this->alpha[i-1];
                this->a1 = this->alpha[i+1];
                T t2 = this->sigmav*this->sigmav*(this->phi*(this->a0 + this->a1) +
                             this->mu*(1.0 -  this->phi)*(1.0 - this->phi))
                             + this->sigma*this->sigma*this->x[i];
                T mx = t2/t1;
                this->alpha[i] = random->normal(mx, sx);
            }
        }
    }
    //
    __host__ __device__ T simulatesigmav()
    {
        T *err = new T[this->n];
        for(int i=0;i<this->n;i++) err[i] = this->x[i] - this->alpha[i];
        NormalModel<T> *nm = new NormalModel<T>(err,this->n,0.0,this->sigmav);
        nm->setseed(this->random->rand());
        nm->setsigmadiffuse(this->sigmavdiffuse);
        nm->setsigmaprior(this->sigmavprior);
        this->sigmav = nm->simulatesigma();
        delete[] err;
        delete nm;
        return this->sigmav;
    }
    //
    __host__ __device__ T simulatemu()
    {
        AR1Model<T> *ar1 = new AR1Model<T>(this->alpha,this->n,this->mu,this->phi,this->sigma);
        ar1->setseed(this->random->rand());
        ar1->setmudiffuse(this->mudiffuse);
        ar1->setmuprior(this->muprior);
        this->mu = ar1->simulatemu();
        delete ar1;
        return this->mu;
    }
    //
    //
    __host__ __device__ T simulatephi()
    {
        AR1Model<T> *ar1 = new AR1Model<T>(this->alpha,this->n,this->mu,this->phi,this->sigma);
        ar1->setseed(this->random->rand());
        ar1->setphidiffuse(this->phidiffuse);
        ar1->setphipriortype(this->phipriortype);
        ar1->setphiprior(this->phiprior);
        this->phi = ar1->simulatephi();
        delete ar1;
        return this->phi;
    }
    //
    //
    __host__ __device__ T simulatesigma()
    {
        AR1Model<T> *ar1 = new AR1Model<T>(this->alpha,this->n,this->mu,this->phi,this->sigma);
        ar1->setseed(this->random->rand());
        ar1->setsigmadiffuse(this->sigmadiffuse);
        ar1->setsigmaprior(this->sigmaprior);
        this->sigma = ar1->simulatesigma();
        delete ar1;
        return this->sigma;
    }
    //
    __host__ __device__ void setsigmavdiffuse(bool svdiffuse)
    {
        this->sigmavdiffuse = svdiffuse;
    }
    //
    __host__ __device__ void setmudiffuse(bool mdiffuse)
    {
        this->mudiffuse = mdiffuse;
    }
    //
    __host__ __device__ void setphidiffuse(bool pdiffuse)
    {
        this->phidiffuse = pdiffuse;
    }
    //
    __host__ __device__ void setsigmadiffuse(bool sdiffuse)
    {
        this->sigmadiffuse = sdiffuse;
    }
    //
    __host__ __device__ void setmuprior(T mprior[2])
    {
        this->muprior[0] = mprior[0];
        this->muprior[1] = mprior[1];
    }
    //
    __host__ __device__ void setphiprior(T pprior[2])
    {
        this->phiprior[0] = pprior[0];
        this->phiprior[1] = pprior[1];
    }
    //
    __host__ __device__ void setsigmaprior(T sprior[2])
    {
        this->sigmaprior[0] = sprior[0];
        this->sigmaprior[1] = sprior[1];
    }
    //
    __host__ __device__ void setsigmavprior(T svprior[2])
    {
        this->sigmavprior[0] = svprior[0];
        this->sigmavprior[1] = svprior[1];
    }
    //
    __host__ __device__ void setphipriortype(int t)
    {
        this->phipriortype = t;
    }
    //
    __host__ __device__ void setseed(unsigned int ss)
    {
        if(random){
            delete random;
        }
        this->seed = ss;
        this->random = new Random<T>(seed);
    }
};

#endif
