#ifndef ar1model_cuh
#define ar1model_cuh


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"

/*
    The model 
    y(t) = mu + phi*(y(t-1) - mu) + sigma*err

*/

template <typename T>
class AR1Model{
protected:
    //
    T *x;
    unsigned int n;
    T mu;
    T phi;
    T sigma;
    //
    //
    bool mudiffuse;
    bool phidiffuse;
    bool sigmadiffuse;
    //
    int phipriortype; // 0 normal; 1 beta
    //
    T muprior[2];
    T phiprior[2];
    T sigmaprior[2];
    //
    unsigned int seed;
    Random<T> *random;
    //
    T *x2;
    T *x1;
    unsigned int m;
    //
    //
public:
    __host__ __device__ AR1Model(T *xx,unsigned int n,T mu,T phi,T sigma)
    {
        this->x = xx;
        this->n = n;
        this->m = n - 1;
        this->x2 = new T[m];
        this->x1 = new T[m];
        for(int i=0;i<m;i++){
            this->x1[i] = this->x[i];
            this->x2[i] = this->x[i+1];
        }
        this->mu = mu;
        this->phi = phi;
        this->sigma = sigma;
        //
        this->mudiffuse = false;
        this->phidiffuse = false;
        this->sigmadiffuse = false;
        //
        this->phipriortype = 1;
        this->muprior[0] = 0.0; this->muprior[1] = 10.0;
        this->phiprior[0] = 25.0; this->phiprior[1] = 2.5;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->seed = 9796969;
        //
        this->random = new Random<T>(this->seed);
    }
    //
    //
    __host__ __device__ ~AR1Model()
    {
        if(random)
        {
            delete random;
        }
        if(x1){
            delete[] x1;
        }
        if(x2){
            delete[] x2;
        }
    }
    //
    //
    __host__ __device__ T simulatemu()
    {
        T *err = new T[this->m];
        for(int i=0;i<this->m;i++){
            err[i] = (this->x2[i] - this->phi*this->x1[i])/(1.0 - this->phi);
        }
        NormalModel<T> *nm = new NormalModel<T>(err,this->m,this->mu,this->sigma/(1.0 - this->phi));
        nm->setseed(this->random->rand());
        nm->setmudiffuse(this->mudiffuse);
        nm->setmuprior(this->muprior);
        this->mu = nm->simulatemu();
        delete[] err;
        delete nm;
        return this->mu;
    }
    //
    //
    __host__ __device__ T simulatephi()
    {
        T *err2 = new T[this->m];
        T *err1 = new T[this->m];
        for(int i=0;i<this->m;i++){
            err2[i] = this->x2[i] - this->mu;
            err1[i] = this->x1[i] - this->mu;
        }
        RegModel<T> *reg = new RegModel<T>(err2,err1,this->m,0.0,this->phi,this->sigma);
        reg->setseed(this->random->rand());
        reg->setbetadiffuse(this->phidiffuse);
        reg->setbetapriortype(this->phipriortype);
        reg->setbetaprior(this->phiprior);
        this->phi = reg->simulatebeta();
        delete[] err2;
        delete[] err1;
        delete reg;
        return this->phi;
    }
    //
    //
    __host__ __device__ T simulatesigma()
    {
        T *err = new T[this->m];
        for(int i=0;i<this->m;i++){
            err[i] = this->x2[i] - this->mu - this->phi*(this->x1[i] - this->mu);
        }
        NormalModel<T> *nm = new NormalModel<T>(err,this->m,0.0,this->sigma);
        nm->setseed(this->random->rand());
        nm->setsigmadiffuse(this->sigmadiffuse);
        nm->setsigmaprior(this->sigmaprior);
        this->sigma = nm->simulatesigma();
        delete[] err;
        delete nm;
        return this->sigma;
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
