#ifndef normalmodel_cuh
#define normalmodel_cuh

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "vector.cuh"
#include "random.cuh"

template <typename T>
class NormalModel{
protected:
    T *x;
    int n;
    //
    T mu;
    T sigma;
    //
    bool mudiffuse;
    bool sigmadiffuse;
    //
    int mupriortype;
    //
    T muprior[2];
    T sigmaprior[2];
    //
    unsigned int seed;
    Random<T> *random;
    //
    //
public:
    //
    //
    __host__ __device__ NormalModel(T *xx,int n,T mu,T sigma)
    {
        this->x = xx;
        this->n = n;
        this->mu = mu;
        this->sigma = sigma;
        this->mudiffuse = false;
        this->sigmadiffuse = false;
        this->mupriortype = 0;
        this->muprior[0] = 0.0; this->muprior[1] = 10.0;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->seed = 9879870;
        this->random = new Random<T>(seed);
    }
    //
    __host__ __device__ ~NormalModel()
    {
        if(random){
            delete random;
        }
    }
    //
    __host__ __device__ T simulatemu()
    {
        if(!this->mudiffuse){
            T mm = Vector<T>(this->x,this->n).mean();
            T std = this->sigma/sqrt((T)this->n);
            this->mu = mm +  std*this->random->normal();
            return this->mu;
        }else{
            T mm = Vector<T>(this->x,this->n).mean();
            T std = this->sigma/sqrt((T)this->n);
            T mpost = (this->muprior[0]*std*std + mm*this->muprior[1]*this->muprior[1])/(std*std + this->muprior[1]*this->muprior[1]);
            T spost = sqrt((std*std*this->muprior[1]*this->muprior[1])/(std*std + this->muprior[1]*this->muprior[1]));
            this->mu = mpost +  spost*this->random->normal();
            return this->mu;
        }
    }
    //
    //
    __host__ __device__ T simulatesigma()
    {
        if(!this->sigmadiffuse){
            T *err = new T[n];
            for(int i=0;i<this->n;i++) err[i] = this->x[i] - this->mu;
            T rss = Vector<T>(err,this->n).sumsq();
            delete[] err;
            this->sigma = 1.0/sqrt(this->random->gamma(0.5*(T)this->n, 0.5*(T)rss));
            return this->sigma;
        }else{
            T *err = new T[n];
            for(int i=0;i<this->n;i++) err[i] = this->x[i] - this->mu;
            T rss = Vector<T>(err,this->n).sumsq();
            delete[] err;
            this->sigma = 1.0/sqrt(this->random->gamma(0.5*(T)n + this->sigmaprior[0], 0.5*(T)rss + this->sigmaprior[1]));
            return this->sigma;
        }
    }
    //
    //
    __host__ __device__ T getmu()
    {
        return this->mu;
    }
    //
    //
    __host__ __device__ T getsigma()
    {
        return this->sigma;
    }
    //
    //
    __host__ __device__ void setmuprior(T *mprior)
    {
        this->muprior[0] = mprior[0];
        this->muprior[1] = mprior[1];
    }
    //
    //
    __host__ __device__ void setsigmaprior(T *sprior)
    {
        this->sigmaprior[0] = sprior[0];
        this->sigmaprior[1] = sprior[1];
    }
    //
    //
    __host__ __device__ void setmudiffuse(bool mdiffuse)
    {
        this->mudiffuse = mdiffuse;
    }
    //
    //
    __host__ __device__ void setsigmadiffuse(bool sdiffuse)
    {
        this->sigmadiffuse = sdiffuse;
    }
    //
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
