#ifndef svl_cuh
#define svl_cuh

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


template <typename T>
class SVLModel{
protected:
    //
    T *x;
    unsigned int n;
    T *alpha;
    T sigmav;
    T mu;
    T phi;
    T sigma;
    T rho;
    //
    T a0;
    T a1;
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
public:
    __host__ __device__ SVLModel(T *x,int n,T mu,T phi,T sigma,T rho)
    {
        this->x = x;
        this->n = n;
        this->mu = mu;
        this->phi = phi;
        this->sigma = sigma;
        this->rho = rho;
        this-> alpha = new T[n];
        for(int i=0;i<this->n;i++) this->alpha[i] = 0.0;
        this->a0 = 0.0; this->a1 = 0.0;
        this->mudiffuse = false;
        this->phidiffuse = false;
        this->sigmadiffuse = false;
        this->phipriortype = 1;
        this->muprior[0] = 0.0; this->muprior[1] = 10.0;
        this->phiprior[0] = 25.0; this->phiprior[1] = 2.5;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->seed = 79798;
        this->random = new Random<T>(this->seed);
    }
    //
    __host__ __device__ ~SVLModel()
    {
        if(random){
            delete random;
        }
        if(alpha){
            delete[] alpha;
        }
    }
    //
    __host__ __device__ T df(T yy,T yy0,T a0,T a,T a1)
    {
        T t1 = -0.5e0 + 0.5e0 * yy * yy * exp(-a);
        T t2 = -(a1 - this->mu - this->phi * (a - this->mu) - this->sigma * this->rho * exp(-0.5e0 * a) * yy) * 
            pow(this->sigma, -0.2e1) / (-this->rho * this->rho + 0.1e1) * (-this->phi + 0.5e0 * this->sigma * this->rho * exp(-0.5e0 * a) * yy);
        T t3 = -(a - this->mu - this->phi * (a0 - this->mu) - this->sigma * this->rho * exp(-0.5e0 * a0) * yy0) * 
            pow(this->sigma, -0.2e1) / (-this->rho * this->rho + 0.1e1);
        return t1 + t2 + t3;
    }
    //
    __host__ __device__ T ddf(T yy,T a)
    {
        T t1 = -0.5e0 * yy * yy * exp(-a);
        T t2 = -pow(-this->phi + 0.5e0 * this->sigma * this->rho * exp(-0.5e0 * a) * yy, 0.2e1) * 
            pow(this->sigma, -0.2e1) / (-this->rho * this->rho + 0.1e1) + 0.25e0 * (a1 - this->mu - this->phi * 
            (a - this->mu) - this->sigma * this->rho * exp(-0.5e0 * a) * yy) / this->sigma / 
            (-this->rho * this->rho + 0.1e1) * this->rho * exp(-0.5e0 * a) * yy;
        T t3 = -pow(this->sigma, -0.2e1) / (double) (-this->rho * this->rho + 1.0);
        return t1 + t2 + t3;
    }
    //
    __host__ __device__ T newton(T yy,T yy0,T a0,T a1)
    {
        T x0 = 0.5*(a0 + a1);
        T g = df(yy,yy0,a0,x0,a1);
        int iter = 0;
        while(abs(g) > 0.00001 && iter < 20){
            T h = ddf(yy,x0);
            x0 = x0 - g/h;
            g = df(yy,yy0,a0,x0,a1);
            iter++;
        }
        return x0;
    }
    //
    __host__ __device__ T meanstate(T yy,T yy0,T a0,T a1)
    {
        return newton(yy,yy0,a0,a1);
    }
        //
    __host__ __device__ T stdstate(T yy,T a)
    {
        return sqrt(-1.0/ddf(yy,a));
    }
    //
    __host__ __device__ T loglik(T yy,T yy0,T a0,T a,T a1)
    {
        T t1 = -0.5e0 * a - 0.5e0 * yy * yy * exp(-a);
        T t2 = -pow(a1 - this->mu - this->phi * (a - this->mu) - this->sigma * this->rho * exp(-0.5e0 * a) * yy, 0.2e1) 
            * pow(this->sigma, -0.2e1) / (-this->rho * this->rho + 0.1e1) / 0.2e1;
        T t3 = -pow(a - this->mu - this->phi * (a0 - this->mu) - this->sigma * this->rho * exp(-0.5e0 * a0) 
            * yy0, 0.2e1) * pow(this->sigma, -0.2e1) / (-this->rho * this->rho + 0.1e1) / 0.2e1;
        return t1 + t2 + t3;
    }
        //
    __host__ __device__ T lognorm(T a,T m,T s)
    {
        T err = (a - m);
        return -0.5*err*err/(s*s);
    }
    //
    //
    __host__ __device__ T metroprob(T anew,T a,T a0,T a1,T yy,T yy0,T m,T s)
    {
        T l1 = loglik(yy,yy0,a0,anew,a1);
        T l0 = loglik(yy,yy0,a0,a,a1);
        T g0 = lognorm(a,m,s);
        T g1 = lognorm(anew,m,s);
        T tt = l1 - l0 + g0 - g1;
        if(tt > 0.0) return 1.0;
        if(tt < -10.0){
            return 0.0;
        }else{
            return exp(tt);
        }
    }
    //
    __host__ __device__ void simulatestates()
    {
        for(int i=0;i<this->n;i++){
            T yy = this->x[i];
            if(i==0){
                this->a0 = this->random->normal(this->mu,this->sigma/sqrt(1.0 - this->phi*this->phi));
                this->a1 = this->alpha[i+1];
                T mx = meanstate(yy,a0,a1);
                T sx = stdstate(yy,mx);
                T atemp = this->random->normal(mx, sx);
                if(this->random->uniform() < metroprob(atemp,this->alpha[i],a0,a1,yy,mx,sx)){
                   this->alpha[i] = atemp;
                }
            }else if(i==(n-1)){
                  this->a0 = this->alpha[i-1];
                  this->a1 = this->mu + this->phi*(this->alpha[i] - this->mu)
                                + this->random->normal(0.0,this->sigma);
                  T mx = meanstate(yy,a0,a1);
                  T sx = stdstate(yy,mx);
                  T atemp = this->random->normal(mx, sx);
                  if(this->random->uniform() < metroprob(atemp,this->alpha[i],a0,a1,yy,mx,sx)){
                    this->alpha[i] = atemp;
                  }
            }else{
               this->a0 = this->alpha[i-1];
               this->a1 = this->alpha[i+1];
               T mx = meanstate(yy,a0,a1);
               T sx = stdstate(yy,mx);
               T atemp = this->random->normal(mx, sx);
               if(this->random->uniform() < metroprob(atemp,this->alpha[i],a0,a1,yy,mx,sx)){
                  this->alpha[i] = atemp;
                }
            }
        }
    }
    //
    __host__ __device__ T simulatemu()
    {
        
        return this->mu;
    }
    //
    //
    __host__ __device__ T simulatephi()
    {
        return this->phi;
    }
    //
    //
    __host__ __device__ T simulatesigma()
    {
        return this->sigma;
    }
    //
    //
    __host__ __device__ void setmudiffuse(bool mdiffuse)
    {
        this->mudiffuse = mdiffuse;
    }
    //
    //
    __host__ __device__ void setphidiffuse(bool pdiffuse)
    {
        this->phidiffuse = pdiffuse;
    }
    //
    //
    __host__ __device__ void setsigmadiffuse(bool sdiffuse)
    {
        this->sigmadiffuse = sdiffuse;
    }
    //
    //
    __host__ __device__ void setmuprior(T mprior[2])
    {
        this->muprior[0] = mprior[0];
        this->muprior[1] = mprior[1];
    }
    //
    //
    __host__ __device__ void setphiprior(T pprior[2])
    {
        this->phiprior[0] = pprior[0];
        this->phiprior[1] = pprior[1];
    }
    //
    //
    __host__ __device__ void setsigmaprior(T sprior[2])
    {
        this->sigmaprior[0] = sprior[0];
        this->sigmaprior[1] = sprior[1];
    }
    //
    //
    __host__ __device__ void setphipriortype(int t)
    {
        this->phipriortype = t;
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
