#ifndef regmodel_cuh
#define regmodel_cuh


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"

/*
the model to estimate is given by
  y = alpha + beta*x + sigma*err
  where err ~ N(0,1)
*/

template <typename T>
class RegModel{
protected:
    //
    T *y;
    T *x;
    int n;
    //
    T alpha;
    T beta;
    T sigma;
    //
    //
    bool alphadiffuse ;
    bool betadiffuse;
    bool sigmadiffuse;
    //
    int betapriortype;  //0 normal; 1 beta
    //
    T alphaprior[2];
    T betaprior[2];
    T sigmaprior[2];
    //
    unsigned int seed;
    Random<T> *random;
    //
    //
public:
    __host__ __device__ RegModel(T *yy,T *xx,int n,T alpha,T beta,T sigma)
    {
        this->y = yy;
        this->x = xx;
        this->n = n;
        this->alpha = alpha;
        this->beta = beta;
        this->sigma = sigma;
        //
        this->alphadiffuse = false;
        this->betadiffuse = false;
        this->sigmadiffuse = false;
        //
        this->betapriortype = 0;
        //
        this->alphaprior[0] = 0.0; this->alphaprior[1] = 10.0;
        this->betaprior[0] = 0.0; this->betaprior[1] = 10.0;
        this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        //
        this->seed = 9796967;
        //
        this->random = new Random<T>(this->seed);
    }
    //
    __host__ __device__ ~RegModel()
    {
        if(random)
        {
            delete random;
        }
    }
    //
    //
    __host__ __device__ T simulatealpha()
    {
        T *err = new T[this->n];
        for(int i=0;i<this->n;i++) err[i] = this->y[i] - this->beta*x[i];
        NormalModel<T> *nm = new NormalModel<T>(err,this->n,this->alpha,this->sigma);
        nm->setseed(this->random->rand());
        nm->setmudiffuse(this->alphadiffuse);
        nm->setmuprior(this->alphaprior);
        this->alpha = nm->simulatemu();
        delete[] err;
        delete nm;
        return this->alpha;
    }
    //
    //
    __host__ __device__ T simulatebeta()
    {
        if(!this->betadiffuse){
            if(this->betapriortype == 0){
                T *ystar = new T[this->n];
                for(int i=0;i<this->n;i++) ystar[i] = this->y[i] - this->alpha;
                T num = 0.0;
                T den = 0.0;
                for(int i=0;i<n;i++){
                    num += ystar[i]*this->x[i];
                    den += this->x[i]*this->x[i];
                }
                T betahat = num/den;
                T sigmahat = this->sigma/sqrt(den);
                T mpost = (betahat*this->betaprior[1]*this->betaprior[1] + this->betaprior[0]*sigmahat*sigmahat)/(sigmahat*sigmahat + this->betaprior[1]*this->betaprior[1]);
                T spost = sqrt((this->betaprior[1]*this->betaprior[1]*sigmahat*sigmahat)/(sigmahat*sigmahat + this->betaprior[1]*this->betaprior[1]));
                this->beta = mpost + spost*this->random->normal();
                delete[] ystar;
                return this->beta;
            }else{
                T *ystar = new T[this->n];
                for(int i=0;i<this->n;i++) ystar[i] = this->y[i] - this->alpha;
                T num = 0.0;
                T den = 0.0;
                for(int i=0;i<this->n;i++){
                    num += ystar[i]*this->x[i];
                    den += this->x[i]*this->x[i];
                }
                T betahat = num/den;
                T sigmahat = this->sigma/sqrt(den);
                delete[] ystar;
                this->beta = this->random->normalbeta(betahat, sigmahat, this->betaprior[0], this->betaprior[1]);
                return this->beta;
            }
        }else{
            T *ystar = new T[this->n];
            for(int i=0;i<this->n;i++) ystar[i] = this->y[i] - this->alpha;
            T num = 0.0;
            T den = 0.0;
            for(int i=0;i<this->n;i++){
                num += ystar[i]*this->x[i];
                den += this->x[i]*this->x[i];
            }
            T betahat = num/den;
            T sigmahat = this->sigma/sqrt(den);
            this->beta = betahat + sigmahat*this->random->normal();
            delete[] ystar;
            return this->beta;
        }
        
    }
    //
    //
    __host__ __device__ T simulatesigma()
    {
        T *err = new T[this->n];
        for(int i=0;i<n;i++) err[i] = this->y[i] - this->beta*this->x[i];
        NormalModel<T> *nm = new NormalModel<T>(err,this->n,this->alpha,this->sigma);
        nm->setseed(this->random->rand());
        nm->setmudiffuse(this->sigmadiffuse);
        nm->setsigmaprior(this->sigmaprior);
        this->sigma = nm->simulatesigma();
        delete nm;
        delete[] err;
        return this->sigma;
    }
    //
    __host__ __device__ void setalphadiffuse(bool adiffuse)
    {
        this->alphadiffuse = adiffuse;
    }
    //
    __host__ __device__ void setbetadiffuse(bool bdiffuse)
    {
        this->betadiffuse = bdiffuse;
    }
    //
    __host__ __device__ void setsigmadiffuse(bool sdiffuse)
    {
        this->sigmadiffuse = sdiffuse;
    }
    //
    __host__ __device__ void setalphaprior(T aprior[2])
    {
        this->alphaprior[0] = aprior[0];
        this->alphaprior[1] = aprior[1];
    }
    //
    __host__ __device__ void setbetaprior(T bprior[2])
    {
        this->betaprior[0] = bprior[0];
        this->betaprior[1] = bprior[1];
    }
    //
    __host__ __device__ void setsigmaprior(T sprior[2])
    {
        this->sigmaprior[0] = sprior[0];
        this->sigmaprior[1] = sprior[1];
    }
    //
    __host__ __device__ void setbetapriortype(int t)
    {
        this->betapriortype = t;
    }
    //
    __host__ __device__ T getalpha()
    {
        return alpha;
    }
    //
    __host__ __device__ T getbeta()
    {
        return beta;
    }
    //
    __host__ __device__ T getsigma()
    {
        return sigma;
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
