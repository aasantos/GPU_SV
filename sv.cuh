#ifndef sv_cuh
#define sv_cuh

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "Vector.cuh"
#include "Random.cuh"
#include "Stats.cuh"
#include "NormalModel.cuh"
#include "RegModel.cuh"
#include "AR1Model.cuh"
//
//
//
template <typename T>
class SVModel{
protected:
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
    SVModel(T *x,int n,T mu,T phi,T sigma)
    {
        this->x = x;
        this->n = n;
        this->mu = mu;
        this->phi = phi;
        this->sigma = sigma;
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
    ~SVModel()
    {
        if(random){
            delete random;
        }
        if(alpha){
            delete[] alpha;
        }
    }
    //
    T df(T yy,T a0,T a,T a1)
    {
        T sigmasq = this->sigma*this->sigma;
        T err1 = a - this->mu - this->phi*(a0 - this->mu);
        T err2 = a1 - this->mu - this->phi*(a - this->mu);
        return -0.5 + 0.5*yy*yy*exp(-1.0*a) - err1/sigmasq + this->phi*err2/sigmasq;
    }
    //
    T ddf(T yy,T a)
    {
        T sigmasq = this->sigma*this->sigma;
        return -0.5*yy*yy/exp(a) - (1.0 + this->phi*this->phi)/sigmasq;
    }
    //
    T newton(T yy,T a0,T a1)
    {
        T x0 = 0.5*(a0 + a1);
        T g = df(yy,a0,x0,a1);
        int iter = 0;
        while(abs(g) > 0.00001 && iter < 20){
            T h = ddf(yy,x0);
            x0 = x0 - g/h;
            g = df(yy,a0,x0,a1);
            iter++;
        }
        return x0;
    }
    //
    T meanstate(T yy,T a0,T a1)
    {
        return newton(yy,a0,a1);
    }
        //
    T stdstate(T yy,T a)
    {
        return sqrt(-1.0/ddf(yy,a));
    }
    //
    T loglik(T yy,T a0,T a,T a1)
    {
        T sigmasq = this->sigma*this->sigma;
        T t1 = -0.5*a - 0.5*yy*yy/exp(a);
        T err2 = (a1 - this->mu - this->phi*(a - this->mu));
        T t2 = -0.5*err2*err2/sigmasq;
        T err3 = (a - this->mu - this->phi*(a0 - this->mu));
        T t3 = -0.5*err3*err3/sigmasq;
        return t1 + t2 + t3;
    }
        //
    T lognorm(T a,T m,T s)
    {
        T err = (a - m);
        return -0.5*err*err/(s*s);
    }
    //
    //
    T metroprob(T anew,T a,T a0,T a1,T yy,T m,T s)
    {
        T l1 = loglik(yy,a0,anew,a1);
        T l0 = loglik(yy,a0,a,a1);
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
    void simulatestates()
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
    T simulatemu()
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
    T simulatephi()
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
    T simulatesigma()
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
    //
    void setmudiffuse(bool mdiffuse)
    {
        this->mudiffuse = mdiffuse;
    }
    //
    //
    void setphidiffuse(bool pdiffuse)
    {
        this->phidiffuse = pdiffuse;
    }
    //
    //
    void setsigmadiffuse(bool sdiffuse)
    {
        this->sigmadiffuse = sdiffuse;
    }
    //
    //
    void setmuprior(T mprior[2])
    {
        this->muprior[0] = mprior[0];
        this->muprior[1] = mprior[1];
    }
    //
    //
    void setphiprior(T pprior[2])
    {
        this->phiprior[0] = pprior[0];
        this->phiprior[1] = pprior[1];
    }
    //
    //
    void setsigmaprior(T sprior[2])
    {
        this->sigmaprior[0] = sprior[0];
        this->sigmaprior[1] = sprior[1];
    }
    //
    //
    void setphipriortype(int t)
    {
        this->phipriortype = t;
    }
    //
    //
    void setseed(unsigned int ss)
    {
        if(random){
            delete random;
        }
        this->seed = ss;
        this->random = new Random<T>(seed);
    }
};

#endif
