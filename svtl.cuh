#ifndef svtl_cuh
#define svtl_cuh

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "vector.cuh"
#include "stats.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"
#include "ar1model.cuh"
//
//
template <typename T>
class SVTLModel{
protected:
    //
    unsigned int n;
    T *x;
    T *lambda;
    T *alpha;
    //
    T mu;
    T phi;
    T sigma;
    T rho;
    int nu;
    //
    T a0;
    T a1;
    //
    //
    bool mudiffuse;
    bool phidiffuse;
    //bool sigmadiffuse;
    //
    int phipriortype; // 0 normal; 1 beta
    //
    T muprior[2];
    T phiprior[2];
    //T sigmaprior[2];
    //
    unsigned int seed;
    Random<T> *random;
    //
    //
public:
    __host__ __device__ SVTLModel(T *x,int n,T mu,T phi,T sigma,T rho,int nu)
    {
        this->n = n;
        this->mu = mu;
        this->phi = phi;
        this->sigma = sigma;
        this->rho = rho;
        this->nu = nu;
        //
        this->x = x;
        this->lambda = new T[n];
        this->alpha = new T[n];
        //
        for(int i=0;i<this->n;i++){
            this->lambda[i] = 1.0;
            this->alpha[i] = 0.0;
        }
        this->a0 = 0.0; this->a1 = 0.0;
        this->mudiffuse = false;
        this->phidiffuse = false;
        //this->sigmadiffuse = false;
        this->phipriortype = 1;
        this->muprior[0] = 0.0; this->muprior[1] = 10.0;
        this->phiprior[0] = 25.0; this->phiprior[1] = 2.5;
        //this->sigmaprior[0] = 2.5; this->sigmaprior[1] = 0.025;
        this->seed = 79798;
        this->random = new Random<T>(this->seed);
    }
    //
    __host__ __device__ ~SVTLModel()
    {
        if(random){
            delete random;
        }
        if(alpha){
            delete[] alpha;
        }
        if(lambda){
            delete[] lambda;
        }
    }
    //
    __host__ __device__ T df(T yy,T yy0,T a0,T a,T a1)
    {
        T sigmat = this->sigma*sqrtf(1.0 - this->rho*this->rho);
        T sigmasq = sigmat*sigmat;
        T tt = this->sigma*this->rho*expf(-0.5*a)*yy;
        T t1 = -0.5 + 0.5*yy*yy*expf(-1.0*a);
        T err0 = 0.5*this->sigma*tt - this->phi;
        T err1 = (a1 - this->mu - this->phi*(a - this->mu) - tt);
        T t2 = -1.0*err0*err1/sigmasq;
        T err2 = (a - this->mu - this->phi*(a0 - this->mu) - this->sigma*this->rho*expf(-0.5*a0)*yy0);
        T t3 = -1.0*err2/sigmasq;
        return t1 + t2 + t3;
    }
    //
    __host__ __device__ T ddf(T yy,T a,T a1)
    {
        T sigmat = this->sigma*sqrtf(1.0 - this->rho*this->rho);
        T sigmasq = sigmat*sigmat;
        T tt = this->rho*expf(-0.5*a)*yy;
        T t1 = -0.5*yy*yy*expf(-1.0*a);
        T err0 = 0.5*this->sigma*tt - this->phi;
        T err1 = (a1 - this->mu - this->phi*(a - this->mu) - this->sigma*tt);
        T t2 = -1.0*err0*err0/sigmasq + 0.25*err1*tt/(sigmasq/this->sigma);
        T t3 = -1.0/sigmasq;
        return t1 + t2 + t3;
    }
    //
    __host__ __device__ T newton(T yy,T yy0,T a0,T a1)
    {
        T x0 = 0.5*(a0 + a1);
        T g = df(yy,yy0,a0,x0,a1);
        int iter = 0;
        while(fabs(g) > 0.00001 && iter < 20){
            T h = ddf(yy,x0,a1);
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
    __host__ __device__ T stdstate(T yy,T a,T a1)
    {
        return sqrt(-1.0/ddf(yy,a,a1));
    }
    //
    __host__ __device__ T loglik(T yy,T yy0,T a0,T a,T a1)
    {
        T sigmat = this->sigma*sqrtf(1.0 - this->rho*this->rho);
        T t1 = -0.5*a - 0.5*yy*yy*expf(-1.0*a);
        T err1 = (a1 - this->mu - this->phi*(a - this->mu) - this->sigma*this->rho*expf(-0.5*a)*yy);
        T t2 = -0.5*err1*err1/(sigmat*sigmat);
        T err2 = (a - this->mu - this->phi*(a0 - this->mu) - this->sigma*this->rho*expf(-0.5*a0)*yy0);
        T t3 = -0.5*err2*err2/(sigmat*sigmat);
        return t1 + t2 + t3;
    }    //
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
        T l1 = this->loglik(yy,yy0,a0,anew,a1);
        T l0 = this->loglik(yy,yy0,a0,a,a1);
        T g0 = this->lognorm(a,m,s);
        T g1 = this->lognorm(anew,m,s);
        T tt = l1 - l0 + g0 - g1;
        if(tt > 0.0) return 1.0;
        if(tt < -10.0){
            return 0.0;
        }else{
            return exp(tt);
        }
    }
    //
    __host__ __device__ double logpdf1(double lt,double yy,double a,double a1)
    {
      double t1 = -0.5*log(lt);
      double t2 = -0.5*yy*yy/(exp(a)*lt);
      double tt3 = (a1 - this->mu - this->phi*(a - this->mu) - this->rho*this->sigma*yy*exp(-0.5*a)/sqrt(lt));
      double t3 = -0.5*tt3*tt3/(this->sigma*this->sigma*(1.0 - this->rho*this->rho));
      double t4 = -1.0*(0.5*(double)this->nu + 1.0)*log(lt);
      double t5 = -0.5*((double)this->nu)/lt;
      return t1 + t2 + t3 + t4 + t5;
    }

    __host__ __device__ double logpdf2(double lt,double yy,double a)
    {
      double t1 = -1.0*(0.5*((double)this->nu +1) + 1.0)*logf(lt);
      double t2 = -0.5*((double)this->nu + yy*yy*expf(-1.0*a))/lt;
      return t1 + t2;
    }

    __host__ __device__ double metroproblambda(double lt1,double lt0,double yy,double a,double a1)
    {
      double lpdf1 = logpdf1(lt1, yy, a, a1);
      double lpdf0 = logpdf1(lt0, yy, a, a1);
      double lpdf1ap = logpdf2(lt1, yy, a);
      double lpdf0ap = logpdf2(lt0, yy, a);
      double tt = lpdf1 - lpdf0 + lpdf0ap - lpdf1ap;
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
            if(i==0){
                this->alpha[i] = this->random->normal(this->mu,this->sigma/sqrt(1.0 - this->phi*this->phi));
                float t1 = 0.5*((T)this->nu + 1.0);
                float t2 = 0.5*this->x[i]*this->x[i]/exp(this->alpha[i]) + 0.5*(T)this->nu;
                float tt = 1.0/this->random->gamma(t1,t2);
                float pm = this->metroproblambda((double)tt,(double)this->lambda[i],(double)this->x[i],(double)this->alpha[i], (double)this->alpha[i+1]);
                if(this->random->uniform() < pm) this->lambda[i] = tt;
            }else if(i==(n-1)){
                T lam = this->lambda[i];
                T lam0 = this->lambda[i-1];
                T yy = this->x[i]/sqrtf(lam);
                T yy0 = this->x[i-1]/sqrtf(lam0);
                this->a0 = this->alpha[i-1];
                this->a1 = this->mu + this->phi*(this->alpha[i] - this->mu)
                + this->random->normal(0.0,this->sigma);
                T mx = this->meanstate(yy,yy0,this->a0,this->a1);
                T sx = this->stdstate(yy,mx,this->a1);
                T atemp = this->random->normal(mx, sx);
                if(this->random->uniform() < this->metroprob(atemp,this->alpha[i],this->a0,this->a1,yy,yy0,mx,sx)){
                    this->alpha[i] = atemp;
                }
                float t1 = 0.5*((T)this->nu + 1.0);
                float t2 = 0.5*this->x[i]*this->x[i]/exp(this->alpha[i]) + 0.5*(T)this->nu;
                float tt = 1.0/this->random->gamma(t1,t2);
                float pm = this->metroproblambda((double)tt,(double)this->lambda[i],(double)this->x[i],(double)this->alpha[i], (double)this->a1);
                if(this->random->uniform() < pm) this->lambda[i] = tt;
            }else{
                T lam = this->lambda[i];
                T lam0 = this->lambda[i-1];
                T yy = this->x[i]/sqrtf(lam);
                T yy0 = this->x[i-1]/sqrtf(lam0);
                this->a0 = this->alpha[i-1];
                this->a1 = this->alpha[i+1];
                T mx = this->meanstate(yy,yy0,this->a0,this->a1);
                T sx = this->stdstate(yy,mx,this->a1);
                T atemp = this->random->normal(mx, sx);
                if(this->random->uniform() < this->metroprob(atemp,this->alpha[i],this->a0,this->a1,yy,yy0,mx,sx)){
                    this->alpha[i] = atemp;
                }
                float t1 = 0.5*((T)this->nu + 1.0);
                float t2 = 0.5*this->x[i]*this->x[i]/exp(this->alpha[i]) + 0.5*(T)this->nu;
                float tt = 1.0/this->random->gamma(t1,t2);
                float pm = this->metroproblambda((double)tt,(double)this->lambda[i],(double)this->x[i],(double)this->alpha[i], (double)this->a1);
                if(this->random->uniform() < pm) this->lambda[i] = tt;
            }
        }
    }
    //
    //
    __host__ __device__ T simulatemu()
    {
        int m = this->n - 1;
        float ss = (this->sigma*sqrtf(1.0 - this->rho*this->rho)/(1.0 - this->phi));
        float *err = new float[m];
        for(int i=0;i<m;i++){
            T yy = this->x[i]/sqrtf(this->lambda[i]);
            err[i] = (this->alpha[i+1] - this->phi*this->alpha[i] -
                      this->sigma*this->rho*expf(-0.5*this->alpha[i])*yy)/(1.0 - this->phi);
        }
        NormalModel<float> *nm = new NormalModel<float>(err,m,this->mu,ss);
        this->mu = nm->simulatemu();
        delete[] err;
        delete nm;
        return this->mu;
    }
    //
    //
    __host__ __device__ T simulatephi()
    {
        int m = this->n - 1;
        float *err2 = new float[m];
        float *err1 = new float[m];
        float ss = this->sigma*sqrtf(1.0 - this->rho*this->rho);
        for(int i=0;i<m;i++){
            T yy = this->x[i]/sqrtf(this->lambda[i]);
            err2[i] = (this->alpha[i+1] - this->mu) - this->sigma*this->rho*exp(-0.5*this->alpha[i])*yy;
            err1[i] = (this->alpha[i] - this->mu);
        }
        RegModel<float> *reg = new RegModel<float>(err2,err1,m,0.0,this->phi,ss);
        reg->setbetapriortype(1);
        reg->setbetaprior(this->phiprior);
        this->phi = reg->simulatebeta();
        delete[] err1;
        delete[] err2;
        return this->phi;
    }
    //
    //
    //
    __host__ __device__ void simulatesigmarho()
    {
        int m = this->n - 1;
        float psi = this->sigma*this->rho;
        float Omega = this->sigma*this->sigma - this->sigma*this->sigma*this->rho*this->rho;
        //
        float *y2 = new float[m];
        float *x2 = new float[m];
        for(int i=0;i<m;i++){
            T yy = this->x[i]/sqrtf(this->lambda[i]);
            y2[i] = this->alpha[i+1] - this->mu - this->phi*(this->alpha[i] - this->mu);
            x2[i] = yy*exp(-0.5*this->alpha[i]);
        }
        //
        float a11 = 0.0;
        float a12 = 0.0;
        for(int i=0;i<m;i++){
            a11 += x2[i]*x2[i];
            a12 += x2[i]*y2[i];
        }
        float mpsi = (a12/a11);
        float spsi = sqrtf(Omega/a11);
        //
        psi = mpsi + spsi*this->random->normal();
        //
        float rss = 0.0;
        for(int i=0;i<m;i++){
            float temp =  y2[i] - psi*x2[i];
            rss += temp*temp;
        }
        Omega = 1.0/this->random->gamma(0.5*(double)m, 0.5*rss);
        delete[] y2;
        delete[] x2;
        this->sigma = sqrtf(Omega + psi*psi);
        this->rho = psi/this->sigma;
    }
    //
    //
    __host__ __device__ int simulatenu()
    {
        T *err = new T[this->n];
        for(int i=0;i<this->n;i++){
            err[i] = this->x[i]*exp(-0.5*this->alpha[i]);
        }
        Stats<T> *stats = new Stats<T>(err,n);
        stats->setSeed(this->random->rand());
        this->nu = stats->sampletstudentdf(3,100);
        delete[] err;
        delete stats;
        return this->nu;
    }
    //
    //
    __host__ __device__ T getsigma()
    {
        return this->sigma;
    }
    //
    //
    __host__ __device__ T getrho()
    {
        return this->rho;
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
    //
};

#endif
