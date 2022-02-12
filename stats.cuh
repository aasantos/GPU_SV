#ifndef stats_cuh
#define stats_cuh


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "random.cuh"


template <typename T>
class Stats{
protected:
    int n;
    T *x;
    Random<T> *random;
    unsigned int seed;
    //
public:
    //
    //
    __host__ __device__ Stats(T *x,int n)
    {
        this->n = n;
        this->x = x;
        this->seed = 86120910;
        this->random = new Random<T>(this->seed);
    }
    //
    __host__ __device__ ~Stats()
    {
        if(random){
            delete random;
        }
    }
    //
    //
    __host__ __device__ T sum()
    {
        T result = 0.0;
        for(int i=0;i<this->n;i++){
            result += this->x[i];
        }
        return result;
    }
    //
    //
    //calculates the mean
    __host__ __device__ float mean()
    {
        float mvalue = 0.0;
        for(int i=0;i<this->n;i++){
            mvalue += (float)this->x[i];
        }
        return mvalue/(float)this->n;
    }
    //
    //
    //calculates the variance
    __host__ __device__ float variance()
    {
        float mm = this->mean();
        float msqvalue = 0.0;
        for(int i=0;i<this->n;i++){
            msqvalue += (float)(this->x[i]*this->x[i]);
        }
        msqvalue /= (float)this->n;
        return msqvalue - mm*mm;
    }
    //
    //
    //
    __host__ __device__ float kurtosis()
    {
        float mm = this->mean();
        float vv = this->variance();
        float value = 0.0;
        for(int i=0;i<this->n;i++){
            value += pow(((float)this->x[i] - mm),4.0);
        }
        value /= (float)this->n;
        return value/(vv*vv);
    }
    //
    //
    //calcutes the log-pdf of a student t distribution with df degrees of freedom
    //using as input the value of the variable x and the df
    //tested
    __host__ __device__ T logpdfstudentt(T x,int df)
    {
        T temp1 = lgamma(0.5*((T)df + 1.0));
        T temp2 = lgamma(0.5*(T)df);
        T temp3 = 0.5*log((T)df*3.141592653589793);
        T temp4 = 0.5*((T)df + 1.0)*log(1.0 + pow(x, 2.0)/(T)df);
        return temp1 - temp2 - temp3 - temp4;
    }
    //
    //
    //calculates the log-pdf of a student t distribution
    //for a Vector with df degrees of freedom
    //tested
    __host__ __device__ void logpdfstudenttvect(int df,T *z)
    {
        for(int i=0;i<this->n;i++){
            z[i] = this->logpdfstudentt(this->x[i], df);
        }
    }
    //
    //
    //tested
    __host__ __device__ T loglikstudentt(int df)
    {
        T *temp = new T[n];
        this->logpdfstudenttvect(df,temp);
        //double loglik = Stats(temp,n).sum();
        T loglik = 0.0;
        for(int i=0;i<this->n;i++){
            loglik += temp[i];
        }
        delete[] temp;
        return loglik;
    }
    //
    //
    //scaled student t
    __host__ __device__ T logpdfstudenttw(T x,int df)
    {
        T temp1 = lgamma(0.5*((T)df + 1.0));
        T temp2 = lgamma(0.5*(T)df);
        T temp3 = 0.5*log((T)df - 2.0);
        T temp4 = 0.5*((T)df + 1.0)*log(1.0 + pow(x, 2.0)/((T)df - 2.0));
        return temp1 - temp2 - temp3 - temp4;
    }
    //
    //
    __host__ __device__ void logpdfstudenttwvect(int df,T *z)
    {
        for(int i=0;i<this->n;i++){
            z[i] = this->logpdfstudenttw(this->x[i], df);
        }
    }
    //
    //
    //
    __host__ __device__ T loglikstudenttw(int df)
    {
        T *temp = new T[this->n];
        this->logpdfstudenttwvect(df,temp);
        double loglik = 0.0;
        for(int i=0;i<this->n;i++){
            loglik += temp[i];
        }
        delete[] temp;
        return loglik;
    }
    //
    //
    //
    __host__ __device__ void loglikdfstudenttw(int *df,int m,T *z)
    {
        T *w = new T[m];
        for(int i=0;i<m;i++){
            T t1 = this->loglikstudenttw(df[i]);
            w[i] = t1;
        }
        for(int i=0;i<m;i++){
            T sumt = 0.0;
            for(int j=0;j<m;j++){
                sumt += exp(w[j] - w[i]);
            }
            z[i] = 1.0/sumt;
        }
        delete[] w;
    }
    //
    //
    //tested
    __host__ __device__ void loglikdfstudentt(int *df,int m,double *z)
    {
        T *w = new T[m];
        for(int i=0;i<m;i++){
            T t1 = this->loglikstudentt(df[i]);
            w[i] = t1;
        }
        for(int i=0;i<m;i++){
            T sumt = 0.0;
            for(int j=0;j<m;j++){
                sumt += exp(w[j] - w[i]);
            }
            z[i] = 1.0/sumt;
        }
        delete[] w;
    }
    //
    //
    //
    __host__ __device__ int sampletstudentdf(int liminf,int limsup)
    {
        int m = limsup - liminf + 1;
        int *df = new int[m];
        double *w = new double[m];
        df[0] = liminf;
        for(int i=1;i<m;i++){
            df[i] = df[i-1] + 1;
        }
        this->loglikdfstudentt(df,m,w);
        int dft = this->random->sampleuni_int(df, w, m);
        delete[] df;
        delete[] w;
        return dft;
    }
    //
    __host__ __device__ void setSeed(unsigned int seed)
    {
        this->seed = seed;
        if(random){
            delete random;
            this->random = new Random<T>(seed);
        }
    }
};



#endif
