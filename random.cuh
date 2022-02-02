#ifndef random_cuh
#define random_cuh

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "vector.cuh"

#define PI  3.141592653589793

#define  A1  (-3.969683028665376e+01)
#define  A2   2.209460984245205e+02
#define  A3  (-2.759285104469687e+02)
#define  A4   1.383577518672690e+02
#define  A5  (-3.066479806614716e+01)
#define  A6   2.506628277459239e+00

#define  B1  (-5.447609879822406e+01)
#define  B2   1.615858368580409e+02
#define  B3  (-1.556989798598866e+02)
#define  B4   6.680131188771972e+01
#define  B5  (-1.328068155288572e+01)

#define  C1  (-7.784894002430293e-03)
#define  C2  (-3.223964580411365e-01)
#define  C3  (-2.400758277161838e+00)
#define  C4  (-2.549732539343734e+00)
#define  C5   4.374664141464968e+00
#define  C6   2.938163982698783e+00

#define  D1   7.784695709041462e-03
#define  D2   3.224671290700398e-01
#define  D3   2.445134137142996e+00
#define  D4   3.754408661907416e+00

#define P_LOW   0.02425
/* P_high = 1 - p_low*/
#define P_HIGH  0.97575

template <typename T>
class Random{
public:
    //
    //
    __host__ __device__ Random(){
        //
        this->x = 302308235;
        this->y = 10970392;
        this->z1 = 2370309;
        this->c1 = 2429979;
        this->z2 = 746245245;
        this->c2 = 4364524;
    }
    //
    //
    __host__ __device__ Random(int idxx)
    {
        unsigned long long idx = (unsigned long long)idxx;
        unsigned long long seed[6] =
        {(idx + 546)*79794,(idx+890)*564534,
            (idx+98)*580345,(idx+9)*45,
            (idx+23)*897,(idx+ 67)*212879987};
        this->x = seed[0];
        this->y = seed[1];
        this->z1 = seed[2];
        this->c1 = seed[3];
        this->z2 = seed[4];
        this->c2 = seed[5];
    }
    //
    //
     __host__ __device__ ~Random(){};
    //
    //
    __host__ __device__ unsigned long long JLKISS64()
    {
        unsigned long long t = 0;
        x = 1490024343005336237ULL * x + 123456789;
        y ^= y << 21; y ^= y >> 17; y ^= y << 30;
        /* Do not set y=0! */
        t = 4294584393ULL * z1 + c1;
        c1 = t >> 32;
        z1 = t;
        t = 4246477509ULL * z2 + c2;
        c2 = t >> 32;
        z2 = t;
        return x + y + z1 + ((unsigned long long)z2 << 32);
        /* Return 64-bit result */
    }
    //
    __host__ __device__ unsigned int rand()
    {
        return (unsigned int)JLKISS64();
    }
    //
    __host__ __device__ T uniform()
    {
        unsigned long long div = 18446744073709551615ULL; //2^64 -1
        return (T)(JLKISS64()/(1.0*div));
    }
    //
    //
    __host__ __device__ T normal()
    {
        T p =  uniform();
        T x = 0.0;
        T q = 0.0;
        T r = 0.0;
        if ((0 < p )  && (p < P_LOW)){
            q = sqrt(-2*log(p));
            x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
        }
        else{
            if ((P_LOW <= p) && (p <= P_HIGH)){
                q = p - 0.5;
                r = q*q;
                x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q /(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1);
            }
            else{
                if ((P_HIGH < p)&&(p < 1)){
                    q = sqrt(-2*log(1-p));
                    x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1);
                }
            }
        }
        return x;
    }
    //
    __host__ __device__ T normal(T mu,T sigma)
    {
        return mu + sigma*normal();
    }
    //
    __host__ __device__ T exponential(T lambda)
    {
        T u = uniform();
        return -1.0*logf(u)/lambda;
    }
    //
    //
    __host__ __device__ T randGammabest(T a)
    {
        int flag = 0;
        T b = 0.0;
        T c = 0.0;
        T u = 0.0;
        T v = 0.0;
        T w = 0.0;
        T y = 0.0;
        T x = 0.0;
        T z = 0.0;
        T test1 = 0.0;
        T test2 = 0.0;
        T test3 = 0.0;
        T temp = 0.0;
        b = a - 1.0;
        c = (12.0*a - 3.0)/4.0;
        flag = 0;
        while(flag == 0){
            u = uniform();
            v = uniform();
            w = u*(1.0-u);
            y = sqrtf(c/w)*(u-0.5);
            x = b + y;
            if(x>0){
                z = 64.0*v*v*w*w*w;
                test1 = 1.0 - 2*y*y/x;
                test2 = 2.0*(b*logf(x/b)-y);
                test3 = logf(z);
                if(z <= test1 || test2 >= test3){
                    temp = x;
                    flag = 1;
                }
            }
        }
        return temp;
    }
    //
    //
    __host__ __device__ T randGammaless(T a)
    {
        T temp = 0.0;
        T x = 0.0;
        T u = uniform();
        temp = randGammabest((a + 1.0));
        x = temp*powf(u,1.0/a);
        return x;
    }
    //
    //
    __host__ __device__ T gamma(T a,T b)
    {
        T g = 0.0;
        if(a<1.0){
            g = (1/b)*randGammaless(a);
        }
        else if(a==1.0){
            g = (1/b)*exponential(a);
        }
        else{
            g = (1/b)*randGammabest(a);
        }
        return g;
    }
    //
    //
    __host__ __device__ T beta(T a,T b)
    {
        T x = gamma(a,1.0);
        T y = gamma(b,1.0);
        T result = x/(x + y);
        return result;
    }
    //
    //
    __host__ __device__ T t(int df)
    {
        T u = normal();
        T v = gamma(0.5*(T)df,0.5);
        return u/sqrt(v/(T)df);
    }
    //
    //
    __host__ __device__ int bernoulli(T p)
    {
        if(uniform() < p){
            return 1;
        }else{
            return 0;
        }
    }
    //
    //
    __host__ __device__ void bivariatenormal(T m[2],T S[4],T x[2])
    {
        T a = S[0];
        T b = S[1];
        T c = S[3];
        //
        T t[2];
        t[0] = normal();
        t[1] = normal();
        x[0] = sqrt(a)*t[0];
        T tt = sqrt((a*c - b*b)/a);
        //stupid error stupid stupid
        //x[1] = (b/sqrt(a))*t[0] + sqrt(tt)*t[1];
        x[1] = (b/sqrt(a))*t[0] + tt*t[1];
        x[0] += m[0];
        x[1] += m[1];
    }
    //
    //
    __host__ __device__ T area(T *x,T *y,int n)
    {
        T aa = 0.0;
        for(int i=0;i<n;i++){
            if(i == 0){
                aa += 0.5*fabs(x[i+1] - x[i])*y[i];
            }else if(i == (n -1)){
                aa += 0.5*fabs(x[i] - x[i-1])*y[i];
            }else{
                aa += 0.5*fabs(x[i+1] - x[i-1])*y[i];
            }
        }
        return aa;
    }
    //
    //
    __host__ __device__ T randemp(T *x,T *y,int n)
    {
        T aa = 0.0;
        T u = uniform();
        int i = -1;
        while(aa < u){
            i += 1;
            if(i == 0){
                aa += 0.5*fabs(x[i+1] - x[i])*y[i];
            }else if(i == (n - 1)){
                aa += 0.5*fabs(x[i] - x[i-1])*y[i];
            }else{
                aa += 0.5*fabs(x[i+1] - x[i-1])*y[i];
            }
        }
        return x[i];
    }
    //
    //
    __host__ __device__ T normal01(T mean,T std)
    {
        T aa = -1.0;
        while( aa < 0 || aa > 1 ){
            aa = mean + std*normal();
        }
        return aa;
    }
    //
    //
    //
    __host__ __device__ T logbeta(T a,T b)
    {
        return lgamma(a) + lgamma(b) - lgamma(a + b);
    }
    //
    //
    __host__ __device__ T normalbeta(T mu,T sigma,T a,T b)
    {
        int n = 500;
        T *x = new T[n];
        for(int i=0;i<250;i++){
            x[i] = normal01(mu,3*sigma);
            x[i + 250] = beta(a, b);
        }
        Vector<T>(x,n).sort();
            T *y = new T[n];
            for(int i=0;i<n;i++){
                y[i] = -0.5*log(2*PI*sigma*sigma)
                        - 0.5*(x[i] - mu)*(x[i] - mu)/(sigma*sigma)
                        - logbeta(a,b)
                        + (a - 1.0)*log(x[i])
                        + (b - 1.0)*log(1.0 - x[i]);
                y[i] = exp(y[i]);
            }
            T aa = area(x, y, n);
            for(int i=0;i<n;i++){
                y[i] = y[i]/aa;
            }
        T res = randemp(x, y, n);
        delete[] x;
        delete[] y;
        return res;
    }
//
    //
    __host__ __device__ double sampleuni(double *x,double *w,int n)
    {
        double z = 0.0;
        double *Q = new double[n];
        Q[0] = w[0];
        for(int i=1;i<n;i++){
            Q[i] = Q[i-1] + w[i];
        }
        int iter = 0;
        double r = uniform();
        if(r < Q[0]){
            z = x[0];
            return z;
        }
        do{
            if( r > Q[iter] && r < Q[iter+1]){
                z = x[iter+1];
                return z;
            }
            iter++;
        }while(iter < (n-1));
        delete[] Q;
        return z;
    }
    //
    //
    __host__ __device__ int sampleuni_int(int *x,double *w,int n)
    {
        int z = 0;
        double *Q = new double[n];
        Q[0] = w[0];
        for(int i=1;i<n;i++){
            Q[i] = Q[i-1] + w[i];
        }
        int iter = 0;
        double r = uniform();
        if(r < Q[0]){
            z = x[0];
            return z;
        }
        do{
            if( r > Q[iter] && r < Q[iter+1]){
                z = x[iter+1];
                return z;
            }
            iter++;
        }while(iter < (n-1));
        delete[] Q;
        return z;
    }
    //

//
private:
    unsigned long long x;
    unsigned long long y;
    unsigned long long z1;
    unsigned long long c1;
    unsigned long long z2;
    unsigned long long c2;
};

#endif
