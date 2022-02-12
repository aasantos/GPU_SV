#ifndef vector_cuh
#define vector_cuh

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


template <typename T>
class Vector{
protected:
    T *x;
    int n;
public:
    //
    __host__ __device__ Vector(T *xx,int n)
    {
        this->x = xx;
        this->n = n;
    }
    //
    //
    __host__ __device__ T sum()
    {
        T res = 0.0;
        for(int i=0;i<this->n;i++) res += x[i];
        return res;
    }
    
    __host__ __device__ T sumsq()
    {
        T res = 0.0;
        for(int i=0;i<this->n;i++) res += x[i]*x[i];
        return res;
    }
    
    __host__ __device__ float mean()
    {
        float ss = (float)this->sum();
        return ss/(float)n;
    }
    
    __host__ __device__ float variance()
    {
        float mm = this->mean();
        return sumsq()/(float)n - mm*mm;
    }
    
    __host__ __device__ int length()
    {
        return n;
    }
    
    __host__ __device__ T& operator[](int j)
    {
        return this->x[j];
    }
    //
    //
    __host__ __device__ Vector<T> operator+(Vector<T> &u)
    {
        Vector<T> v = Vector<T>(this->n);
        for(int i=0;i<n;i++){
            v[i] = this->x[i] + u[i];
        }
        return v;
    }
    //
    //
    __host__ __device__ Vector<T> operator-(Vector<T> &u)
    {
        Vector<T> v  = Vector<T>(this->n);
        for(int i=0;i<n;i++){
            v[i] = this->x[i] + u[i];
        }
        return v;
    }
    //
    //
    __host__ __device__ Vector<T> operator*(Vector<T> &u)
    {
        Vector<T> v = Vector<T>(this->n);
        for(int i=0;i<n;i++){
            v[i] = this->x[i]*u[i];
        }
        return v;
    }
    //
    //
    __host__ __device__ Vector<T> operator/(Vector<T> &u)
    {
        Vector<T> v = Vector<T>(this->n);
        for(int i=0;i<n;i++){
            v[i] = this->x[i]/u[i];
        }
        return v;
    }
    //
    //innerprod
    __host__ __device__ T operator^(Vector<T> &u)
    {
        T res = 0;
        for(int i=0;i<n;i++){
            res += this->x[i]*u[i];
        }
        return res;
    }
    
    __host__ __device__ void swap(T *a,T *b)
    {
        T temp = *a;
        *a = *b;
        *b = temp;
    }
    
    __host__ __device__ void bubbleSort(T array[],int n)
    {
        for(int i=0;i<n-1;i++)
            for(int j=0;j<n-i-1;j++) 
                if(array[j] > array[j+1])
                   swap(&array[j],&array[j+1]);
    }
    
    __host__ __device__ void sort()
    {
        bubbleSort(this->x,this->n);
    }
    
    __host__ __device__ void linspace(T start,T end)
    {
        T step = (end - start)/(double)(n-1);
        this->x[0] = start;
        for(int i=1;i<this->n;i++){
            this->x[i] = this->x[i-1] + step;
        }
    }
    
};
#endif
