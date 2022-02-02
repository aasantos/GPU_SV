#ifndef io_cuh
#define io_cuh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cuda.h>

template <typename T>
T* readArray(const char* file,int *n)
{
    FILE *fp;
    fp = fopen(file, "r");
    char buff[64];
    int nrow = -1;
    while (!feof(fp)) {
        fscanf(fp, "%s",buff);
        nrow++;
    }
    *n = nrow;
    rewind(fp);
    T *result = (T*)malloc(nrow*sizeof(T));
    for(int i=0;i<nrow;i++){
        fscanf(fp, "%s",buff);
        result[i] = atof(buff);
    }
    fclose(fp);
    return result;
}
//
//
//
template <typename T>
T* readArray2cuda(const char* file,int *n)
{
	T *x = readArray<T>(file,n);
	int m = *n;
	T *x_d;
	cudaMallocManaged(&x_d,m*sizeof(T));
	for(int i=0;i<m;i++) x_d[i] = x[i];
	free(x);
	return x_d;
}
//
//
//
template <typename T>
void writeArray(T *x,const char* file,int n)
{
    FILE *fp;
    fp = fopen(file, "wa");
    for(int i=0;i<n;i++) fprintf(fp,"%.6f\n",x[i]);
    fclose(fp);
}
//
//
void writeArray_int(int *x,const char* file,int n)
{
    FILE *fp;
    fp = fopen(file, "wa");
    for(int i=0;i<n;i++) fprintf(fp,"%d\n",x[i]);
    fclose(fp);
}


#endif
