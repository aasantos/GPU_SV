#ifndef runfunc.cuh
#define runfunc.cuh

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include "io.cuh"
#include "vector.cuh"
#include "random.cuh"
#include "normalmodel.cuh"
#include "regmodel.cuh"
#include "ar1model.cuh"
#include "dlm.cuh"


void simulate_dlm()
{
  int n = 1000;
  float sigmav = 0.15;
  float mu = -0.5;
  float phi = 0.97;
  float sigma = 0.2;
  //
  //
  Random *random = new Random(8687);
  
  delete random;
  
}



#endif
