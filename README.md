# GPU_SV

Implements sequential and parallel computing four estimating for types
of stochastic volatility (SV) models

SV basic

SV with leverage (SVL)

SV with fat-tails (SVT)

SV with leverage and fat-tails (SVTL)

The application can be complied just using:

nvcc main.cu -o programApp


To run the application, two inputs are needed, first, the raw text file with the returns; 
in this repository exists one associated with the S&P500 index daily returns 
form 1980 to 1987 (sp500_ret_80_87.txt). Second a code indicating the model 
to be estimated and if CPU or GPU is used

0 SV-CPU

1 SVT-CPU

2 SVL-CPU

3 SVTL-CPU

4 SV-GPU

5 SVT-GPU

6 SVL-GPU

7 SVTL-GPU


A given set of parameters is assumed, i.e., number of iterations, burnin iterations 
and initial values for the parameters. Any desired modification, 
at this stage, must me done within the original code.

Examples: 

Estimating the SVTL model sequentially using just CPU

./programApp sp500_ret_80_87.txt 3


Estimating the SVTL model in parallel using GPU

./programApp sp500_ret_80_87.txt 7


The output of the analysis is saved on a .txt file whose name 
indicates the type of model estimated and when GPU is used, 
the tag gpu is added. For example, after estimating the SVTL-GPU (7) model, 
the file svtlestimgpu.txt is created. 

