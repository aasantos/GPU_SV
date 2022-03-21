# GPU_SV

Implements sequential and parallel computing for estimating 4 types
of stochastic volatility (SV) models

The basic SV

SV with leverage (SVL)

SV with fat-tails (SVT)

SV with leverage and fat-tails (SVTL)

The application can be complied just using:

nvcc main.cu -o programApp


To run the application, two inputs are needed, first, the raw text file with returns; in this repository exists one associated with the S&P500 index daily returns form 1980 to 1987 (sp500_ret_80_87.txt). Second a code indicating the model to be estimated and if CPU or GPU is used

0 SV-CPU
1 SVT-CPU
2 SVL-CPU
3 SVTL-CPU
4 SV-GPU
5 SVT-GPU
6 SVL-GPU
7 SVTL-GPU

