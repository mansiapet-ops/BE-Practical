%%writefile cuda_prog.cu
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void vectoraddCPU(int *a,int *b,int *c,int n){
  for(int i=0;i<n;i++){
    c[i]=a[i]+b[i];
  }
}

__global__ void vectoraddGPU(int *a,int *b,int *c,int n){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<n){
    c[idx]=a[idx]+b[idx];
  }
}

void matrixmultiplyCPU(int *A,int *B,int *C,int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            int sum=0;
            for(int k=0;k<N;k++){
                sum+=A[i*N+k]*B[k*N+j];
            }
            C[i*N+j]=sum;
        }
    }
}

__global__ void matrixmultiplyGPU(int *A,int *B,int *C,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<N){
        int sum=0;
        for(int k=0;k<N;k++){
            sum+=A[row*N+k]*B[k*N+col];
        }
        C[row*N+col]=sum;
    }
}

int main(){
    int vecSize=1 << 24; //vecSize 2^24
    int N=512 ; //512X512 matrix

    int *a=new int[vecSize];
    int *b=new int[vecSize];
    int *c_cpu=new int[vecSize];
    int *c_gpu=new int[vecSize];

    for(int i=0;i<vecSize;i++){
        a[i]=rand()%100;
        b[i]=rand()%100;
    }

    auto start=high_resolution_clock::now();
    vectoraddCPU(a,b,c,vecSize);
    auto end=high_resolution_clock::now();
    cout<<"/n Vector Addition CPU Time:"<<duration_cast<milliseconds>(end-start).count()<<" milliseconds/n";

    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,vecSize*sizeof(int));
    cudaMalloc(&d_b,vecSize*sizeof(int));
    cudaMalloc(&d_c,vecSize*sizeof(int));

    cudaMemcpy(d_a,a,vecSize*sizeof(int),cudaMemcpyHosttoDevice);
    cudaMemcpy(d_b,b,vecSize*sizeof(int),cudaMemcpyHosttoDevice);

    start=high_resolution_clock::now();
    vectoraddGPU<<<(vecSize+255)/256,256>>>(d_a,d_b,d_c,vecSize);
    cudaDeviceSynchronize();
    end=high_resolution_clock::now();
    cout<<"Vector Add CPU Time:"<<duration_cast<milliseconds>(end-start).count()<<" milliseconds/n";

    cudaMemcpy(c_gpu,d_c,vecSize*sizeof(int),cudaMemcpyDevicetoHost);

    int size=N*N*sizeof(int);
    int* A=new int[N*N];
    int *B=new int[N*N];
    int *C_cpu=new int[N*N];
    int *C_gpu=new int[N*N];

    for(int i=0;i<N*N;i++){
        A[i]=rand()%10;
        B[i]=rand()%10;
    }

    start=high_resolution_clock::now();
    matrixmultiplyCPU(A, B, C_cpu,N);
    end=high_resolution_clock::now();
    cout<<"Matrix Multipy Time by CPU:"<<duration_cast<milliseconds>(end-start).count()<<"ms\n";

    int *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,A,size,cudaMemcpyHosttoDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHosttoDevice);

    dim3 threads(16,16); //craete a 2D Grid MATRIX
    dim3 blocks((N+15)/16,(N+15)/16);

    start=high_resolution_clock::now();
    matrixmultiplyGPU<<<blocks,threads>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    end=high_resolution_clock::now();
    cout<<"Matrix Multipy Time by GPU:"<<duration_cast<milliseconds>(end-start).count()<<"ms\n";
   
    cudaMemcpy(C_gpu,d_C,size,cudaMemcpyDevicetoHost);

    delete[] a;delete[] b;delete[] c_cpu; delete[] c_gpu;
    delete[] A;delete[] B;delete[] C_cpu; delete[] C_gpu;
    
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);

    return 0;
}
