//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "fft.h"

// Define macros for thread and block indices
#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define TILE_DIM 32
// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

//-----------------------------------------------------------------------------

// Kernel function that performs bit-reversal permutation on input array using shared memory
__global__ void bit_reverse_sort(float* in1, float* in2, unsigned int M, int k) 
{

    // Get global thread index
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;

    // Declare shared array of size 256
    __shared__ float sdata[256];

    // Copy input element from input1 to shared array
    sdata[tx] = in2[i + k];

    // Synchronize threads
    __syncthreads();

    // Swap bits of index using bitwise operations
    unsigned int j = i;
	
    j = ((j & 0xcccccccc) >> 2) | ((j & 0x33333333) << 2);
    j = ((j & 0xf0f0f0f0) >> 4) | ((j & 0x0f0f0f0f) << 4);
    j = ((j & 0xff00ff00) >> 8) | ((j & 0x00ff00ff) << 8);
    j = (j >> 16) | (j << 16);
    j >>= 32-M;

    // Copy element from shared array to input at bit-reversed index
    in1[j]=sdata[tx];
    
}


// Kernel function that performs radix-2 FFT on input arrays
__global__ void fft_radix2 (float* x_real_d, float* x_imag_d ,const unsigned int N, int M){

    // Get global thread index
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;

    // Compute twiddle factor using trigonometric functions
    float theta = -1 * 2 * PI * ((i*(N/(2*M)))-(i/M)*(N/2)) / N;
	float Wreal = cos(theta);                   
	float Wimag = sin(theta);

	float x1_real , x2_real , x1_imag, x2_imag;
	
	// Get input elements from real and imaginary arrays
	x1_real  = x_real_d[i+(i/M)*M];         
	x2_real  = x_real_d[i+(i/M)*M+(M)];
	
	x1_imag = x_imag_d[i+(i/M)*M];
	x2_imag = x_imag_d[i+(i/M)*M+(M)];
   
	// Perform butterfly operation and store output in real and imaginary arrays
	x_real_d[i+(i/M)*M] = x1_real  + Wreal * x2_real  - Wimag * x2_imag;
	x_imag_d[i+(i/M)*M] = x1_imag + Wreal * x2_imag + Wimag * x2_real;
	
	x_real_d[i+(i/M)*M+(M)] = x1_real  - Wreal * x2_real + Wimag * x2_imag;
	x_imag_d[i+(i/M)*M+(M)] = x1_imag - Wreal * x2_imag - Wimag * x2_real;		
		
}


// Device function that computes twiddle factors
__device__ void twiddle(float& real, float& imag, float x_real, float x_imag, float theta) {
  float theta1 = cos(theta);
  float theta2 = sin(theta);
  real = x_real * theta1 - x_imag * theta2;
  imag = x_real * theta2 + x_imag * theta1;
}

// Kernel function that performs radix-4 FFT on input arrays
__global__ void fft_radix4 (float* x_real_d, float* x_imag_d, const unsigned int N, int M,unsigned int k)
{
    // Get global thread index
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
	float theta  = -2*PI*(i%M) / (M*4);	
	float x1_real, x2_real, x1_imag, x2_imag, x4_real, x3_real, x4_imag, x3_imag;	
	float y2_i, y3_i, y4_i, y2_r, y3_r, y4_r;
	
	
	// Get input elements from real and imaginary arrays
	x1_real = x_real_d[(i/M)*(4*M)+(i%M) + k];      
	x2_real = x_real_d[(i/M)*(4*M)+(i%M) + M + k];
	x3_real = x_real_d[(i/M)*(4*M)+(i%M) + 2*M + k];
	x4_real = x_real_d[(i/M)*(4*M)+(i%M) + 3*M + k];
	
	x1_imag = x_imag_d[(i/M)*(4*M)+(i%M) + k];
	x2_imag = x_imag_d[(i/M)*(4*M)+(i%M) + M + k];
	x3_imag = x_imag_d[(i/M)*(4*M)+(i%M) + 2*M + k];
	x4_imag = x_imag_d[(i/M)*(4*M)+(i%M) + 3*M + k];	
	
	// Compute twiddle factors using device function
	twiddle(y2_r, y2_i, x2_real, x2_imag, theta);
	twiddle(y3_r, y3_i, x3_real, x3_imag, 2*theta);
	twiddle(y4_r, y4_i, x4_real, x4_imag, 3*theta);
	
	// Perform butterfly operation and store output in real and imaginary arrays
	x_real_d[(i/M)*(M*4)+(i%M) + k] = x1_real + y2_r + y3_r + y4_r;
	x_imag_d[(i/M)*(M*4)+(i%M) + k] = x1_imag + y2_i + y3_i + y4_i;
	
	x_real_d[(i/M)*(M*4)+(i%M) + M + k] = x1_real + y2_i - y3_r - y4_i;
	x_imag_d[(i/M)*(M*4)+(i%M) + M + k] = x1_imag - y2_r - y3_i + y4_r;
	
	x_real_d[(i/M)*(M*4)+(i%M) + 2*M + k] = x1_real - y2_r + y3_r - y4_r;
	x_imag_d[(i/M)*(M*4)+(i%M) + 2*M + k] = x1_imag - y2_i + y3_i - y4_i;
	
	x_real_d[(i/M)*(M*4)+(i%M) + 3*M + k] = x1_real - y2_i - y3_r + y4_i;
	x_imag_d[(i/M)*(M*4)+(i%M) + 3*M + k] = x1_imag + y2_r - y3_i - y4_r;
	
}



// Kernel function that copies elements from temp and temp1 arrays to x and x1 arrays
__global__ void transfer(float* x, float* temp, float* x1, float* temp1) {

	// Get global thread index
	int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;

	// Copy elements from temp and temp1 to x and x1
	x[i] = temp[i];
	x1[i] = temp1[i];

}


// Kernel function that copies elements from temp array to x array with an offset of k
__global__ void  copy_array(float* x, float* temp, int k) {

	// Get global thread index
	int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;

	// Copy element from temp to x with an offset of k
	x[i + k] = temp[i];
}



// Kernel function that transposes the input arrays by swapping their even and odd indices
__global__ void transpose(float* x, float* tmp, float* x1, float* tmp1, const unsigned int N) 
{
    // Get global thread index
    int i = (bz*gridDim.y*gridDim.x + by * gridDim.x + bx) * blockDim.x  + tx;
	int j = i % 2;
	// Check if index is even or odd
	tmp[i/2 + j*N/2] = x[i];
	tmp1[i/2+ j*N/2] = x1[i];
    

}


// Function that calls transpose, transfer and sort_even_number kernels to sort an odd-sized input array by its bit-reversed indices
void sort_number(float* x_r_d, float* x_i_d ,const unsigned int N,const unsigned int M , int k, int odd_even)
{
	if(odd_even == 1){
    // Allocate device memory for temporary arrays
    float* tmp_r;
    float* tmp_i;

    cudaMalloc((void**)&tmp_r, sizeof(float) * N);
    cudaMalloc((void**)&tmp_i, sizeof(float) * N);

    // Define grid and block dimensions
    dim3 dimGrid1(N/1024, 1, 1);
	dim3 dimBlock1(1024, 1, 1);

    // Call transpose kernel for real and imaginary arrays
    transpose<<<dimGrid1 , dimBlock1>>>(x_r_d, tmp_r, x_i_d, tmp_i, N);
    
    // Call transfer kernel for real and imaginary arrays
    transfer<<<dimGrid1 , dimBlock1>>>(x_r_d, tmp_r, x_i_d, tmp_i);

    // Free device memory for temporary arrays
    cudaFree(tmp_r);
    cudaFree(tmp_i);
	
	// Call sort_even_number function for two subarrays of size N/2
	int NN = N/2;
	int MM = M-1;
	float* tmp;
    cudaMalloc((void**)&tmp, sizeof(float) * NN);
    // Define grid and block dimensions
    dim3 dimGrid((NN / (512*512)), 32, 32);
	dim3 dimBlock(256, 1, 1);
    // Call bit_reverse_sort kernel for real and imaginary arrays
    bit_reverse_sort <<< dimGrid, dimBlock >>>(tmp, x_r_d, MM, 0);
    copy_array <<< dimGrid, dimBlock >>>(x_r_d,tmp, 0);
    bit_reverse_sort <<< dimGrid, dimBlock >>>(tmp, x_i_d, MM, 0);
    copy_array <<< dimGrid, dimBlock >>>(x_i_d,tmp, 0);
    // Free device memory for temporary array
    cudaFree(tmp);
	float* tmp1;
    cudaMalloc((void**)&tmp1, sizeof(float) * NN);

    // Call bit_reverse_sort kernel for real and imaginary arrays
    bit_reverse_sort <<< dimGrid, dimBlock >>>(tmp, x_r_d, MM,  N/2);
    copy_array <<< dimGrid, dimBlock >>>(x_r_d,tmp,  N/2);
    bit_reverse_sort <<< dimGrid, dimBlock >>>(tmp, x_i_d, MM,  N/2);
    copy_array <<< dimGrid, dimBlock >>>(x_i_d,tmp,  N/2);
    // Free device memory for temporary array
    cudaFree(tmp1);}
	else{
		
	}

}



// Function that calls different sorting and FFT kernels depending on whether M is even or odd
void gpuKernel(float* x_r_d, float* x_i_d, const unsigned int N, const unsigned int M)
{
 	
	// Check if M is even or odd
	sort_number(x_r_d, x_i_d, N, M,M%2);
	if(M%2 == 0)
    {
	// Allocate device memory for temporary array
	
    // Define grid and block dimensions
    dim3 dimGrid1((N / (16*256)), 8, 1);
	dim3 dimBlock1(128, 1, 1);

	// Loop over different FFT sizes and call fft_radix4 kernel for each size
	for (int i=1; i<N; i=i*4)  
	    {
	        fft_radix4 <<< dimGrid1, dimBlock1 >>>(x_r_d, x_i_d, N, i, 0);
		}

    }
	

    else
    {


        // Define grid and block dimensions
        dim3 dimGrid((N / (32*256)), 8, 1);
	    dim3 dimBlock(128, 1, 1);

        // Loop over different FFT sizes and call fft_radix4 kernel for each size and subarray
        for (int i=1; i<N/2; i*=4)  
	    {
	        fft_radix4 <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, N, i, 0);
	        fft_radix4 <<< dimGrid, dimBlock >>>(x_r_d, x_i_d, N, i, N/2);
		}

        // Define grid and block dimensions
        dim3 dimGrid1((N / (1024*256*2)), 32, 32);
	    dim3 dimBlock1(256, 1, 1);

        // Call fft_radix2 kernel for the final FFT size
        fft_radix2  <<< dimGrid1, dimBlock1 >>>(x_r_d, x_i_d, N, N/2);
        
    }

	
}
