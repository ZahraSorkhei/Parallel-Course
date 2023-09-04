//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block

#define TILEX 32
#define TILEY 16


#define MIN (TILEX <= TILEY ? TILEX : TILEY)

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}

dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

	
	
	int i = by * TILEY + ty;
	int j = bx * TILEX + tx;
	

	float sum = 0;
	int index1,index2,index3,index4;
	
	__shared__ float a[TILEY][4*MIN];
	__shared__ float b[4*MIN][TILEX];

	for (int z = 0 ; z < (n/MIN) ; z += 4){
		

		if(tx < TILEY){
			index1 = (i<< m) + (MIN*z+tx);
			a[ty][tx] = ad[index1];
		}
		
		if(tx < TILEY){
			index2 = (i<< m) + (MIN*(z+1)+tx);
			a[ty][MIN + tx ] = ad[index2];
		}
		
		if(tx < TILEY){
			index3 = (i<< m) + (MIN*(z+2)+tx);
			a[ty][2*MIN + tx] = ad[index3];
		}
		
		if(tx < TILEY){
			index4 = (i << m) + (MIN*(z+3)+tx);
			a[ty][3*MIN + tx] = ad[index4];
		}

		

		if (ty < TILEX){
			index1 = (MIN*z + ty << m) + (j);
			b[ty][tx] = bd[index1];
		}

		if (ty < TILEX){
			index2 = (MIN * (z+1) + ty << m) + (j);
			b[MIN+ty][tx] = bd[index2];
		}
		
		if (ty < TILEX){
			index3 = (MIN * (z+2) + ty << m) + (j);
			b[2*MIN+ty][tx] = bd[index3];

		}
		
		if (ty < TILEX){
			index4 = (MIN * (z+3) + ty << m) + (j);
			b[3*MIN+ty][tx] = bd[index4];		
		}
		
	
		__syncthreads();
		

		for (int u = 0 ; u < 4 * MIN ; u+=1){ 
			sum = sum + a[ ty ][ u ] * b[ u][ tx ];
		}
		__syncthreads();
	}
	
	int index = (i << m) + j;
	cd[index] = sum;
}