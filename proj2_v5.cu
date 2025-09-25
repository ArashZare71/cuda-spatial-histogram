/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define BOX_SIZE	23000	 // Size of the data box on one dimension

double *h_x, *h_y, *h_z;	// pointers to the host data

unsigned int	PDH_acnt;	// Total number of data points
int num_buckets;			// Total number of buckets in the histogram
float   PDH_res;			// Width of each bucket

// Devicd constant memory variables
__constant__ unsigned int PDH_acnt_CUDA;	// Total number of data points on the device
__constant__ float PDH_res_CUDA;			// Width of each bucket on the device

// typedef struct hist_entry{
// 	unsigned int d_cnt;   /* need a long long type as the count might be huge */
// } bucket;
// bucket *histogram;		// pointers to the histogram
unsigned int *histogram;	// pointers to the histogram

// Time tracking CPU kernel
struct timezone Idunno;	
struct timeval startTime, endTime;

/* 
	reporting running time in seconds for the CPU version
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/* 
	Brute force on CPU 
*/
void PDH_baseline()
{
	// compute pairwise distances
	for(int i = 0; i < PDH_acnt; i++)
	{
		for(int j = i+1; j < PDH_acnt; j++)
		{
			double dx = h_x[i] - h_x[j];
			double dy = h_y[i] - h_y[j];
			double dz = h_z[i] - h_z[j];
			double dist = sqrt(dx*dx + dy*dy + dz*dz);
			int bin = (int)(dist / PDH_res);
			histogram[bin]++;
		}
	}
}

/* 
	Output the histogram and bucket contents
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15llu ", histogram[i]);
		total_cnt += histogram[i];
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}



/*==========================================================
Programmer: Ari Zare
GPU device funcition for computing the histogram distances
This function is same as p2p_distance but for GPU
===========================================================*/


/*
	SDH solution for GPU multi-threading
*/
__global__ void PDH_kernel(double *d_x, double *d_y, double *d_z,
    unsigned int *d_hist, int block_size ,int block_number, int num_buckets)
{
    int b = blockIdx.x;
    int B = blockDim.x;
    int t = threadIdx.x;
    int tid = b * B + t;
	// Early out if the thread is out of the range
    if (tid >= PDH_acnt_CUDA) return;

    extern __shared__ double sharedMemory[];	// shared memory for each block
	// Left Array
    double *L_x = (double*)sharedMemory;
    double *L_y = &L_x[block_size];
    double *L_z = &L_y[block_size];
	// Right Array
    double *R_x = &L_z[block_size];
    double *R_y = &R_x[block_size];
    double *R_z = &R_y[block_size];
	// Shared memory for histogram counts
    int *SHMOut = (int*)&R_z[block_size];  // Histogram storage
	
	/* Zero out the histogram shared memory */
    for(int i = 0; i < num_buckets; i++)
    {
        SHMOut[i] = 0;
    }
	/*	Intialize the left array with current thread's data points */
    L_x[t] = d_x[tid];
    L_y[t] = d_y[tid];
    L_z[t] = d_z[tid];
	/*	Initialze the registers for faster access */
    register double reg_x = L_x[t];
    register double reg_y = L_y[t];
    register double reg_z = L_z[t];

    int h_pos = 0;		// histogram bucket index
    double dist = 0.0;	// Distance
    double invPDH_res = 1.0 / PDH_res_CUDA;	// Precompute the inverse of the bucket width

	__syncthreads();

	/* 
		First loop: For each subsequent block (j), load the right array data and compute the
    	distances between the current thread's point (in left array) and the points in the right block. 
	*/
	for(int j = b+1; j < block_number; j++)
		{
			/* Load right block data into shared memory */
            R_x[t] = d_x[j * B + t];
            R_y[t] = d_y[j * B + t];
            R_z[t] = d_z[j * B + t];
            __syncthreads();
			/* 
				Inner loop: Compute the distance between the current point and each candidate point
           		in the right block. Note: This loop uses the same temporary value for each iteration.
			 */
            for(int k=0; k < block_size && j * B + k < PDH_acnt_CUDA; k++)
            {
				double dx = reg_x - R_x[k];	// Compute distance in x dim
				double dy = reg_y - R_y[k];	// Compute distance in y dim
				double dz = reg_z - R_z[k];	// Compute distance in z dim

				dist = sqrt(dx*dx + dy*dy + dz*dz);
                h_pos = (int) (dist * invPDH_res);
                atomicAdd(&SHMOut[h_pos], 1);
            }
            __syncthreads();
            }
        /*
			Second loop: Compute distances among points within the same block (avoiding duplicate calculations).
		*/            
        for (int i = t + 1; i < block_size && b * B + i < PDH_acnt_CUDA; i++)
        {
			double dx = reg_x - L_x[i];	// Compute distance in x dim
			double dy = reg_y - L_y[i];	// Compute distance in y dim
			double dz = reg_z - L_z[i];	// Compute distance in z dim

			dist = sqrt(dx*dx + dy*dy + dz*dz);
            h_pos = (int) (dist * invPDH_res);
            atomicAdd(&SHMOut[h_pos], 1);
        }
        __syncthreads();
	/*
		After all threads have computed their partial histograms, thread 0 adds the shared histogram
    	into the global histogram.
	*/
	if(threadIdx.x== 0)   
	{
		for(int b = 0; b < num_buckets; b++)
		{
			atomicAdd(&d_hist[b],SHMOut[b]);
		}
	}
}

/*============================================================================================
Outputing the result of the histogram. Its same as output_histogram but with a parameter.
The out_put_histogram by default is showing the histogram
of the global variable histogram, but this function is for showing the histogram 
that is created by the GPU for comparisions, and did not manipulate
the original histogram.
============================================================================================*/
void output_histogram_param(unsigned int *hist, int n_buckets){
	int i;
    long long total_cnt = 0;
    for(i = 0; i < n_buckets; i++) {
        if(i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15llu ", hist[i]);
        total_cnt += hist[i];
        if(i == n_buckets - 1)
            printf("\n Total Count: %lld\n", total_cnt);
        else
            printf("| ");
    }
}

/* Comparing the Histogram calculated by the host (CPU) and device */
void compare_histograms(unsigned int *cpu_hist, unsigned int *gpu_hist) {
    printf("\nDifference between CPU and GPU histograms:\n");
	int num_dif = 0;
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i] - gpu_hist[i];
		if (diff != 0)
			num_dif++;
        if (i % 5 == 0)
            printf("\n%02d: ", i);
        printf("%15llu ", diff);
        if (i != num_buckets - 1)
            printf("| ");
    }
    printf("\n");
	printf("Number of differences: %d\n", num_dif);
}


int main(int argc, char **argv)
{
	if (argc < 4){
        fprintf(stderr, "Usage: %s <num_points> <bucket_width> <block_size>\n", argv[0]);
        exit(1);
	}

    // parse arguments
    PDH_acnt = atoi(argv[1]);
    PDH_res= atof(argv[2]);
    int threadsPerBlock = atoi(argv[3]);

    if(PDH_acnt <= 0 || PDH_res <= 0.0f || threadsPerBlock <= 0 || threadsPerBlock >= 1000)
    {
        fprintf(stderr, "ERROR: invalid input parameter(s)\n");
        exit(1);
    }

	// Allocate memory for the data arrays
	unsigned int *h_histogram;	// Host histogram, for device to host memory copy
	unsigned int *d_histogram;	// Device histogram

	double *d_x, *d_y, *d_z;	// Device data points

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	printf("Number of buckets: %d\n", num_buckets);
	
	/* Allocate memory for host(CPU) histogram , and device (on host) */
	histogram = (unsigned int *)malloc(sizeof(unsigned int) * num_buckets);
	h_histogram = (unsigned int *)malloc(sizeof(unsigned int)*num_buckets);
	/* Allocate host data points*/
    h_x = (double*) malloc(PDH_acnt * sizeof(double));
    h_y = (double*) malloc(PDH_acnt * sizeof(double));
    h_z = (double*) malloc(PDH_acnt * sizeof(double));

	/* Allocate memory on the device for data points and histogram */		
	cudaMalloc((void **)&d_histogram, sizeof(unsigned int)*num_buckets);
	cudaMemset(d_histogram, 0, sizeof(unsigned int)*num_buckets);
	cudaMalloc((void **)&d_x, sizeof(double)*PDH_acnt);
	cudaMalloc((void **)&d_y, sizeof(double)*PDH_acnt);
	cudaMalloc((void **)&d_z, sizeof(double)*PDH_acnt);

    // Generate random data points
	srand(1);

	for(int i = 0; i < PDH_acnt; i++)
	{
		h_x[i] = ((float)rand() / RAND_MAX) * BOX_SIZE;
		h_y[i] = ((float)rand() / RAND_MAX) * BOX_SIZE;
		h_z[i] = ((float)rand() / RAND_MAX) * BOX_SIZE;
	}

	// copy the data to the device
	cudaMemcpy(d_x,h_x,sizeof(double)*PDH_acnt,cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,sizeof(double)*PDH_acnt,cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,h_z,sizeof(double)*PDH_acnt,cudaMemcpyHostToDevice);
 	// Copy Necessary parameters to the device constant memory
    cudaMemcpyToSymbol (PDH_acnt_CUDA, &PDH_acnt, sizeof(unsigned int));
	cudaMemcpyToSymbol (PDH_res_CUDA, &PDH_res, sizeof(float));


	// Start timer
	gettimeofday(&startTime, &Idunno);
	/* call CPU brute force function */
	PDH_baseline();
	// Stop timer and output the timer results 
	report_running_time();
	
	/* print out the CPU histogram */
	output_histogram();

    int block_number = ceil(PDH_acnt / threadsPerBlock) + 1;
	/*
		Calculate shared memory size for the GPU kernel:
       - 6 arrays (for left and right x, y, z coordinates) each of size 'threadsPerBlock' (of type double)
       - 1 array for the histogram (of type int) with 'num_buckets' elements.
	*/
    int shared_mem = (threadsPerBlock * 6)* sizeof(double) + num_buckets * sizeof(int);
    
	// Start the timer for GPU version
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	// Call the GPU kernel
	PDH_kernel<<<block_number, threadsPerBlock, shared_mem>>>(d_x,d_y,d_z,d_histogram,threadsPerBlock, block_number, num_buckets);

	// Stop the timer for GPU version
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Running time for GPU version: %0.5f ms\n", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();

	// Copy the histogram from the device to the host
	cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int)*num_buckets, cudaMemcpyDeviceToHost);

	printf("\n GPU histogram\n");
	output_histogram_param(h_histogram, num_buckets);

	// Compare the CPU and GPU histograms, and print the differences
	compare_histograms(histogram, h_histogram);

	// free the memory allocated on the device
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_histogram);

	// free the memory allocated on the host
	free(h_x);
	free(h_y);
	free(h_z);
	free(histogram);
	free(h_histogram);
	
	return 0;
}

