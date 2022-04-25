#include<cuda.h>
#include "cuda_sort.h"

typedef struct {
	//unsigned long int physical_id;
	char *physical_id;
	//unsigned long int virtual_address[NUM_PAGES];
	char *virtual_address[NUM_PAGES];
	uint32_t hashes[NUM_PAGES];

}page_hash;

typedef struct {

	char *physical_id;
	char *virtual_address[NUM_PAGES];
	char data[NUM_PAGES][PAGE_SIZE];
}page_data;

typedef struct {
	//unsigned long int matched_address[NUM_PAGES][100];
	char *matched_address1[NUM_PAGES][100];
	char *matched_address2[NUM_PAGES][100];
	unsigned short matched_count[NUM_PAGES]; //shared count per page
}matched;

__device__ void dev_strcpy(char* destination, char *source, int length,int offset) {
	int i=0;
	for(i=0;i<length;i++) {
		destination[offset+i]=source[i];
	}

}

__global__ void exchange(page_hash *data,int *zero,int *one,int iteration, page_hash *rst,int total)
	{
	 int id=blockIdx.x*blockDim.x+threadIdx.x;
	 if(id<total)	
	 	{
	 	 if((data->hashes[id]>>iteration) & 1)  {
		 	rst->hashes[one[id]]=data->hashes[id];
			rst->virtual_address[one[id]]=data->virtual_address[id];
		 }

	 	 else  {
		 	rst->hashes[zero[id]]=data->hashes[id];
			rst->virtual_address[zero[id]]=data->virtual_address[id];
		 }
	 	}	

	}

//Function calculating the prefix sum

__global__ void prefixSum(int *zero,int *one, int *backzero,int *backone,int total,int iteration)
	{	
	 int id=blockIdx.x*blockDim.x+threadIdx.x;
	 if((id<total) && (id-iteration)>=0)
		{
       		 backone[id]=one[id-iteration]+one[id];
		 backzero[id]=zero[id-iteration]+zero[id];
		}

	}

//Function for writing results

__global__ void writeback(page_hash *data, page_hash *rst, int total)
	{
	 int id = blockIdx.x*blockDim.x + threadIdx.x;
	 if(id<total)
		{
		 data->hashes[id] = rst->hashes[id];
		 data->virtual_address[id]=rst->virtual_address[id];
		}
	}

__global__ void addrank(int *one,int *zero, int val, int N)
	{
	 int id=blockIdx.x*blockDim.x+threadIdx.x;
		if(id<N)
		{	 one[id]=one[id]+val-1;
		  	 zero[id] = zero[id]-1;
		}
	}

//Function for writing the bit values

__global__ void radix(page_hash *data,int *zero,int *one,int iteration,int total)
	{
         int id=blockIdx.x*blockDim.x+threadIdx.x;	 
	 if(id<total) 
	 	{
	 	 if((data->hashes[id]>>iteration) & 1)
			{
	 	 	 one[id]=1;
		 	 zero[id]=0;	 
			}
		else
			{
		 	 one[id]=0;
		 	 zero[id]=1;
			}
		}
	}

void radix_sort() {
	
       	int bound=0,*value,NODES;
	 //int *data_h,*data_d;
	 page_hash *rst;
	 int *bitone_d,*bitzero_d,*backzero_d,*backone_d;
	 int blocks,iteration,jump;
	 double begin,end,runtime;
	 /*if(argc<3)
		{
	 	 printf("\n---!!Filename required as -f argument!!--\n");
		 return 0;
		}
 	 while(fscanf(fin,"%d",&read)!=EOF)
		 records++;*/
	 NODES=NUM_PAGES;
//	 printf("NODES-->%d",records);
	 //rewind(fin);
	 //data_h=(int*)malloc(NODES*sizeof(int));
	 bound=ceil((double)log(NODES)/(double)log(2));
	 printf("\nbound=%d\n",bound);
	 value=(int*)malloc(sizeof(int));
	 srandom(clock());
	 //for(i=0;i<NODES;i++)
	 //	  data_h[i]=rand()%NODES;
	 //cudaMalloc(&data_d,NODES*sizeof(int));
	 cudaMalloc(&bitone_d,NODES*sizeof(int));
	 cudaMalloc(&bitzero_d,NODES*sizeof(int));
	 cudaMalloc(&backone_d,NODES*sizeof(int));
	 cudaMalloc(&backzero_d,NODES*sizeof(int));
	 cudaMalloc(&rst,sizeof(page_hash));
	 //cudaMemcpy(data_d,data_h,NODES*sizeof(int),cudaMemcpyHostToDevice);
	 int val;
	 blocks=ceil((double)NODES/(double)512);
	 
	 printf("blocks-->%d\n",blocks);
	 begin=clock();
	 for(i=0;i<32;i++)
	 	{
	 	 iteration=0;
	 	 jump=1;
	 	 radix<<<blocks,512>>>(page_hashes_vm1_dev,bitzero_d,bitone_d,i,NODES);
	 	 while(iteration<bound)
              		{
			cudaMemcpy(backzero_d,bitzero_d,NODES*sizeof(int),cudaMemcpyDeviceToDevice);
			cudaMemcpy(backone_d,bitone_d,NODES*sizeof(int),cudaMemcpyDeviceToDevice);
                	prefixSum<<<blocks,512>>>(bitzero_d,bitone_d,backzero_d,backone_d,NODES,jump);
			cudaMemcpy(bitzero_d,backzero_d,NODES*sizeof(int),cudaMemcpyDeviceToDevice);
                        cudaMemcpy(bitone_d,backone_d,NODES*sizeof(int),cudaMemcpyDeviceToDevice);
                 	jump=2*jump;
                 	iteration++;
                	}
		 cudaMemcpy(value,(bitzero_d+NODES-1),sizeof(int),cudaMemcpyDeviceToHost);
		 val = *value;
	 	 addrank<<<blocks,512>>>(bitone_d,bitzero_d, val, NODES);
		 exchange<<<blocks,512>>>(page_hashes_vm1_dev,bitzero_d,bitone_d,i,rst,NODES);
		 writeback<<<blocks, 512>>>(page_hashes_vm1_dev, rst, NODES);
		}
	 end=clock();
	 runtime=(end-begin)/(CLOCKS_PER_SEC);
	 //cudaMemcpy(data_h,data_d,NODES*sizeof(int),cudaMemcpyDeviceToHost);
	 /*printf("\n--new data written in file named data--\n");

	 //Writing the result to files

	 FILE *fout;
	 fout=fopen("data","w");
 	 for(i=0;i<NODES;i++)
	 	{
		 fprintf(fout,"%d\n",data_h[i]);
		}
	 printf("Runtime=%lf\n",runtime);
	 fclose(fout);*/
	// cudaFree(data_d);
	 cudaFree(bitone_d);
	 cudaFree(bitzero_d);
	 cudaFree(backzero_d);
	 cudaFree(backone_d);
	 cudaFree(rst);
	//cudaDeviceSynchronize();
	cudaMemcpy(page_hashes_vm1_cpu,page_hashes_vm1_dev,sizeof(page_hash),cudaMemcpyDeviceToHost);

}
