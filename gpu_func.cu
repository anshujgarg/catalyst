#include<cuda.h>
#include<stdint.h>
#include<stdio.h>
#include "gpu_func.h"
#include<cuda_runtime.h>

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)+(uint32_t)(((const uint8_t *)(d))[0]))

#define PAGE_SIZE 4096
#define PAGE_SHIFT 12
#define THREADS_PER_BLOCK 512


#ifdef DBG
#define dprint printf
#else
#define dprint(x,...)   
#endif

#define sprint printf

#ifdef TIME
#define tprint printf
#else
#define tprint(x,...)
#endif


#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

#if defined(__i386__)

static __inline__ unsigned long long rdtsc(void)
{
	  unsigned long long int x;
	       __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
	            return x;
}
#elif defined(__x86_64__)


static __inline__ unsigned long long rdtsc(void)
{
	  unsigned hi, lo;
	    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	      return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#endif


inline void cudaAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

   }
}


__global__  void gpu_fast_page_hash(char *page_data, page_hash *page_hash_gpu,int page_size,int total_threads) {

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t len = PAGE_SIZE;
	uint32_t hash ,tmp;
	char *temp_ptr;
	hash=len;
	int rem;

	int thread_page_index=thread_id*PAGE_SIZE;
	//	page_hashes->physical_id=data->physical_id;
	temp_ptr=page_data+thread_page_index;
	//	temp_ptr=data->data[thread_id];

	//	int test=0;
	//	if(test==1) {	
	if(thread_id<total_threads) {
		//if (len <= 0 || data == NULL) return 0;
		rem = len & 3; // And operation netween PAGE_SIZE and 11 (which is 3)
		len >>= 2; // left shift page size. 4096>>2 = 1024 
		/* Main loop */
		for (;len > 0; len--) {  //will run for 1024 times. 
			hash  += get16bits (temp_ptr);  //Sum first two bytes of data char array
			tmp    = (get16bits (temp_ptr+2) << 11) ^ hash;  //sum of 3rd and 4th byte of data char array xored with sum of 1st two bytes
			hash   = (hash << 16) ^ tmp;
			temp_ptr  += 2*sizeof (uint16_t);
			hash  += hash >> 11;
		}

		/* Handle end cases */

		switch (rem) {
		case 3: hash += get16bits (temp_ptr);
		hash ^= hash << 16;
		hash ^= ((signed char)temp_ptr[sizeof (uint16_t)]) << 18;
		hash += hash >> 11;
		break;
		case 2: hash += get16bits (temp_ptr);
		hash ^= hash << 11;
		hash += hash >> 17;
		break;
		case 1: hash += (signed char)*temp_ptr;
		hash ^= hash << 10;
		hash += hash >> 1;
		}		

		/* Force "avalanching" of final 127 bits */

		hash ^= hash << 3;
		hash += hash >> 5;
		hash ^= hash << 4;
		hash += hash >> 17;
		hash ^= hash << 25;
		hash += hash >> 6;
		//	page_hashes->hashes[thread_id]=hash;
		//	page_hashes->virtual_address[thread_id]=data->virtual_address[thread_id];
		//	page_hash[thread_id]=hash;
		page_hash_gpu[thread_id].hash_value=hash;
		//return hash; Instead of returning store this value in some array
		//	page_hash[thread_id]=thread_id;
	}
}


void compute_hash(char *page_data,page_hash *page_hash_gpu,int page_size,unsigned int total_threads,unsigned long long *trans_data_gpu)
{
	unsigned long size;
	unsigned int num_blocks;
	char *page_data_gpu;
	unsigned long long cycles_trans_data_gpu=0,cycle_begin=0,cycle_end=0;
	
	num_blocks=(total_threads/THREADS_PER_BLOCK) + 1;
	size= total_threads*PAGE_SIZE;

	cycle_begin=rdtsc();
	cudaErrorCheck(cudaMalloc((void**)&page_data_gpu,size*sizeof(char)));
	cudaErrorCheck(cudaMemcpy(page_data_gpu,page_data,size,cudaMemcpyHostToDevice));
	cycle_end=rdtsc();

	cycles_trans_data_gpu=cycle_end-cycle_begin;
	*trans_data_gpu=*trans_data_gpu+cycles_trans_data_gpu;

	gpu_fast_page_hash<<<num_blocks,THREADS_PER_BLOCK>>>(page_data_gpu,page_hash_gpu,PAGE_SIZE,total_threads);
	cudaErrorCheck(cudaFree(page_data_gpu));
}





__global__ void exchange(page_hash *page_hash_gpu,int *zero,int *one,int iteration, page_hash *rst,int total)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<total)	
	{
		if((page_hash_gpu[id].hash_value>>iteration) & 1)  {
			rst[one[id]].hash_value=page_hash_gpu[id].hash_value;
			//rst->virtual_address[one[id]]=data->virtual_address[id];
			//rst[one[id]].cuda_virt_address=page_hash_gpu[id].cuda_virt_address;
			rst[one[id]].vm_virt_address=page_hash_gpu[id].vm_virt_address;
		}

		else  {
			// 	rst->hashes[zero[id]]=data->hashes[id];
			//		rst->virtual_address[zero[id]]=data->virtual_address[id];
			rst[zero[id]].hash_value=page_hash_gpu[id].hash_value;
			//rst->virtual_address[one[id]]=data->virtual_address[id];
			//rst[zero[id]].cuda_virt_address=page_hash_gpu[id].cuda_virt_address;
			rst[zero[id]].vm_virt_address=page_hash_gpu[id].vm_virt_address;
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

__global__ void writeback(page_hash *page_hash_gpu, page_hash *rst, int total)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	if(id<total)
	{
		//data->hashes[id] = rst->hashes[id];
		//data->virtual_address[id]=rst->virtual_address[id];
		page_hash_gpu[id].hash_value=rst[id].hash_value;
		//page_hash_gpu[id].cuda_virt_address=rst[id].cuda_virt_address;
		page_hash_gpu[id].vm_virt_address=rst[id].vm_virt_address;

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

__global__ void radix(page_hash *page_hash_gpu,int *zero,int *one,int iteration,int total)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;	 
	if(id<total) 
	{
		if((page_hash_gpu[id].hash_value>>iteration) & 1)
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

__global__ void exchange_virt_addr(page_hash *page_hash_gpu,int *zero,int *one,int iteration, page_hash *rst,int total)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<total)	
	{
		if((page_hash_gpu[id].vm_virt_address>>iteration) & 1)  {
			rst[one[id]].hash_value=page_hash_gpu[id].hash_value;
			//rst->virtual_address[one[id]]=data->virtual_address[id];
			//rst[one[id]].cuda_virt_address=page_hash_gpu[id].cuda_virt_address;
			rst[one[id]].vm_virt_address=page_hash_gpu[id].vm_virt_address;
		}

		else  {
			// 	rst->hashes[zero[id]]=data->hashes[id];
			//		rst->virtual_address[zero[id]]=data->virtual_address[id];
			rst[zero[id]].hash_value=page_hash_gpu[id].hash_value;
			//rst->virtual_address[one[id]]=data->virtual_address[id];
			//rst[zero[id]].cuda_virt_address=page_hash_gpu[id].cuda_virt_address;
			rst[zero[id]].vm_virt_address=page_hash_gpu[id].vm_virt_address;
		}
	}	

}



//Function for writing the bit values


__global__ void radix_virt_addr(page_hash *page_hash_gpu,int *zero,int *one,int iteration,int total)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;	 
	if(id<total) 
	{
		if((page_hash_gpu[id].vm_virt_address>>iteration) & 1)
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



//Sort the hash list by virtual addresses

void cuda_sort_virt_addr(page_hash *page_hash_gpu,int total_threads) 
{
	page_hash *rst;
	int *bitone_d,*bitzero_d,*backzero_d,*backone_d;
	int blocks,iteration,jump;
	int bound=0,*value,NODES;
	int i;
	int val;

	NODES=total_threads;
	bound=ceil((double)log(NODES)/(double)log(2));
	dprint("\nbound=%d\n",bound);
	value=(int*)malloc(sizeof(int));
	srandom(clock());
	//for(i=0;i<NODES;i++)
	//	  data_h[i]=rand()%NODES;
	//cudaMalloc(&data_d,NODES*sizeof(int));
	cudaMalloc(&bitone_d,NODES*sizeof(int));
	cudaMalloc(&bitzero_d,NODES*sizeof(int));
	cudaMalloc(&backone_d,NODES*sizeof(int));
	cudaMalloc(&backzero_d,NODES*sizeof(int));
	cudaMalloc((void**)&rst,NODES*sizeof(page_hash));
	//cudaMemcpy(data_d,data_h,NODES*sizeof(int),cudaMemcpyHostToDevice);
	blocks=ceil((double)NODES/(double)512);

	dprint("blocks-->%d\n",blocks);
	//	begin=clock();
	for(i=0;i<64;i++)
	{
		iteration=0;
		jump=1;
		radix_virt_addr<<<blocks,512>>>(page_hash_gpu,bitzero_d,bitone_d,i,NODES);
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
		exchange_virt_addr<<<blocks,512>>>(page_hash_gpu,bitzero_d,bitone_d,i,rst,NODES);
		writeback<<<blocks, 512>>>(page_hash_gpu, rst, NODES);
	}
	//	end=clock();
	//	sort_time=(end-begin)/(CLOCKS_PER_SEC);
	//	tprint("\nTime to sort the hashes is %.10lf\n",sort_time);
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

}


void cuda_sort(page_hash *page_hash_gpu,int total_threads) 
{
	page_hash *rst;
	int *bitone_d,*bitzero_d,*backzero_d,*backone_d;
	int blocks,iteration,jump;
	int bound=0,*value,NODES;
	int i;
	int val;

	
	NODES=total_threads;
	bound=ceil((double)log(NODES)/(double)log(2));
	dprint("\nbound=%d\n",bound);
	value=(int*)malloc(sizeof(int));
	srandom(clock());
	//for(i=0;i<NODES;i++)
	//	  data_h[i]=rand()%NODES;
	//cudaMalloc(&data_d,NODES*sizeof(int));
	cudaMalloc(&bitone_d,NODES*sizeof(int));
	cudaMalloc(&bitzero_d,NODES*sizeof(int));
	cudaMalloc(&backone_d,NODES*sizeof(int));
	cudaMalloc(&backzero_d,NODES*sizeof(int));
	cudaMalloc((void**)&rst,NODES*sizeof(page_hash));
	//cudaMemcpy(data_d,data_h,NODES*sizeof(int),cudaMemcpyHostToDevice);
	blocks=ceil((double)NODES/(double)512);

	dprint("blocks-->%d\n",blocks);
	//	begin=clock();
	for(i=0;i<32;i++)
	{
		iteration=0;
		jump=1;
		radix<<<blocks,512>>>(page_hash_gpu,bitzero_d,bitone_d,i,NODES);
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
		exchange<<<blocks,512>>>(page_hash_gpu,bitzero_d,bitone_d,i,rst,NODES);
		writeback<<<blocks, 512>>>(page_hash_gpu, rst, NODES);
	}
	//printf("cuda sort done \n");
	//	end=clock();
	//	sort_time=(end-begin)/(CLOCKS_PER_SEC);
	//	tprint("\nTime to sort the hashes is %.10lf\n",sort_time);
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

}



//Function calculating the prefix sum for is_virt_addr_present

__global__ void prefixSum_virt_addr_present(int *is_virt_addr_present,int *back_is_virt_addr_present,int total,int iteration)
{	
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	if((id<total) && (id-iteration)>=0)
	{
		back_is_virt_addr_present[id]=is_virt_addr_present[id-iteration]+is_virt_addr_present[id];

	}

}


__global__ void copy_hints(page_hash *page_hash_gpu, page_hash *hints_gpu,int *is_a_hint,int *is_a_hint_prefix_sum,int num) 
{
	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int rank;
	
	if(thread_id<num)
	{
		rank=is_a_hint_prefix_sum[thread_id]-1;
		if(is_a_hint[thread_id]==1)
		{
			hints_gpu[rank].vm_virt_address=page_hash_gpu[thread_id].vm_virt_address;
			//hints_gpu[rank].cuda_virt_address=page_hash_gpu[thread_id].cuda_virt_address;
			hints_gpu[rank].hash_value=page_hash_gpu[thread_id].hash_value;
		}
	}

}


__global__ void generate_hints_gpu(page_hash *page_hash_gpu,int *is_a_hint,int num) 
{

	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(thread_id<num) 
	{
		//is_a_hint[thread_id]=0;
		if(thread_id>0 && thread_id<(num-1))
		{	
			if(page_hash_gpu[thread_id].hash_value==page_hash_gpu[thread_id+1].hash_value || page_hash_gpu[thread_id].hash_value==page_hash_gpu[thread_id-1].hash_value) 
			{
				is_a_hint[thread_id]=1;
			}
		}
		else if(thread_id==0)
		{
			if(page_hash_gpu[thread_id].hash_value==page_hash_gpu[thread_id+1].hash_value)
			{
				is_a_hint[thread_id]=1;
			}
		}
		else
		{
			if(page_hash_gpu[thread_id].hash_value==page_hash_gpu[thread_id-1].hash_value)
			{
				is_a_hint[thread_id]=1;
			}
		}



	}

}

__global__ void hints_using_diff_map(int *is_a_hint,int *diff_map_gpu,int num)
{

	int thread_id= blockDim.x*blockIdx.x + threadIdx.x;
	if(thread_id<num)
	{
		if(diff_map_gpu[thread_id]==1)
			is_a_hint[thread_id]=1;
	}

}

void generate_hints(page_hash *page_hash_gpu,int *is_a_hint_gpu,int num) 
{
	unsigned int num_blocks;
	num_blocks=ceil((double)num/(double)THREADS_PER_BLOCK);
	//cuda_sort(page_hash_gpu,num);
	generate_hints_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu,is_a_hint_gpu,num);
}	

int prefix_sum(int *is_a_hint,int *is_a_hint_prefix_sum,int num) 
{
	
	//int *is_a_hint_prefix_sum;
	int *back_is_a_hint_prefix_sum;
	unsigned int num_blocks;
	int total_hints=0;
	int bound=0;
	int iteration=0;
	int jump=1;

	num_blocks=ceil((double)num/(double)THREADS_PER_BLOCK);

	//cudaErrorCheck(cudaMalloc((void**)&is_a_hint_prefix_sum,num*sizeof(int)));
	cudaErrorCheck(cudaMalloc((void**)&back_is_a_hint_prefix_sum,num*sizeof(int)));
	
	cudaErrorCheck(cudaMemcpy(is_a_hint_prefix_sum,is_a_hint,num*sizeof(int),cudaMemcpyDeviceToDevice));
	
	bound=ceil((double)log(num)/(double)log(2));
	//printf("Bound =%d \n",bound);

	//comuting prefix sum in GPU

	while(iteration<bound) {
		cudaMemcpy(back_is_a_hint_prefix_sum,is_a_hint_prefix_sum,num*sizeof(int),cudaMemcpyDeviceToDevice);
		prefixSum_virt_addr_present<<<num_blocks,512>>>(is_a_hint_prefix_sum,back_is_a_hint_prefix_sum,num,jump);
		cudaMemcpy(is_a_hint_prefix_sum,back_is_a_hint_prefix_sum,num*sizeof(int),cudaMemcpyDeviceToDevice);
		iteration++;
		jump=2*jump;


	}

	cudaErrorCheck(cudaFree(back_is_a_hint_prefix_sum));
/*
	int *prefix_sum;
	prefix_sum=(int*)malloc(num*sizeof(int));
	cudaErrorCheck(cudaMemcpy(prefix_sum,is_a_hint_prefix_sum,num*sizeof(int),cudaMemcpyDeviceToHost));
	int i=0;
	for(i=0;i<num;i++)
	{
		printf("prefix_sum %d= %d \n" ,i+1,prefix_sum[i]);
	}*/

	cudaErrorCheck(cudaMemcpy(&total_hints,is_a_hint_prefix_sum+num-1,sizeof(int),cudaMemcpyDeviceToHost));
	
	
	//cudaErrorCheck(cudaFree(is_a_hint_prefix_sum));
	return total_hints;
	
}

/*Compare the hashes of two VMs
 * VM1 should be sorted by virtual address
 * is_a_hint for each VM tells whether element at that index is a hint or not
 */



__device__ int binary_search_hash(page_hash *page_hash_gpu_vm1, uint32_t hash_value_vm2, int *is_a_hint_vm1,int num_pages_vm1) 
{
	
	int low=0,mid=0,high = num_pages_vm1;
	while(low<=high)
	{
		mid= (high + low)/2;
		if(page_hash_gpu_vm1[mid].hash_value==hash_value_vm2)
		{
			is_a_hint_vm1[mid]=1;
			return 1;			
		}	
		else if(page_hash_gpu_vm1[mid].hash_value>hash_value_vm2)
			high=mid-1;

		else
			low = mid+1;
	}
	return 0;

}

__global__ void compare_hash_gpu(page_hash *page_hash_gpu_vm1,page_hash *page_hash_gpu_vm2,int *is_a_hint_vm1,int *is_a_hint_vm2,int num_pages_vm1,int num_pages_vm2)
{

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int found =0;
	if(thread_id<num_pages_vm2) 
	{
		found = binary_search_hash(page_hash_gpu_vm1,page_hash_gpu_vm2[thread_id].hash_value,is_a_hint_vm1,num_pages_vm1);
		if(found==1)
			is_a_hint_vm2[thread_id]=found;


	}
}


void compare_hash(page_hash* page_hash_gpu_vm1,page_hash *page_hash_gpu_vm2,int *is_a_hint_vm1,int *is_a_hint_vm2,int num_pages_vm1,int num_pages_vm2)
{
	unsigned int num_blocks;
	num_blocks=ceil((double)num_pages_vm2/(double)THREADS_PER_BLOCK);
		//cuda_sort(page_hash_gpu,num);
	compare_hash_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu_vm1,page_hash_gpu_vm2,is_a_hint_vm1,is_a_hint_vm2,num_pages_vm1,num_pages_vm2);
	
	
}

__global__ void copy_hints_gpu(page_hash *page_hash_gpu, page_hash *hints_gpu,int *is_a_hint,int *is_a_hint_prefix_sum,int num) 
{
	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int rank;
	
	if(thread_id<num)
	{
		rank=is_a_hint_prefix_sum[thread_id]-1;
		if(is_a_hint[thread_id]==1)
		{
			hints_gpu[rank].vm_virt_address=page_hash_gpu[thread_id].vm_virt_address;
			//hints_gpu[rank].cuda_virt_address=page_hash_gpu[thread_id].cuda_virt_address;
			hints_gpu[rank].hash_value=page_hash_gpu[thread_id].hash_value;
		}
	}

}

int copy_hints(page_hash *page_hash_gpu,unsigned long **hints,page_hash **current_hints_gpu,int *is_a_hint_gpu,int total_pages,unsigned long long *trans_hints_gpu)
{
	unsigned int num_blocks;
	int *is_a_hint_prefix;
	page_hash *hints_gpu,*hints_cpu;
	int total_hints=0;
	int i=0;
	unsigned long *temp_hints_ptr;
	unsigned long long cycles_trans_hints_gpu=0;
	unsigned long long cycle_begin=0,cycle_end=0;

	num_blocks=ceil((double)total_pages/(double)THREADS_PER_BLOCK);
	
		
	cudaErrorCheck(cudaMalloc((void**)&is_a_hint_prefix,total_pages*sizeof(int)));
	
	total_hints=prefix_sum(is_a_hint_gpu,is_a_hint_prefix,total_pages);
	//printf("total_hints =%d \n", total_hints);
	if(total_hints>0)
	{
		//hints_gpu=*current_hints_gpu;
		cudaErrorCheck(cudaMalloc((void **)&(hints_gpu),total_hints*sizeof(page_hash)));
		copy_hints_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu,hints_gpu,is_a_hint_gpu,is_a_hint_prefix,total_pages);
				
				
		cuda_sort_virt_addr(hints_gpu,total_hints);

		cycle_begin=rdtsc();

		hints_cpu=(page_hash*)malloc(total_hints*sizeof(page_hash));
		cudaErrorCheck(cudaMemcpy(hints_cpu,hints_gpu,total_hints*sizeof(page_hash),cudaMemcpyDeviceToHost));
		
		*hints=(unsigned long *)malloc(total_hints*sizeof(unsigned long));
		temp_hints_ptr=*hints;
		
		for(i=0;i<total_hints;i++)
		{
			temp_hints_ptr[i]=hints_cpu[i].vm_virt_address;
		}
		*current_hints_gpu=hints_gpu;

		cycle_end=rdtsc();
		cycles_trans_hints_gpu=cycle_end-cycle_begin;
		*trans_hints_gpu=*trans_hints_gpu+cycles_trans_hints_gpu;
		
		free(hints_cpu);
		


		cudaErrorCheck(cudaFree(is_a_hint_prefix));
		//cudaErrorCheck(cudaFree(hints_gpu)); //Free hints_gpu
	
		
		return total_hints;
	}
	
	return 0;	
}

__device__ int binary_search_virt_addr(page_hash *page_hash_gpu_old, unsigned long virtual_address,uint32_t hash_value, int old_hash_num) 
{
	/*
		   Binary search hash within the old hash list.
		   Make sure it doesn't be the same virtual address.  
	 */
	int low=0,mid=0,high = old_hash_num-1;
	while(low<=high)
	{
		mid= (high+low)/2;
		if(page_hash_gpu_old[mid].vm_virt_address==virtual_address)
		{
			page_hash_gpu_old[mid].hash_value=hash_value;
			return 0;

		}	
		else if(page_hash_gpu_old[mid].vm_virt_address>virtual_address)
			high=mid-1;

		else
			low = mid+1;
	}
	return 1;

}

__global__ void find_virt_addr(page_hash *page_hash_gpu_old,page_hash *page_hash_gpu_new,int num_threads,int *is_virt_addr_present,int old_hash_num)
{

	//page_hash_gpu -> current_hints_gpu
	//page_hash_gpu_old -> active_hints_gpu
	//num_old_hashes_val -> num_active_hints
	//num_thread ->hints_generated


//active_hints sorted by virtual addresses: perform binary search of Virt addr on it

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int found =0;
	if(thread_id<num_threads) 
	{
		is_virt_addr_present[thread_id]=1;
		found = binary_search_virt_addr(page_hash_gpu_old,page_hash_gpu_new[thread_id].vm_virt_address,page_hash_gpu_new[thread_id].hash_value,old_hash_num);
		is_virt_addr_present[thread_id]=found;


	}
}


__global__ void extend_list(page_hash *page_hash_extended,page_hash *page_hash_gpu,int *is_virt_addr_present,int *is_virt_addr_present_prefix,
		int num_old_hashes_val,int num)
{
	int id=blockIdx.x*blockDim.x+threadIdx.x;
	int rank=0;
	if(id<num)
	{	
		if(is_virt_addr_present[id]==1)
		{
			rank=num_old_hashes_val+is_virt_addr_present_prefix[id]-1;

			page_hash_extended[rank].vm_virt_address=page_hash_gpu[id].vm_virt_address;
			//page_hash_extended[rank].cuda_virt_address=page_hash_gpu[id].cuda_virt_address;
			page_hash_extended[rank].hash_value=page_hash_gpu[id].hash_value;
		}
	}

}

int cuda_extend_hash_list(page_hash **page_hash_old,page_hash *current_hints_gpu,page_hash *active_hints_gpu,unsigned int num_active_hints ,int hints_generated) 
{

	//page_hash_gpu -> current_hints_gpu
	//page_hash_gpu_old -> active_hints_gpu
	//num_old_hashes_val -> num_active_hints
	//num ->hints_generated

	int *is_virt_addr_present;
	int *is_virt_addr_present_prefix;
	//int *is_virt_addr_present_prefix_cpu;
	int *back_is_virt_addr_present_prefix;
	int NODES=0;
	int bound=0;
	int iteration=0;
	int jump=1;
	//num of unique virtual addresses in this pass compared to previous pass
	int *unique_virt_address;
	int unique_virt_address_val=0;
	int extended_list_size=0;
	int num_blocks;

	page_hash *page_hash_extended;
	page_hash *page_hash_old_ptr;
	
	//	cudaError_t result;

	unique_virt_address=(int*)malloc(sizeof(int));
//	is_virt_addr_present_prefix_cpu=(int*)malloc(num*sizeof(int));
	NODES=hints_generated;
	bound=ceil((double)log(NODES)/(double)log(2));
	
	num_blocks=ceil((double)NODES/(double)512);
	cuda_sort_virt_addr(active_hints_gpu,num_active_hints);
	//printf("List sorted by virtual addresses \n");
	cudaMalloc((void**)&is_virt_addr_present,hints_generated*sizeof(int));
	cudaMalloc((void**)&is_virt_addr_present_prefix,hints_generated*sizeof(int));
	cudaMalloc((void**)&back_is_virt_addr_present_prefix,hints_generated*sizeof(int));

	//sort by virtual addresses old hash
	//No need to sort new hashes they are already sorted by virt address
	//binary search new virt address in old virt address list
	//extend list later after find hints
	//cudaMalloc(&page_hash_old_gpu,num_old_hashes*sizeof(page_hash));				
	//cudaMemcpy(page_hash_old,page_hash_old_gpu,num_old_hashes*sizeof(page_hash),cudaMemcpyDeviceToHost);


	find_virt_addr<<<num_blocks,512>>>(active_hints_gpu,current_hints_gpu,hints_generated,is_virt_addr_present,num_active_hints);
//	printf("find virt add executed \n");

	cudaMemcpy(is_virt_addr_present_prefix,is_virt_addr_present,hints_generated*sizeof(int),cudaMemcpyDeviceToDevice);
	//printf("bound=%d\n",bound);
	
	while(iteration<bound) {
		cudaMemcpy(back_is_virt_addr_present_prefix,is_virt_addr_present_prefix,hints_generated*sizeof(int),cudaMemcpyDeviceToDevice);
		prefixSum_virt_addr_present<<<num_blocks,512>>>(is_virt_addr_present_prefix,back_is_virt_addr_present_prefix,NODES,jump);
		cudaMemcpy(is_virt_addr_present_prefix,back_is_virt_addr_present_prefix,hints_generated*sizeof(int),cudaMemcpyDeviceToDevice);
		iteration++;
		//printf("iteration = %d\n",iteration);
		jump=2*jump;
		//cudaDeviceSynchronize();


	}
	cudaFree(back_is_virt_addr_present_prefix);
	cudaMemcpy(unique_virt_address,(is_virt_addr_present_prefix+hints_generated-1),sizeof(int),cudaMemcpyDeviceToHost);
	//printf("memcpy done unique virt addr \n");
	unique_virt_address_val=*unique_virt_address;
	//unique_virt_address_val = is_virt_addr_present_prefix_cpu[num-1];
	extended_list_size=num_active_hints+ unique_virt_address_val;
	//printf("Unique Virtual Addresses = %d and Extended List size = %d\n",unique_virt_address_val,extended_list_size);

	if(unique_virt_address_val>0)
	{
		cudaMalloc((void**)&page_hash_extended,extended_list_size*sizeof(page_hash));
		cudaMemcpy(page_hash_extended,active_hints_gpu,num_active_hints*sizeof(page_hash),cudaMemcpyDeviceToDevice);
	
		extend_list<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_extended,current_hints_gpu,is_virt_addr_present,is_virt_addr_present_prefix,num_active_hints,hints_generated);

	//Now we need to extend the old hash list to include the value which was not present
	//in the previous list but are present in the current
		page_hash_old_ptr=*page_hash_old;
		cudaErrorCheck(cudaFree(page_hash_old_ptr));
		cudaMalloc((void**)&page_hash_old_ptr,extended_list_size*sizeof(page_hash));
		
		cudaMemcpy(page_hash_old_ptr,page_hash_extended,extended_list_size*sizeof(page_hash),cudaMemcpyDeviceToDevice);
		*page_hash_old=page_hash_old_ptr;

		cudaFree(page_hash_extended);
	}
//if there are unique virt addr in current hints then page_hash_old will be active hints
	else 
	{
		page_hash_old_ptr=*page_hash_old;
		cudaErrorCheck(cudaFree(page_hash_old_ptr));
		cudaMalloc((void**)&page_hash_old_ptr,num_active_hints*sizeof(page_hash));
		cudaMemcpy(page_hash_old_ptr,active_hints_gpu,num_active_hints*sizeof(page_hash),cudaMemcpyDeviceToDevice);
		*page_hash_old=page_hash_old_ptr;

	}

	cudaFree(is_virt_addr_present);
	cudaFree(is_virt_addr_present_prefix);
//	cudaFree(back_is_virt_addr_present_prefix);
    return extended_list_size;
}




__device__ int binary_search_active_other(page_hash *page_hash_gpu_old, unsigned long virtual_address,uint32_t hash_value, int old_hash_num) 
{
	/*
		   Binary search hash within the old hash list.
		   Make sure it doesn't be the same virtual address.  
	 */
	int low=0,mid=0,high = old_hash_num-1;
	while(low<=high)
	{
		mid= (high + low)/2;
		if(page_hash_gpu_old[mid].hash_value==hash_value)
		{
			return 1;
		}	
		else if(page_hash_gpu_old[mid].hash_value>hash_value)
			high=mid-1;

		else
			low = mid+1;
	}
	return 0;

}

__global__ void diff_hash_active_other(page_hash *page_hash_gpu_old,page_hash *page_hash_gpu_new,int num_threads,int *diff_map,int old_hash_num)
{

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int found =0;
	if(thread_id<num_threads) 
	{
		//diff_map[thread_id]=0;
		found = binary_search_active_other(page_hash_gpu_old,page_hash_gpu_new[thread_id].vm_virt_address,page_hash_gpu_new[thread_id].hash_value,old_hash_num);
		if(found==1)
			diff_map[thread_id]=found;


	}
}

void compare_active_other(page_hash *active_hints_gpu,page_hash *page_hash_gpu,int num_threads,int *is_a_hint,int num_active_hints)
{
	int num_blocks;
	num_blocks= ceil((double)num_threads/(double)THREADS_PER_BLOCK);	
	diff_hash_active_other<<<num_blocks,THREADS_PER_BLOCK>>>(active_hints_gpu,page_hash_gpu,num_threads,is_a_hint,num_active_hints);
}


__device__ int binary_search(page_hash *page_hash_gpu_old, unsigned long virtual_address,uint32_t hash_value, int old_hash_num) 
{
	/*
		   Binary search hash within the old hash list.
		   Make sure it doesn't be the same virtual address.  
	 */
	int low=0,mid=0,high = old_hash_num-1;
	while(low<=high)
	{
		mid= (high + low)/2;
		if(page_hash_gpu_old[mid].hash_value==hash_value)
		{
			if(page_hash_gpu_old[mid].vm_virt_address==virtual_address)
			{
				if(mid>0 && mid<(old_hash_num-1))
				{
					if(page_hash_gpu_old[mid-1].hash_value==hash_value || page_hash_gpu_old[mid+1].hash_value==hash_value)
						return 1;
					else 
						return 0;
				}
				else if (mid == 0)
				{
					if(page_hash_gpu_old[mid+1].hash_value==hash_value)
						return 1;
					else 
						return 0;
				}
				else 
				{
					if(page_hash_gpu_old[mid-1].hash_value==hash_value)
						return 1;
					else 
						return 0;

				}
			}

			else				
				return 1;

		}	
		else if(page_hash_gpu_old[mid].hash_value>hash_value)
			high=mid-1;

		else
			low = mid+1;
	}
	return 0;

}

__global__ void diff_hash(page_hash *page_hash_gpu_old,page_hash *page_hash_gpu_new,int num_threads,int *diff_map,int old_hash_num)
{

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int found =0;
	if(thread_id<num_threads) 
	{
		//diff_map[thread_id]=0;
		found = binary_search(page_hash_gpu_old,page_hash_gpu_new[thread_id].vm_virt_address,page_hash_gpu_new[thread_id].hash_value,old_hash_num);
		if(found==1)
			diff_map[thread_id]=found;


	}
}

void compare_active_self(page_hash *active_hints_gpu,page_hash *page_hash_gpu,int num_threads,int *is_a_hint,int num_active_hints)
{
	int num_blocks;
	num_blocks= ceil((double)num_threads/(double)THREADS_PER_BLOCK);	
	diff_hash<<<num_blocks,THREADS_PER_BLOCK>>>(active_hints_gpu,page_hash_gpu,num_threads,is_a_hint,num_active_hints);
}


__device__ int binary_search_active_hints(unsigned long *mapped_pages_list_gpu, unsigned long virtual_address,int num_mapped_pages) 
{
	/*
		   Binary search old hint virtual address within the new mapped pages list.
		     
	 */
	int low=0,mid=0,high = num_mapped_pages-1;
	while(low<=high)
	{
		mid= (high-low)/2 + low;
		if(mapped_pages_list_gpu[mid]==virtual_address)
		{
			return 1;

		}	
		else if(mapped_pages_list_gpu[mid]>virtual_address)
			high=mid-1;

		else
			low = mid+1;
	}
	return 0;

}

__global__ void find_active_hints_gpu(page_hash *page_hash_gpu_old,unsigned long *mapped_pages_list_gpu,int *is_hashed_page_mapped,int num,int num_mapped_pages)
{
	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	int is_mapped=1;
	if (thread_id < num) 
	{	
	//	is_mapped=check_is_page_mapped_gpu(page_hash_gpu_old[thread_id].vm_virt_address,is_page_mapped_gpu);
	
		is_mapped=binary_search_active_hints(mapped_pages_list_gpu,page_hash_gpu_old[thread_id].vm_virt_address,num_mapped_pages);
		is_hashed_page_mapped[thread_id]=is_mapped;	

	}
	

}


__global__ void get_active_hints_gpu(page_hash *page_hash_gpu_old,page_hash *active_hints_gpu,int *is_hashed_page_mapped_gpu,int *is_hashed_page_mapped_prefix_sum_gpu,int num_old) 
{
	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	int rank=0;
	if(thread_id<num_old)
	{

		rank = is_hashed_page_mapped_prefix_sum_gpu[thread_id]-1;
		if(is_hashed_page_mapped_gpu[thread_id]==1)
		{
			active_hints_gpu[rank].vm_virt_address=page_hash_gpu_old[thread_id].vm_virt_address;
			//active_hints_gpu[rank].cuda_virt_address=page_hash_gpu_old[thread_id].cuda_virt_address;
			active_hints_gpu[rank].hash_value=page_hash_gpu_old[thread_id].hash_value;

		}

	}
	
}
void get_active_hints(page_hash *page_hash_gpu_old,page_hash *active_hints_gpu,int *is_hashed_page_mapped_gpu,int *is_hashed_page_mapped_prefix_sum_gpu,int num_old_hashes_val) 
{
	int num_blocks;
	num_blocks=ceil((double)num_old_hashes_val/(double)512);

	get_active_hints_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu_old,active_hints_gpu,is_hashed_page_mapped_gpu,is_hashed_page_mapped_prefix_sum_gpu,num_old_hashes_val);

}

int find_active_hints(page_hash *page_hash_gpu_old,unsigned long *mapped_pages_list,int *is_hashed_page_mapped_gpu,int *is_hashed_page_mapped_prefix_sum_gpu,int num,int num_mapped_pages)
{
//	int *is_hashed_page_mapped;
//	int *is_hashed_page_mapped_gpu;
	unsigned int num_blocks;
	int active_hints=0;
	unsigned int bound=0;
	int iteration =0;
	int jump=1;
//	int *is_page_mapped_prefix_sum_gpu;
	int *back_is_page_mapped_prefix_sum_gpu;
	unsigned long *mapped_pages_list_gpu;

//	is_hashed_page_mapped = (int*)malloc(num*sizeof(int));

//	cudaMalloc((void**)&is_hashed_page_mapped_gpu,num*sizeof(int));
	cudaMalloc((void**)&mapped_pages_list_gpu,num_mapped_pages*sizeof(unsigned long));

	num_blocks=ceil((double)num/(double)THREADS_PER_BLOCK);

	cudaMemcpy(mapped_pages_list_gpu,mapped_pages_list,num_mapped_pages*sizeof(unsigned long),cudaMemcpyHostToDevice);
	find_active_hints_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu_old,mapped_pages_list_gpu,is_hashed_page_mapped_gpu,num,num_mapped_pages);

//	cudaMemcpy(is_hashed_page_mapped,is_hashed_page_mapped_gpu,num*sizeof(int),cudaMemcpyDeviceToHost);


/*	int i=0;
	printf("is hashed page mapped cpu \n");
	for(i=0;i<num;i++)
		printf("is hashed mapped %d is   %d \n",i+1,is_hashed_page_mapped[i]);*/
//	cudaMalloc((void**)&is_page_mapped_prefix_sum_gpu,num*sizeof(int));
	cudaMalloc((void**)&back_is_page_mapped_prefix_sum_gpu,num*sizeof(int));


	cudaMemcpy(is_hashed_page_mapped_prefix_sum_gpu,is_hashed_page_mapped_gpu,num*sizeof(int),cudaMemcpyDeviceToDevice);

	bound=ceil((double)log(num)/(double)log(2));

//comuting prefix sum in GPU
	
	while(iteration<bound) {

	cudaMemcpy(back_is_page_mapped_prefix_sum_gpu,is_hashed_page_mapped_prefix_sum_gpu,num*sizeof(int),cudaMemcpyDeviceToDevice);
		prefixSum_virt_addr_present<<<num_blocks,512>>>(is_hashed_page_mapped_prefix_sum_gpu,back_is_page_mapped_prefix_sum_gpu,num,jump);
		cudaMemcpy(is_hashed_page_mapped_prefix_sum_gpu,back_is_page_mapped_prefix_sum_gpu,num*sizeof(int),cudaMemcpyDeviceToDevice);
		iteration++;
		jump=2*jump;


	}
	cudaMemcpy(&active_hints,is_hashed_page_mapped_prefix_sum_gpu+num-1,sizeof(int),cudaMemcpyDeviceToHost);


	

	cudaFree(mapped_pages_list_gpu);
//	cudaFree(is_page_mapped_prefix_sum_gpu);
	cudaFree(back_is_page_mapped_prefix_sum_gpu);
//	cudaFree(is_hashed_page_mapped_gpu);
//	free(is_hashed_page_mapped);

	return active_hints;
}


int generate_hints_old(page_hash *page_hash_gpu,page_hash **hints_cpu,int *diff_map_gpu,int num) 
{
	page_hash *hints_gpu;	
	int *is_a_hint;
	int *is_a_hint_prefix_sum;
	int *back_is_a_hint_prefix_sum;
	unsigned int num_blocks;
	int total_hints=0;
	int bound=0;
	int iteration=0;
	int jump=1;
	

	cudaMalloc((void**)&is_a_hint,num*sizeof(int));
	cudaMalloc((void**)&is_a_hint_prefix_sum,num*sizeof(int));
	cudaMalloc((void**)&back_is_a_hint_prefix_sum,num*sizeof(int));
	
	num_blocks=ceil((double)num/(double)THREADS_PER_BLOCK);

	generate_hints_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu,is_a_hint,num);	

	
//Considering Diff map while generating hints

	if(diff_map_gpu!=NULL)
	{
		//This function ORs the is_a_hint with diff map to get total hints
		hints_using_diff_map<<<num_blocks,THREADS_PER_BLOCK>>>(is_a_hint,diff_map_gpu,num);
	}


	cudaMemcpy(is_a_hint_prefix_sum,is_a_hint,num*sizeof(int),cudaMemcpyDeviceToDevice);

	bound=ceil((double)log(num)/(double)log(2));

//comuting prefix sum in GPU
	
	while(iteration<bound) {
		cudaMemcpy(back_is_a_hint_prefix_sum,is_a_hint_prefix_sum,num*sizeof(int),cudaMemcpyDeviceToDevice);
		prefixSum_virt_addr_present<<<num_blocks,512>>>(is_a_hint_prefix_sum,back_is_a_hint_prefix_sum,num,jump);
		cudaMemcpy(is_a_hint_prefix_sum,back_is_a_hint_prefix_sum,num*sizeof(int),cudaMemcpyDeviceToDevice);
		iteration++;
		jump=2*jump;


	}

	cudaFree(back_is_a_hint_prefix_sum);

	cudaMemcpy(&total_hints,is_a_hint_prefix_sum+num-1,sizeof(int),cudaMemcpyDeviceToHost);

	//printf("Total Hints Generated on GPU %d \n",total_hints);

	if(total_hints>0) 
	{	
		cudaMalloc((void**)&hints_gpu,total_hints*sizeof(page_hash));

	//copy hints from page_hash_gpu data structure to another array
	
		copy_hints<<<num_blocks,THREADS_PER_BLOCK>>>(page_hash_gpu,hints_gpu,is_a_hint,is_a_hint_prefix_sum,num);
	
		//XXX cuda_sort_virt_addr added on 26ht oct to send sorted hints to KSM
		cuda_sort_virt_addr(hints_gpu,total_hints); //KSM needs hints sorted by virt addr.
		*hints_cpu = (page_hash*)malloc(total_hints*sizeof(page_hash));
		cudaErrorCheck(cudaMemcpy(*hints_cpu,hints_gpu,total_hints*sizeof(page_hash),cudaMemcpyDeviceToHost));
		cudaFree(hints_gpu);
	}


	cudaFree(is_a_hint);
	cudaFree(is_a_hint_prefix_sum);
	
	return total_hints;

}	

__global__ void initialize_gpu(int *array,int total_num)
{

	int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	if(thread_id<total_num) 
	{
		array[thread_id]=0;
	}
}

void initialize(int *array,int total_num)
{
	unsigned int num_blocks;
	num_blocks=ceil((double)total_num/(double)THREADS_PER_BLOCK);
	initialize_gpu<<<num_blocks,THREADS_PER_BLOCK>>>(array,total_num);
	
}
