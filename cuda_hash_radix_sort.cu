#include<stdio.h>
#include<stdint.h>      //This header file is for uint32_t and uint16_t ;
#include<cuda.h>
#define PAGE_SIZE 128
#define NUM_PAGES 27
#define THREADS_PER_BLOCK 512
#define NUM_VM 2

/*


   NOTE 1: You will need page number with the page data either virtual or physical page number.
  	 look for the input of madvise and check what it stores about VMs. 
   NOTE 2: While sending the page data you can fit the vm id and virtual page number of page along with
	  the data sent. <vm id, page_number1, page data1,page_number2, page data2> All this in the input. Or you can send the page number in a seperate array.  
   NOTE 3: You need to repeatedly send the page data to the GPU as the page might change or the content might change. 

   1.The input to the program can be the vm id and the pages that needed to be hashed. For testing purpose you can do 
     two mallocs say of size 100 ( m1 and m2). To the gpu program you can pass the malloc id (m1 and m2) then virtual addresses &m1 to &m1+99 and 
     &m2 to &m2+99. Then you can transfer data of these mallocs to GPU and compute hashes. After that the output format of the computation should be 
     like a set of matches pages [ (m1,51)->(m1,20)->(m2,10) ; (m1,12)->(m2,19)->(m2,34)] like this.
   2.The output can be the sharing opportunities among the VMs.The regions that KSM should scan
     in order to get shared pages.
   3.Experiments:
   	a. One experiment can be start a mix of VMs 2 windows and 2 Linux, look for the sharing opportunities.
	b. The other can be keep increasing number of identical VMs from 1 to 8 and check how fast hasing can 
	   be done and number of CPU cycle saveds to achieve the same amount of hashing. 
	c. One can be sharing achived in the same time or increase in sharing with time. 
	d. One might be included which give the run time of GPU based hashing of the pages. 
	e. One simple experiment can be done to check the correctness where two identical VMs run to check the
	   sharing opportunities. 

*/


//Part of Superfast Hashcode Used as it is.
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )


//The above macro convert 8 bit character d[1] to 16 bit unit and sums it with d[0] to get a 
//16 bit number 


//This function implements "superfasthash" algorithm. In your GPU code all the threads should be running
//hash function on different pages.

/*

	The code below can be found in the azillionmonkey website: saved under /gpu_page_hashing as Superfast_Hash_Function.html . They have directly used the code of superfast hash without modifying it so just ttreat it as a blackbox


*/ 

typedef struct {
	//unsigned long int physical_id;
	char *physical_id;
	//unsigned long int virtual_address[NUM_PAGES];
	char *virtual_address[NUM_PAGES];
	uint32_t hashes[NUM_PAGES];

}page_hash;

typedef struct {

	char *physical_id;
	//unsigned long int virtual_address[NUM_PAGES];
	char *virtual_address[NUM_PAGES];
	char data[NUM_PAGES][PAGE_SIZE];
}page_data;

/*typedef struct {
	char *physical_id_1, *physical_id_2;
	unsigned long int virtual_address_1, virtual_address_2;
}matched_tuple;*/

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


//Note matched1 and matched 2 will contain the correspoding matched pages of VMs.
/*
   Here two implementation logics can be used.
	1. Give all thread one page or a set of pages. And compare those pages.
		If you are giving a set of page to all thread you can overlap data transfer and
		page comparison. 
        2. Give each thread a page and compare that page with all other page. 	

   */

/*
__global__ void compare_hashes(uint32_t *vm1_hash, uint32_t *vm2_hash,int *matched1,int *matched2,int size_vm1,int size_vm2) {

	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	int i=0;
	unsigned int index=0;
	if(thread_id<size_vm2) {
		for(i=0;i<size_vm1;i++) {
			if(vm1_hash[i]==vm2_hash[thread_id]) {
				matched1[i]=thread_id; //This should be the virtual page number
				matched2[thread_id]=i; //This should be the virtual page number
			}
		}

	}	

}

*/
__global__ void compare_hashes(page_hash *vm1_page_hash, page_hash *vm2_page_hash, matched *shared_pages,int size_vm1,int size_vm2) {

	int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
	int i=0;
	unsigned int index=0;
	//int Max=NUM_PAGES+1;
	shared_pages->matched_count[thread_id]=0;
	if(thread_id<size_vm2) {
		for(i=0;i<size_vm1;i++) {
			shared_pages->matched_address1[thread_id][index]=NULL;
			shared_pages->matched_address2[thread_id][index]=NULL;
			if(vm1_page_hash->hashes[i]==vm2_page_hash->hashes[thread_id]) {
				//matched1[i]=thread_id; //This should be the virtual page number
				//matched2[thread_id]=i; //This should be the virtual page number
				shared_pages->matched_address1[thread_id][index]=vm1_page_hash->virtual_address[i];
				shared_pages->matched_address2[thread_id][index]=vm2_page_hash->virtual_address[thread_id];
				index++;
				shared_pages->matched_count[thread_id]+=1;
			}

		//shared_pages->matched_address1[thread_id][index+1]=NULL;
		//shared_pages->matched_address2[thread_id][index]=NULL;
		}
	


	}	

}
__global__  void gpu_fast_page_hash(page_data *data, page_hash *page_hashes,int page_size,int total_threads) {
   
        int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t len = PAGE_SIZE;
     	uint32_t hash ,tmp;
	char *temp_ptr;
	hash=len;
	int rem;

	int thread_page_index=thread_id*PAGE_SIZE;
	page_hashes->physical_id=data->physical_id;
	//temp_ptr=data+thread_page_index;
	temp_ptr=data->data[thread_id];
	
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
		page_hashes->hashes[thread_id]=hash;
		page_hashes->virtual_address[thread_id]=data->virtual_address[thread_id];
	        //return hash; Instead of returning store this value in some array
	}

}


/*

   NOTE 1: For each request for scan of VM data areas you need to have a seperate data structure.
   NOTE 2: After hashing the pages for one vm you just need to keep the hashes in memory not the page data. 
   	   Hence you can free the page data from GPUs.
   NOTE 3: Matching of hashes should be done among all the VM hashes in order to get the total sharing matrix of VMs.
   NOTE 4. Need to maintain a map of sharing among VMs.
 
   Initially you can write the test code to compute the hash among the two VMs. 
   

*/

int main(int argc, char *argv[]) {
	char *data;
	uint32_t *hash,*hash_2;
	page_hash *page_hashes_vm1_dev,*page_hashes_vm1_cpu;
	page_hash *page_hashes_vm2_dev,*page_hashes_vm2_cpu;
	page_data *data_vm1_dev,*data_vm1_cpu,*data_vm2_dev,*data_vm2_cpu;
	matched *matched_data_cpu,*matched_data_dev;


	int i=0,j=0;
	int *dev_matched1,*dev_matched2;
	int *matched1,*matched2;

	char *dev_data,*dev_data_2;  //should be equal to number of VM. Look whether can be dynamically created
	uint32_t *dev_hash,*dev_hash_2; //Should be equal to number of VMs
	
	unsigned int num_blocks,total_threads;
	num_blocks=(NUM_PAGES%THREADS_PER_BLOCK)+1;
	total_threads=NUM_PAGES;
        //The dev_data variable should be VM data and the count of the data variable should be equal to number of VMs.	
	cudaMalloc((void**)&dev_data,PAGE_SIZE*NUM_PAGES*sizeof(char));
	cudaMalloc((void**)&dev_hash,NUM_PAGES*sizeof(uint32_t));
	
	data=(char*)malloc(PAGE_SIZE*NUM_PAGES*sizeof(char));
	hash=(uint32_t*)malloc(NUM_PAGES*sizeof(uint32_t));
	
	cudaMalloc((void**)&dev_data_2,PAGE_SIZE*NUM_PAGES*sizeof(char));
	cudaMalloc((void**)&dev_hash_2,NUM_PAGES*sizeof(uint32_t));

	cudaMalloc((void**)&dev_matched1,NUM_PAGES*sizeof(int)); 
	cudaMalloc((void**)&dev_matched2,NUM_PAGES*sizeof(int)); 

	matched1=(int*)malloc(NUM_PAGES*sizeof(int));
	matched2=(int*)malloc(NUM_PAGES*sizeof(int));

	cudaMalloc((void**)&page_hashes_vm1_dev,sizeof(page_hash));
	page_hashes_vm1_cpu=(page_hash*)malloc(sizeof(page_hash));
	
	cudaMalloc((void**)&data_vm1_dev,sizeof(page_data));
	data_vm1_cpu=(page_data*)malloc(sizeof(page_data));
	
	cudaMalloc((void**)&data_vm2_dev,sizeof(page_data));
	data_vm2_cpu=(page_data*)malloc(sizeof(page_data));
	
	cudaMalloc((void**)&page_hashes_vm2_dev,sizeof(page_hash));
	page_hashes_vm2_cpu=(page_hash*)malloc(sizeof(page_hash));
	
	cudaMalloc((void**)&matched_data_dev,sizeof(matched));
	matched_data_cpu=(matched*)malloc(sizeof(matched));

	/* This can also be stored as a bitmap using cchar array to dave memory and changing the bit of char array using bit operations	

	  */

	hash_2=(uint32_t*)malloc(NUM_PAGES*sizeof(uint32_t));
	//The code below is to generate random page data

	char a;
	/*
	for(i=0;i<NUM_PAGES;i++) {
		a='A';
       		for(j=0;j<PAGE_SIZE;j++) {
			memcpy(&data[(i*PAGE_SIZE+j)],&a,1);
			a=a+1;
			if(a>'Z')
				a='A';
		}
	}

	*/
	data_vm1_cpu->physical_id=&data_vm1_cpu->data[0][0];
	data_vm2_cpu->physical_id=&data_vm2_cpu->data[0][0];
	for(i=0;i<NUM_PAGES;i++) {
		//a='A'+(i%26);
		//a='A';
		
		//data_vm1_cpu->virtual_address[i]=i;
		//data_vm2_cpu->virtual_address[i]=i;
		data_vm1_cpu->virtual_address[i]=data_vm1_cpu->data[i];
		data_vm2_cpu->virtual_address[i]=data_vm2_cpu->data[i];
       		for(j=0;j<PAGE_SIZE;j++) {
			//memcpy(&data[(i*PAGE_SIZE+j)],&a,1);
			data_vm1_cpu->data[i][j]=a;
			data_vm2_cpu->data[i][j]=a;
			a=a+1;
			if(a>'Z')
				a='A';
		}
	}
	/*
	for(i=0;i<NUM_PAGES;i++) {
		strcpy(&data[i*PAGE_SIZE],"ABCDEFG");
		memcpy(&data[(i*PAGE_SIZE) +7],&a,1);
		a=a+1;
	}*/

	//cudaMemcpy(dev_data,data,sizeof(char)*NUM_PAGES*PAGE_SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(data_vm1_dev,data_vm1_cpu,sizeof(page_data),cudaMemcpyHostToDevice);
	cudaMemcpy(data_vm2_dev,data_vm2_cpu,sizeof(page_data),cudaMemcpyHostToDevice);/*for(i=0;i<NUM_PAGES;i++) {
		printf("Text=%.*s\n",PAGE_SIZE,&data[i*PAGE_SIZE]);
	}*/
	printf("\nCUDA KERNEL EXECUTES \n");
	
	gpu_fast_page_hash<<<num_blocks,THREADS_PER_BLOCK>>>(data_vm1_dev,page_hashes_vm1_dev,PAGE_SIZE,total_threads);
//	cudaFree(dev_data);
	cudaFree(data_vm1_dev);
	//cudaDeviceSynchronize();

	cudaMemcpy(page_hashes_vm1_cpu,page_hashes_vm1_dev,sizeof(page_hash),cudaMemcpyDeviceToHost);
	gpu_fast_page_hash<<<num_blocks,THREADS_PER_BLOCK>>>(data_vm2_dev,page_hashes_vm2_dev,PAGE_SIZE,total_threads);
	cudaFree(data_vm2_dev);
	cudaMemcpy(page_hashes_vm2_cpu,page_hashes_vm2_dev,sizeof(page_hash),cudaMemcpyDeviceToHost);
	/*
	for(i=0;i<NUM_PAGES;i++) {
		printf("id=%d, physical_id=%p,virtual_address=%lu,hash=%u \n",i,page_hashes_vm1_cpu->physical_id,page_hashes_vm1_cpu->virtual_address[i],page_hashes_vm1_cpu->hashes[i]);
	}
	printf("\nData for VM2\n");
	for(i=0;i<NUM_PAGES;i++) {
		printf("id=%d, physical_id=%p,virtual_address=%lu,hash=%u \n",i,page_hashes_vm2_cpu->physical_id,page_hashes_vm2_cpu->virtual_address[i],page_hashes_vm2_cpu->hashes[i]);
	} */
	compare_hashes<<<num_blocks,THREADS_PER_BLOCK>>>(page_hashes_vm1_dev,page_hashes_vm2_dev,matched_data_dev,NUM_PAGES,NUM_PAGES);
	cudaMemcpy(matched_data_cpu,matched_data_dev,sizeof(matched),cudaMemcpyDeviceToHost);
	cudaFree(matched_data_dev);


	
	for(i=0;i<NUM_PAGES;i++)
	{
       		for(j=0;j<matched_data_cpu->matched_count[i];j++)	
				{
				if(matched_data_cpu->matched_address1[i][j]!=NULL)
				 printf("Num=%d,Matched_Count=%d,matched_address 1=%p , matched_address 2=%p\n",i,matched_data_cpu->matched_count[i],matched_data_cpu->matched_address2[i][j],matched_data_cpu->matched_address1[i][j]);			
				}
	}
	
/*

*********   Very Important   **********

NOTE: Be careful with resuing the data structure. They might be storing the old values which might create problems or show up while resuing. 



*/

	//self comparisoni
	matched *matched_data_cpu1;
	matched_data_cpu1=(matched*)malloc(sizeof(matched));
	cudaMalloc((void**)&matched_data_dev,sizeof(matched));
	compare_hashes<<<num_blocks,THREADS_PER_BLOCK>>>(page_hashes_vm1_dev,page_hashes_vm1_dev,matched_data_dev,NUM_PAGES,NUM_PAGES);
	cudaMemcpy(matched_data_cpu1,matched_data_dev,sizeof(matched),cudaMemcpyDeviceToHost);
	for(i=0;i<NUM_PAGES;i++)
	{
       		for(j=0;j<matched_data_cpu1->matched_count[i];j++)	
				{
				if(matched_data_cpu1->matched_address1[i][j]!=NULL)
				 printf("Num=%d,Matched_Count=%d,matched_address 1=%p , matched_address 2=%p\n",i,matched_data_cpu1->matched_count[i],matched_data_cpu1->matched_address2[i][j],matched_data_cpu1->matched_address1[i][j]);			
				}
	}


/*	
	char bitmap[NUM_PAGES];
	for(i=0;i<NUM_PAGES;i++)
		bitmap[i]='1';
	for(i=0;i<NUM_PAGES;i++)
		for(j=0;j<NUM_PAGES;j++)
		{
			if(i==j)
				continue;
			if(bitmap[i]=='1' ) {
				if(matched_data_cpu1->matched_address1[i][j])
					bitmap[j]='0';									
			}			
		}
	for(i=0;i<NUM_PAGES;i++)
		printf("Bitmap %d =%c \t",i,bitmap[i]);

	printf("\n");*/

	 //int records=0,read;
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
	for(i=0;i<NUM_PAGES;i++) {
		printf("Num =%d, Page Hash=  %u, Page Address = %p \n",i,page_hashes_vm1_cpu->hashes[i],page_hashes_vm1_cpu->virtual_address[i]);
	
	}

	printf("\n");
	/*
	int flag=0;
	for(i=0;i<NUM_PAGES-1;i++) {
		if((page_hashes_vm1_cpu->hashes[i]==page_hashes_vm1_cpu->hashes[i+1]) && flag==0) {
			printf("%p,"page_hashes_vm1_cpu->virtual_address[i]);
		}
		else if((page_hashes_vm1_cpu->hashes[i]==page_hashes_vm1_cpu->hashes[i+1]) && flag==0) {
			printf("%p,"page_hashes_vm1_cpu->virtual_address[i]);
		}

	}*/
	
	/*
	cudaDeviceSynchronize();
	
	cudaMemcpy(dev_data_2,data,sizeof(char)*NUM_PAGES*PAGE_SIZE,cudaMemcpyHostToDevice);
	gpu_fast_page_hash<<<num_blocks,THREADS_PER_BLOCK>>>(dev_data_2,dev_hash_2,PAGE_SIZE,total_threads);	
	cudaFree(dev_data_2);
	cudaMemcpy(hash_2,dev_hash_2,sizeof(uint32_t)*NUM_PAGES,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	 compare_hashes<<<num_blocks,THREADS_PER_BLOCK>>>(dev_hash,dev_hash_2,dev_matched1,dev_matched2,NUM_PAGES,NUM_PAGES);

	cudaMemcpy(matched1,dev_matched1,sizeof(int)*NUM_PAGES,cudaMemcpyDeviceToHost);
	cudaMemcpy(matched2,dev_matched2,sizeof(int)*NUM_PAGES,cudaMemcpyDeviceToHost);
	*
	

		
	for(i=0;i<NUM_PAGES;i++) {
		//printf("Text=%.*s\n",PAGE_SIZE,&hash[i*PAGE_SIZE]);
		printf("Hashes %d= %u \n",i, hash[i]);
	}

	for(i=0;i<NUM_PAGES;i++) {
		//printf("Text=%.*s\n",PAGE_SIZE,&hash[i*PAGE_SIZE]);
		printf("Hashes %d= %u \n",i, hash_2[i]);
	}
	for(i=0;i<NUM_PAGES;i++) {
	//printf("Text=%.*s\n",PAGE_SIZE,&hash[i*PAGE_SIZE]);
		printf("Matches %d= %d \n",matched1[i], matched2[i]);
	} */
	return 0;
}
