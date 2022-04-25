#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<fcntl.h>
#include<unistd.h>
#include<time.h>
#include<cuda.h>
#include<sys/mman.h>
#include "gpu_func.h"

#define PAGE_SIZE 4096
#define THREADS_PER_BLOCK 512
#define PAGE_SHIFT 12

extern "C" {
#include "change_mapping.h"
}

#ifdef DBG
#define dprint printf
#else
#define dprint(x,...)   
#endif

#ifdef TIME
#define tprint printf
#else
#define tprint(x,...)
#endif



#define sprint printf

#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
		+(uint32_t)(((const uint8_t *)(d))[0]) )


#define MAX_VM 16


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
unsigned long long get_ksm_total_cycles()
{
	FILE *fp;
	char path[1035];
	char *ptr;
	char *token;
	char search[2] = " ";
	char command[]="cat /sys/kernel/mm/ksm/total_cycles";
	unsigned long long ksm_total_cycles=0;
	
			
	fp = popen(command,"r");
	
		if (fp == NULL) {
			printf("Failed to run command\n" );
			exit(1);
		}

		while (fgets(path, sizeof(path)-1, fp) != NULL) {
	
			token = strtok(path,search);
			ksm_total_cycles = strtoull(token,&ptr,10);
	//		printf("KSM total cycles = %llu \n",ksm_total_cycles);

		}
		/* close */
	pclose(fp);
	return ksm_total_cycles;

}

unsigned long long get_ksm_wasted_cycles()
{
	FILE *fp;
	char path[1035];
	char *ptr;
	char *token;
	char search[2] = " ";
	char command[]="cat /sys/kernel/mm/ksm/wasted_cycles";
	unsigned long long ksm_wasted_cycles=0;
	
			
	fp = popen(command,"r");
	
		if (fp == NULL) {
			printf("Failed to run command\n" );
			exit(1);
		}

		while (fgets(path, sizeof(path)-1, fp) != NULL) {
	
			token = strtok(path,search);
			ksm_wasted_cycles = strtoull(token,&ptr,10);
		//	printf("KSM wasted cycles = %llu \n",ksm_wasted_cycles);

		}
		/* close */
	pclose(fp);
	return ksm_wasted_cycles;

}
struct vm_hash_data 
{

	int hash_value;
	page_hash *page_hash_gpu;
	page_hash *old_hints_gpu;
	page_hash *active_hints_gpu;
	unsigned int num_old_hints;
	int num_active_hints;
	int *is_a_hint_gpu;
	int are_old_hashes_available;
	long process_id;
	unsigned num_pages;
	struct vm_hash_data *next_vm;
	struct vm_hash_data *prev_vm;

};

typedef struct 
{
	long process_id;
	char vm_name[100];
	unsigned int num_pages;
	int has_hash_data;
	struct vm_hash_data *hash_data;
}vm_info;




static struct vm_hash_data *head_vm_hash_data=NULL,*tail_vm_hash_data=NULL;

void print_vm_info(vm_info *vm_info,int num_vms)
{
	int i=0;
	for(i=0;i<num_vms;i++)
	{
		
		printf("\nVM Number= %d\n",i+1);
		printf("VM Name = %s",vm_info[i].vm_name);
		printf("VM Process id =%ld\n",vm_info[i].process_id);
		printf("VM Total Pages =%u\n",vm_info[i].num_pages);
		printf("VM has hash data =%d\n",vm_info[i].has_hash_data);
	}	
}
int get_vm_info(vm_info *vm_info)
{
	FILE *fp;
	char path[1035];
	char *ptr;
	char *token;
	char search[2] = " ";
	const char * command = "/bin/ps awx | grep qemu | grep -v grep | sed 's/\\s*\\([0-9]*\\).*-name \\(\\w*\\) .*/\\1 \\2/'";
	int num_vm=0;
	long vm_process_id;
			
	fp = popen(command,"r");
	
	if (fp == NULL) 
	{
		printf("Failed to run command\n" );
		exit(1);
	}
	/* Read the output a line at a time - output it. */
	while (fgets(path, sizeof(path)-1, fp) != NULL) {
		token = strtok(path,search);
		vm_process_id=strtol(token,&ptr,10);
		vm_info[num_vm].process_id=vm_process_id;
		vm_info[num_vm].has_hash_data=0;
		token = strtok(NULL,search);
		strcpy(vm_info[num_vm].vm_name,token);
		num_vm++;
		if(num_vm>MAX_VM)
			printf("Error: Too many VMs. Supports only 16 VMs");
	}
	return num_vm;
}

void get_vm_pages(vm_info *vm_info,int num_vms) 
{
	FILE *fp;
	char path[1035];
	char *ptr;
	char *token;
	char search[2] = " ";
	char command1[]="virsh dominfo ";
	char command2[]="|grep \"Max memory\"";
	char *joined;
	int i=0;
	
	long physical_memory=0;
	int num_pages=0;
			
	for(i=0;i<num_vms;i++)
	{
		joined = (char*)malloc(strlen(command1)+strlen(vm_info[i].vm_name)+strlen(command2)+1);
		strcpy(joined,command1);
		strncat(joined,vm_info[i].vm_name,strlen(vm_info[i].vm_name)-1);
		strcat(joined,command2);
		fp = popen(joined,"r");
		if (fp == NULL) {
			printf("Failed to run command\n" );
			exit(1);
		}

		while (fgets(path, sizeof(path)-1, fp) != NULL) {
			token = strtok(path,search);
			token = strtok(NULL,search);
			token = strtok(NULL,search);
			physical_memory=strtol(token,&ptr,10);
			num_pages=physical_memory/4;
			vm_info[i].num_pages=num_pages;
			/*
			if(i==1)
				vm_info[i].num_pages=10;
			else
				vm_info[i].num_pages=12;
			 */
		}
	}
	/* close */
	pclose(fp);
}


struct vm_hash_data* create_vm_hash_data(vm_info vm_info)
{
	struct vm_hash_data *temp_vm_hash_data;
	struct vm_hash_data *new_node=NULL;
	temp_vm_hash_data=head_vm_hash_data;
	if(temp_vm_hash_data==NULL)
	{

		new_node = (struct vm_hash_data *)malloc(sizeof(struct vm_hash_data));
		new_node->next_vm=NULL;
		new_node->prev_vm=NULL;
		new_node->hash_value=1;
		new_node->num_pages=vm_info.num_pages;
		new_node->process_id=vm_info.process_id;
		new_node->num_old_hints=0;
		new_node->are_old_hashes_available=0;
		new_node->is_a_hint_gpu=NULL;
		head_vm_hash_data=new_node;
		tail_vm_hash_data=new_node;	
		//printf("head_process_id = %ld\n",head_vm_hash_data->process_id);	

	}
	else
	{
		new_node = (struct vm_hash_data *)malloc(sizeof(struct vm_hash_data));
		new_node->next_vm=NULL;
		new_node->prev_vm=tail_vm_hash_data;
		new_node->hash_value=1;
		new_node->num_pages=vm_info.num_pages;
		new_node->process_id=vm_info.process_id;
		new_node->num_old_hints=0;
		new_node->are_old_hashes_available=0;
		new_node->is_a_hint_gpu=NULL;
		tail_vm_hash_data->next_vm=new_node;
		tail_vm_hash_data=new_node;		

	}
	return new_node;

	
}


void remove_stale_vm_hash_data(vm_info *vm_info,int num_virt_machines) 
{

	int i;
	struct vm_hash_data *index_vm_hash_data,*temp_vm_hash_data;
	index_vm_hash_data=head_vm_hash_data;

	if(index_vm_hash_data==NULL)
	{
	//	printf("NO hash Data for any VM \n");
		return;
	}
	
	
	while(index_vm_hash_data!=NULL)
	{
		
		for(i=0;i<num_virt_machines;i++)
		{
			if(index_vm_hash_data->process_id==vm_info[i].process_id)
			{
				vm_info[i].has_hash_data=1;
				index_vm_hash_data=index_vm_hash_data->next_vm;
				break;
			}		

		}
		if(i==num_virt_machines)
		{
			
			if(index_vm_hash_data==head_vm_hash_data && index_vm_hash_data==tail_vm_hash_data )
			{
				temp_vm_hash_data=index_vm_hash_data;
				head_vm_hash_data=NULL;
				tail_vm_hash_data=NULL;
				free(temp_vm_hash_data);
				index_vm_hash_data=NULL;
				return;
			}
				
			else if(index_vm_hash_data==head_vm_hash_data  && index_vm_hash_data!=tail_vm_hash_data)
			{
				temp_vm_hash_data=index_vm_hash_data;
				head_vm_hash_data=index_vm_hash_data->next_vm;
				head_vm_hash_data->prev_vm=NULL;
				index_vm_hash_data=head_vm_hash_data;
				free(temp_vm_hash_data);
		
			}	
			else if(index_vm_hash_data==tail_vm_hash_data)
			{
				tail_vm_hash_data=index_vm_hash_data->prev_vm;
				tail_vm_hash_data->next_vm=NULL;
				temp_vm_hash_data=index_vm_hash_data;
				index_vm_hash_data=NULL;
				free(temp_vm_hash_data);
			}
			else if(index_vm_hash_data!=head_vm_hash_data  && index_vm_hash_data!=tail_vm_hash_data)
			{
				temp_vm_hash_data=index_vm_hash_data;
				(index_vm_hash_data->next_vm)->prev_vm=(index_vm_hash_data->prev_vm);
				(index_vm_hash_data->prev_vm)->next_vm=(index_vm_hash_data->next_vm);
				index_vm_hash_data=index_vm_hash_data->next_vm;
				free(temp_vm_hash_data);
			}	
		}
	}
	

}

void print_vm_hash_data()
{
	struct vm_hash_data *temp_vm_hash_data;
	temp_vm_hash_data=head_vm_hash_data;
	int count=1;
	if(temp_vm_hash_data==NULL)
		printf("Head Null \n");
	while(temp_vm_hash_data!=NULL)
	{
		
		printf("VM number %d \n",count);
		printf("VM process ID %ld \n",temp_vm_hash_data->process_id);
		printf("VM Total Pages %u \n",temp_vm_hash_data->num_pages);
		printf("VM hash value %d \n",temp_vm_hash_data->hash_value);
		count = count+1;
		temp_vm_hash_data=temp_vm_hash_data->next_vm;
	}	
}

/*
void hex_dump(unsigned long address)
{
	//  int i=0;
	unsigned int *x = (unsigned int *)address;
	dprint("Value 0 = %x ",x[0] );  
	dprint(", Value 128 = %x ",x[128] );  
	dprint(",  Value 256 = %x ",x[256] );  
	dprint(",  Value 500 = %x",x[500] );  
	dprint(",  Value 900 = %x",x[900] );  

	// for(i=0; i < 1024; ++i)
	//     dprint(" %x ",x[i]);


}
*/

void create_all_vm_hash_data(vm_info *virt_machines_info,int num_virt_machines)
{
	int i;
	for(i=0;i<num_virt_machines;i++)
	{
		if(virt_machines_info[i].has_hash_data==0)
		{
			//printf("Inside create_vm_hash_data\n");
			virt_machines_info[i].hash_data=create_vm_hash_data(virt_machines_info[i]);
		}
	}
}

static unsigned int cumulative_hints_per_round=0;
int main(int argc, char *argv[]) {


	vm_info *virt_machines_info;
	int num_virt_machines=0;
	char *page_data;	
	page_hash *page_hash_cpu;
	//page_hash *page_hash_old;
//	unsigned int num_old_hash=0; //number of elements in old hash list- needed for extending lists
	void *ptr; //void ptr
	unsigned long p;
	int i,j;
	int num=525000;
	unsigned int size;
	size = 4096*num;
	unsigned int target_process_id;
	unsigned long target_address;
//	double begin,end,change_mapping_time,cuda_hash_time;
	unsigned long *hints;
	unsigned long *mapped_pages_list=NULL;
	unsigned int num_mapped_pages=0;
	unsigned int total_hints=0;
	//int are_old_hashes_available=0;
	unsigned int mappings_changed=0;
	unsigned int hints_shared_per_round=0;

#ifdef ADAPT
	float new_sleep_time;
	float old_sleep_time=5;
	unsigned int num_old_hints=0;
	unsigned int num_new_hints;

#endif

	//Variables to measure the CPU cycles.
       unsigned long long cycle_begin=0,cycle_end=0;
       unsigned long long cycles_change_mapping_round=0,cycles_restore_mapping_round=0;
       //unsigned long long cumulative_cycles_change_mapping=0,cumulative_cycles_restore_mapping=0;
       unsigned long long trans_hints_ksm=0;
       unsigned long long trans_hints_gpu=0;
       unsigned long long trans_data_gpu=0;
       unsigned long long cycles_total_ksm=0,cycles_wasted_ksm=0;
       

	
       int loop = 0;

	ptr = malloc(size + 8192);
	bzero(ptr, size+8192);

	p = (unsigned long)ptr;
	dprint("Before Shifting Adrress %lx\n", p);
	if(((p>>PAGE_SHIFT)<<PAGE_SHIFT)!=p) { 
		p = ((p >> PAGE_SHIFT) + 1) << PAGE_SHIFT;
	}
	page_data = (char*)p;
	dprint("After Shifting Address %lx\n",p );

	target_address = (unsigned long)page_data;
	target_process_id=getpid();


//	int num_vm_hash_data=0;

	//XXX Start of each round you need to look for running VMs again

	while(1)
	{
		virt_machines_info = (vm_info*)malloc(MAX_VM*sizeof(vm_info));
		num_virt_machines = get_vm_info(virt_machines_info);
		if(num_virt_machines>0)
		{
			get_vm_pages(virt_machines_info,num_virt_machines);
			//print_vm_info(virt_machines_info,num_virt_machines);
		}
		/*
		else
			printf("No virtual Machines Running \n");*/

		if(num_virt_machines>0)
		{
			
		//	printf("\n");
		        printf("Round %d\n",loop);
			cycles_total_ksm=get_ksm_total_cycles();
			cycles_wasted_ksm=get_ksm_wasted_cycles();
			printf("Cycles_Total_KSM_before_Round %llu\n",cycles_total_ksm);	
			printf("Cycles_Wasted_KSM_before_Round %llu\n",cycles_wasted_ksm);
			remove_stale_vm_hash_data(virt_machines_info,num_virt_machines);
			create_all_vm_hash_data(virt_machines_info,num_virt_machines);
		
			//printf("\n");
			//print_vm_hash_data();
		
			//int threads =10;
			struct vm_hash_data *current_vm_hash_data,*compared_to_vm;

			for (i=0;i<num_virt_machines;i++)
			{
				current_vm_hash_data = virt_machines_info[i].hash_data;
			//	printf("current VM = %ld \n",current_vm_hash_data->process_id);
			
				page_hash *page_hash_cpu,*page_hash_gpu;
				page_hash_cpu=(page_hash*)malloc(virt_machines_info[i].num_pages*sizeof(page_hash));

				cycle_begin=rdtsc();
				
				mappings_changed=change_mapping(virt_machines_info[i].process_id,target_address,target_process_id,page_hash_cpu,&mapped_pages_list,&num_mapped_pages);

				cycle_end=rdtsc();
				cycles_change_mapping_round+=(cycle_end-cycle_begin);

				virt_machines_info[i].num_pages=mappings_changed;
	//			printf("Mappings Changed  VM %d = %d \n",i+1,mappings_changed);
											
				cudaErrorCheck(cudaMalloc((void**)&(current_vm_hash_data->page_hash_gpu),virt_machines_info[i].num_pages*sizeof(page_hash)));
				cudaErrorCheck(cudaMemcpy(current_vm_hash_data->page_hash_gpu,page_hash_cpu,virt_machines_info[i].num_pages*sizeof(page_hash),cudaMemcpyHostToDevice));
								
				compute_hash(page_data,current_vm_hash_data->page_hash_gpu,PAGE_SIZE,virt_machines_info[i].num_pages,&trans_data_gpu);
			
				cuda_sort(current_vm_hash_data->page_hash_gpu,virt_machines_info[i].num_pages);
				
				free(page_hash_cpu);
	
				cycle_begin=rdtsc();

				clear_cuda_mappings(); //restore mappings

				cycle_end=rdtsc();
				cycles_restore_mapping_round+=(cycle_end-cycle_begin);
				
				if(num_mapped_pages==0)
					current_vm_hash_data->num_active_hints=0;					
				
				if(current_vm_hash_data->are_old_hashes_available==1 && num_mapped_pages>0 && mappings_changed>0)
				{
				//	printf("Generate Active Hints\n");
					unsigned int num_old_hashes_val= current_vm_hash_data->num_old_hints;
					int num_active_hints;
					int *is_hashed_page_mapped_prefix_sum_gpu;
					int *is_hashed_page_mapped_gpu;
					
					//page_hash *active_hints_cpu;

					cudaErrorCheck(cudaMalloc((void**)&is_hashed_page_mapped_gpu,num_old_hashes_val*sizeof(int)));
					cudaErrorCheck(cudaMalloc((void**)&is_hashed_page_mapped_prefix_sum_gpu,num_old_hashes_val*sizeof(int)));

					//cudaErrorCheck(cudaMalloc(&page_hash_gpu_old,num_old_hashes_val*sizeof(page_hash)));	
					//cudaErrorCheck(cudaMemcpy(page_hash_gpu_old,*page_hash_old,num_old_hashes_val*sizeof(page_hash),cudaMemcpyHostToDevice));

					num_active_hints=find_active_hints(current_vm_hash_data->old_hints_gpu,mapped_pages_list,is_hashed_page_mapped_gpu,is_hashed_page_mapped_prefix_sum_gpu,num_old_hashes_val,num_mapped_pages);		
					current_vm_hash_data->num_active_hints=num_active_hints;
					if(current_vm_hash_data->num_active_hints>0) 
					{
						cudaMalloc((void**)&(current_vm_hash_data->active_hints_gpu),num_active_hints*sizeof(page_hash));
						get_active_hints(current_vm_hash_data->old_hints_gpu,current_vm_hash_data->active_hints_gpu,is_hashed_page_mapped_gpu,is_hashed_page_mapped_prefix_sum_gpu,num_old_hashes_val);
						
						//Test Code
						//active_hints_cpu=(page_hash*)malloc(num_active_hints*sizeof(page_hash));
						//cudaErrorCheck(cudaMemcpy(active_hints_cpu,current_vm_hash_data->active_hints_gpu,num_active_hints*sizeof(page_hash),cudaMemcpyDeviceToHost));
						
						cuda_sort(current_vm_hash_data->active_hints_gpu,num_active_hints);
						/*
						for(k=0;k<num_active_hints;k+=400)
							printf("Active Hint %d = %lu \n",k+1,active_hints_cpu[k].vm_virt_address);*/
						

					}
				//	printf("Total Active Hints are %d\n",num_active_hints);

					cudaFree(is_hashed_page_mapped_prefix_sum_gpu);
					cudaFree(is_hashed_page_mapped_gpu);
				
				}
				if(num_mapped_pages>0)
					free(mapped_pages_list);
				mapped_pages_list=NULL;
				num_mapped_pages=0;
				garbage_collection();

				
//XXX mappings_changed is used as total_threads
			
			}
			
			unsigned long *hints;
			//XXX Comparing Hashes
			for(i=0;i<num_virt_machines;i++)
			{
				int total_hints=0;

				current_vm_hash_data = virt_machines_info[i].hash_data;

				//int *is_a_hint_cpu;
				//is_a_hint_cpu=(int*)malloc(virt_machines_info[i].num_pages*sizeof(int));
				if(current_vm_hash_data->is_a_hint_gpu==NULL)
				{
					cudaErrorCheck(cudaMalloc((void**)&(current_vm_hash_data->is_a_hint_gpu),virt_machines_info[i].num_pages*sizeof(int)));
					initialize(current_vm_hash_data->is_a_hint_gpu,virt_machines_info[i].num_pages);
				}
				//generate_self_hints(VM i)
				//page_hash *current_hints;
				
				for(j=i+1;j<num_virt_machines;j++)
				{	
					
					compared_to_vm = virt_machines_info[j].hash_data;
					if(compared_to_vm->is_a_hint_gpu==NULL)
					{
						cudaErrorCheck(cudaMalloc((void**)&(compared_to_vm->is_a_hint_gpu),virt_machines_info[j].num_pages*sizeof(int)));
						initialize(compared_to_vm->is_a_hint_gpu,virt_machines_info[j].num_pages);
					}
					compare_hash(current_vm_hash_data->page_hash_gpu,compared_to_vm->page_hash_gpu,current_vm_hash_data->is_a_hint_gpu,
							compared_to_vm->is_a_hint_gpu,virt_machines_info[i].num_pages,virt_machines_info[j].num_pages);
					if(compared_to_vm->num_active_hints>0)
						compare_active_other(compared_to_vm->active_hints_gpu,current_vm_hash_data->page_hash_gpu,virt_machines_info[i].num_pages,current_vm_hash_data->is_a_hint_gpu,compared_to_vm->num_active_hints);
					
				}
				//cuda_sort(current_vm_hash_data->page_hash_gpu,threads);
				if((current_vm_hash_data->num_active_hints) >0)
				{
					//cuda_sort(current_vm_hash_data->active_hints_gpu,current_vm_hash_data->num_active_hints);
					compare_active_self(current_vm_hash_data->active_hints_gpu,current_vm_hash_data->page_hash_gpu,virt_machines_info[i].num_pages,current_vm_hash_data->is_a_hint_gpu,current_vm_hash_data->num_active_hints);
				}
				generate_hints(current_vm_hash_data->page_hash_gpu,current_vm_hash_data->is_a_hint_gpu,virt_machines_info[i].num_pages);
				//cudaErrorCheck(cudaMemcpy(is_a_hint_cpu,current_vm_hash_data->is_a_hint_gpu,virt_machines_info[i].num_pages*sizeof(int),cudaMemcpyDeviceToHost));
				/*
				for(k=0;k<virt_machines_info[i].num_pages;k++)
				{
					printf("is a hint %d = %d\n",k+1,is_a_hint_cpu[k]);
				}*/		
		
				//NOTE that KSM needs hints in sorted order of Virt Address
				page_hash *current_hints_gpu;
			
			//	cycle_begin=rdtsc();
				total_hints=copy_hints(current_vm_hash_data->page_hash_gpu,&hints,&current_hints_gpu,current_vm_hash_data->is_a_hint_gpu,virt_machines_info[i].num_pages,&trans_hints_gpu);
			
			//	cycle_end=rdtsc();
			//	trans_hints_gpu+=(cycle_end-cycle_begin);
			
				cumulative_hints_per_round+=total_hints;
				
			//	printf("Total Hints for VM %d = %d \n ", i+1,total_hints);
				if(current_vm_hash_data->are_old_hashes_available==0)
				{
					if(total_hints>0)
					{
						cudaErrorCheck(cudaMalloc((void**)&(current_vm_hash_data->old_hints_gpu),total_hints*sizeof(page_hash)));
						cudaErrorCheck(cudaMemcpy(current_vm_hash_data->old_hints_gpu,current_hints_gpu,total_hints*sizeof(page_hash),cudaMemcpyDeviceToDevice));
						current_vm_hash_data->are_old_hashes_available=1;
						current_vm_hash_data->num_old_hints=total_hints;
						cudaFree(current_hints_gpu);
					}
					
				}
				
				//Extend the old page hashes stores as history
				
				else if(current_vm_hash_data->are_old_hashes_available==1 && mappings_changed>0) 
				{

					int extended_list_size=0;
					unsigned int num_old_hashes_val= current_vm_hash_data->num_old_hints;
					int num_active_hints;
					num_active_hints=current_vm_hash_data->num_active_hints;

					if(num_active_hints>0 && total_hints>0)
						extended_list_size=cuda_extend_hash_list(&(current_vm_hash_data->old_hints_gpu),current_hints_gpu,current_vm_hash_data->active_hints_gpu,num_active_hints,total_hints);

					else if(num_active_hints !=0 && total_hints==0)
					{
						cudaErrorCheck(cudaFree(current_vm_hash_data->old_hints_gpu));
						//current_vm_hash_data->old_hints_gpu=(page_hash*)malloc(num_active_hints*sizeof(page_hash));
						cudaErrorCheck(cudaMalloc((void**)&(current_vm_hash_data->old_hints_gpu),num_active_hints*sizeof(page_hash)));

						cudaErrorCheck(cudaMemcpy(current_vm_hash_data->old_hints_gpu,current_vm_hash_data->active_hints_gpu,num_active_hints*sizeof(page_hash),cudaMemcpyDeviceToDevice));
						extended_list_size=num_active_hints;

					}
					else if(num_active_hints==0 && total_hints!=0)
					{
						cudaErrorCheck(cudaFree(current_vm_hash_data->old_hints_gpu));
						cudaErrorCheck(cudaMalloc((void**)&(current_vm_hash_data->old_hints_gpu),total_hints*sizeof(page_hash)));

						//current_vm_hash_data->=(page_hash*)malloc(total_hints*sizeof(page_hash));
						cudaErrorCheck(cudaMemcpy(current_vm_hash_data->old_hints_gpu,current_hints_gpu,total_hints*sizeof(page_hash),cudaMemcpyDeviceToDevice));
						extended_list_size=total_hints;
					}

				//	printf("Num old hashes value = %d and Extended List size = %d Active Hints = %d\n",current_vm_hash_data->num_old_hints,extended_list_size,num_active_hints);	
					current_vm_hash_data->num_old_hints=extended_list_size; //new hashes needs to be calculated

					if(extended_list_size>0)
						current_vm_hash_data->are_old_hashes_available=1;
					else
						current_vm_hash_data->are_old_hashes_available=0;

					if(num_active_hints>0)
					{
						cudaErrorCheck(cudaFree(current_vm_hash_data->active_hints_gpu));
						current_vm_hash_data->num_active_hints=0;
					}
					if(total_hints>0)
						cudaErrorCheck(cudaFree(current_hints_gpu));

				}
				
				/*
				for(k=0;k<total_hints;k+=400)
				{
					printf("hint %d = %ld \n",k+1,hints[k] );
				}*/
				cudaErrorCheck(cudaFree(current_vm_hash_data->is_a_hint_gpu));
				//free(is_a_hint_cpu);
				current_vm_hash_data->is_a_hint_gpu=NULL;
				
				if(total_hints>0)
				{
					cycle_begin=rdtsc();
					send_hints_to_module(hints,total_hints,virt_machines_info[i].process_id);
					cycle_end=rdtsc();
					trans_hints_ksm+=(cycle_end-cycle_begin);
					free(hints);
				}
				if(mappings_changed>0)
					cudaErrorCheck(cudaFree(current_vm_hash_data->page_hash_gpu));

					
			}
			//Hints generated and wriiten to KSM; Send signal to start scanning
			start_scanning_hints(); //signals KSM to start hint scanning
			int are_hints_available;
			are_hints_available=are_hints_exhausted();
			free(virt_machines_info);
			while(1)
			{
				if(are_hints_available==1)
				{
					sleep(30);
				}
				else
				{
					hints_shared_per_round=hints_shared();
			
					clear_pid_to_mm();
					//printf("Round %d\n",loop);
					printf("Cumulative_hints %u\n",cumulative_hints_per_round);
					printf("Hints_Shared %u\n",hints_shared_per_round);
					printf("Cycle_Change_Mapping %llu\n",cycles_change_mapping_round);
					printf("Cycle_Restore_Mapping %llu\n",cycles_restore_mapping_round);
					printf("Cycle_Trans_Data_GPU %llu\n",trans_data_gpu);
					printf("Cycle_Trans_Hints_GPU %llu\n",trans_hints_gpu);
					printf("Cycles_Trans_Hints_KSM %llu\n",trans_hints_ksm);
					
					cycles_change_mapping_round=0;
					cycles_restore_mapping_round=0;
					trans_data_gpu=0;
					trans_hints_gpu=0;
					trans_hints_ksm=0;
					cumulative_hints_per_round=0;
					break;
				}
				are_hints_available=are_hints_exhausted();
			//break;
			}

		loop++;
		}

	}

	free(ptr);
	return 0;
}
