#include<stdio.h>
#include<stdlib.h>
#include<errno.h>
#include<fcntl.h>
#include<string.h>
#include<unistd.h>
#include<sys/uio.h>
#include<time.h>
#include "change_mapping.h"
#include<sys/ioctl.h>
#include "ioctl_def.h"
#define BUFFER_LENGTH 256

#ifdef TIME
	#define tprint printf
#else
	#define tprint(x,...)
#endif

struct proc_info {
	unsigned int source_process_id;
	unsigned int target_process_id;
	unsigned long target_address;
};


static char receive[BUFFER_LENGTH];
static int fd;
struct proc_info *process_info;
unsigned long *vm_address;
int clflush()
{
   char buff[5];
   write(fd, buff, 4);
   return 0;
}

unsigned int hints_shared(void)
{
	unsigned int hints_shared_per_round=0;
	ioctl(fd,HINTS_SHARED,&hints_shared_per_round);
	return hints_shared_per_round;
}


/*
   are_hints_exhausted return 0 if hints have been completly scanned by KSM
   */


int are_hints_exhausted(void) {
	int hints_available;
	ioctl(fd,ARE_HINTS_EXHAUSTED,&hints_available);
	return hints_available;
}

/*
   This function sends the info (hints) about page that have equal hashes
   */

void clear_cuda_mappings(void)
{
	ioctl(fd,CLEAR_MAPPINGS,1);
	return;
}

void clear_pid_to_mm()
{
	ioctl(fd,CLEAR_PID_TO_MM,1);
	close(fd);
	return;
}

int send_hints_to_module(unsigned long *hints,unsigned int total_hints,long process_id) {
	double begin,end,clear_mapping_time;
	srandom(clock());
	begin = clock();
	ioctl(fd,WRITE_HINTS_PID,&process_id);
	ioctl(fd,WRITE_HINTS_NUM,total_hints); 
	end = clock();
	clear_mapping_time = (end - begin)/(CLOCKS_PER_SEC);
	tprint("\nTime to clear Mappins is %f\n",clear_mapping_time);
	ioctl(fd,WRITE_HINTS,hints);
	return 0;	
}

void start_scanning_hints()
{
	ioctl(fd,START_SCAN,1);
	return;
}

unsigned int change_mapping(unsigned int source_process_id,unsigned long target_address,unsigned int target_process_id,page_hash *page_hash_cpu,unsigned long **mapped_pages_list,unsigned int *num_mapped_pages) {

	int ret;
	int i;
	char stringToSend[BUFFER_LENGTH];
	unsigned long *add_value;
	unsigned int num_mappings;
	unsigned int num_mapped_pages_temp;
//	printf("The user test program executes \n");
	fd = open("/dev/ebbchar",O_RDWR);
	if (fd < 0) {
		perror("Failed to Open the device...\n");
		return errno;
	}
	process_info = (struct proc_info*)malloc(sizeof(struct proc_info));
	process_info->source_process_id=source_process_id;
	process_info->target_process_id=target_process_id;
	process_info->target_address=target_address;
	ioctl(fd,WRITE_PROC_INFO,process_info);
	ioctl(fd,READ_NUM_MAPPINGS,&num_mappings);
	
	ioctl(fd,READ_NUM_PAGES_MAPPED,&num_mapped_pages_temp);
	*num_mapped_pages=num_mapped_pages_temp;
//	printf("num mapped pages = %u \n",num_mapped_pages_temp);
	if(num_mapped_pages_temp>0)
	{
		*mapped_pages_list=(unsigned long*)malloc(num_mapped_pages_temp*sizeof(unsigned long));
//	add_value=(unsigned long*)malloc(num_mapped_pages_temp*sizeof(unsigned long));
		ioctl(fd,READ_PAGES_MAPPED,*mapped_pages_list); 
	}

	vm_address = (unsigned long*)malloc(num_mappings*sizeof(unsigned long));
	ioctl(fd,READ_MAPPINGS,vm_address);
	for(i=0;i<num_mappings;i++) {
	//	printf("vm address %d is %lx\n",i, vm_address[i]) ;
		page_hash_cpu[i].vm_virt_address=vm_address[i];

	}
	
	return num_mappings ;

}

void garbage_collection(void) {
	ioctl(fd,GARBAGE_COLLECT,0);
	
}	
