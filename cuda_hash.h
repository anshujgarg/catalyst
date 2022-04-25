#include<stdint.h>
#ifndef CUDA_HASH_H
#define CUDA_HASH_H
typedef struct {
	
	uint32_t hash_value;
	unsigned long vm_virt_address;
	unsigned long cuda_virt_address;
}page_hash;


unsigned int cuda_hash(char *page_data,page_hash *page_hash_cpu,unsigned long **hints,int page_size,int total_threads,
			page_hash **page_hash_old,int *are_old_hashes_available,unsigned int *num_old_hash,unsigned long *mapped_pages_list,unsigned long);
void compute_hash(char *page_data,page_hash *page_hash_cpu,unsigned long **hints,int page_size,int total_threads,
		page_hash **page_hash_old,int *are_old_hashes_available,unsigned int *num_old_hash,unsigned long *mapped_pages_list,unsigned long);

#endif
