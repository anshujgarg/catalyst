#include<stdint.h>
#ifndef GPU_FUNC_H
#define GPU_FUNC_H
typedef struct 
{	
	uint32_t hash_value;
	unsigned long vm_virt_address;
}page_hash;

void compute_hash(char *page_data,page_hash *page_hash_gpu,int page_size,unsigned int total_threads,unsigned long long *trans_data_gpu);
void generate_hints(page_hash *page_hash_gpu,int *is_a_hint,int total_threads); 
void cuda_sort(page_hash *page_hash_gpu,int total_threads);
void cuda_sort_virt_addr(page_hash *page_hash_gpu,int total_threads);
//int prefix_sum(int *is_a_hint,int num);
void compare_hash(page_hash* page_hash_gpu_vm1,page_hash *page_hash_gpu_vm2,int *is_a_hint_vm1,int *is_a_hint_vm2,int num_pages_vm1,int num_pages_vm2);
void initialize(int *array,int total_num);
int copy_hints(page_hash *page_hash_gpu,unsigned long **hints,page_hash **current_hints_gpu,int *is_a_hint_gpu,int total_pages,unsigned long long *trans_hints_gpu);
int find_active_hints(page_hash *page_hash_gpu_old,unsigned long *mapped_pages_list,int *is_hashed_page_mapped_gpu,int *is_hashed_page_mapped_prefix_sum_gpu,int num,int num_mapped_pages);
void get_active_hints(page_hash *page_hash_gpu_old,page_hash *active_hints_gpu,int *is_hashed_page_mapped_gpu,int *is_hashed_page_mapped_prefix_sum_gpu,int num_old);
void compare_active_self(page_hash *active_hints_gpu,page_hash *page_hash_gpu,int num_threads,int *is_a_hint,int num_active_hints);
int cuda_extend_hash_list(page_hash **page_hash_old,page_hash *current_hints_gpu,page_hash *active_hints_gpu,unsigned int num_active_hints ,int total_hints); 
void compare_active_other(page_hash *active_hints_gpu,page_hash *page_hash_gpu,int num_threads,int *is_a_hint,int num_active_hints);


#endif
