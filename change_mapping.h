#include<stdio.h>
#include "gpu_func.h"
#ifndef CHANGE_MAPPING_H
#define CHANGE_MAPPING_H

unsigned int change_mapping(unsigned int ,unsigned long ,unsigned int, page_hash*,unsigned long**,unsigned int*);
int send_hints_to_module(unsigned long*,unsigned int,long);
void garbage_collection(void);
int clflush(void);
int are_hints_exhausted(void);
void clear_cuda_mappings(void);
void clear_pid_to_mm(void);
void start_scanning_hints(void);
unsigned int hints_shared(void);
#endif
