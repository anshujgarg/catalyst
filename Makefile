obj-m += catalyst_kernel_module.o
obj-m += print_ksm_info_module.o
TIME=
HINT_DIST=
#TIME = -DTIME and HINT_DIST=-DHINT_DIST
#ccflags-y += -DDBG
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules


target_cuda:change_mapping.o gpu_func.o target_program_cuda.cu ioctl_def.h change_mapping.h
	nvcc $(TIME) $(HINT_DIST) change_mapping.o gpu_func.o catalyst_daemon.cu -o catalyst_daemon

change_mapping.o:change_mapping.c change_mapping.h ioctl_def.h
	gcc -c $(TIME) change_mapping.c -o change_mapping.o
gpu_func.o:gpu_func.cu gpu_func.h
	nvcc -c  $(TIME) gpu_func.cu -o gpu_func.o

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
