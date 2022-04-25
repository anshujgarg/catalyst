# catalyst
Catalyst: Gpu-assisted rapid memory deduplication in virtualization environments

Refer to the research paper for technical details: https://doi.org/10.1145/3050748.3050760


## Details
This implementation works for kernel version 3.14.65.
The kernel should be compiled after updating the corresponding kernel file
in the "kernel_files" folder. The following steps will work on the newly compiled
kernel.

## Makefile
"Makefile" compiles the kernel module (Catalyst_module) and user space daemon (Catalyst_daemon).
There is additional module "print_ksm_info_module.c" for getting debug information.

## Starting the setup
First run "echo 2 >/sys/kernel/mm/ksm/run"
Before running catalyst_daemon make sure catalyst_module in inserted (insmod catalyst_module).
To run the catalyst_daemon just type (./catalyst_daemnon).

## monitoring the pages shared

Use the command "cat /sys/kernel/mm/ksm/pages_shared" to list 
the number of pages which are shared

