#include <linux/init.h>           // Macros used to mark up functions e.g. __init __exit
#include <linux/module.h>         // Core header for loading LKMs into the kernel
#include <linux/device.h>         // Header to support the kernel Driver Model
#include <linux/kernel.h>         // Contains types, macros, functions for the kernel
#include <linux/fs.h>             // Header for the Linux file system support
#include <asm/uaccess.h>          // Required for the copy to user function
#include<linux/mm.h>
#include<linux/sched.h>
#include<linux/slab.h>
#include<linux/moduleparam.h>
#include<linux/types.h>
#include<linux/highmem.h>
#include<linux/errno.h>
#include<linux/rmap.h>
#include<asm/tlbflush.h>
#include<linux/pagemap.h>
#include<linux/ioctl.h>
#include<linux/uio.h>
#include<linux/delay.h>
#include<linux/ksm.h>
#include "ioctl_def.h"


#define  DEVICE_NAME "ebbchar"    ///< The device will appear at /dev/ebbchar using this value
#define  CLASS_NAME  "ebb"        ///< The device class -- this is a character device driver
#define PAGE_SHIFT 12

MODULE_LICENSE("GPL");            ///< The license type -- this affects available functionality
MODULE_AUTHOR("Anshuj Garg");    ///< The author -- visible when you use modinfo
MODULE_DESCRIPTION("A simple Linux char driver for the BBB");  ///< The description -- see modinfo
MODULE_VERSION("0.1");            ///< A version number to inform users

static int    majorNumber;                  ///< Stores the device number -- determined automatically
static char   message[256] = {0};           ///< Memory for the string that is passed from userspace
static short  size_of_message;              ///< Used to remember the size of the string stored
static int    numberOpens = 0;              ///< Counts the number of times the device is opened
static struct class*  ebbcharClass  = NULL; ///< The device-driver class struct pointer
static struct device* ebbcharDevice = NULL; ///< The device-driver device struct pointer

// The prototype functions for the character driver -- must come before the struct definition
static int     dev_open(struct inode *, struct file *);
static int     dev_release(struct inode *, struct file *);
static ssize_t dev_read(struct file *, char *, size_t, loff_t *);
static ssize_t dev_write(struct file *, const char *, size_t, loff_t *);
static long dev_ioctl(struct file *filep,unsigned int cmd,unsigned long arg);


int process_id=0;
unsigned int source_process_id=0;
unsigned int target_process_id=0;
struct mm_struct *source_mm=NULL;
struct mm_struct *target_mm=NULL;
struct mm_struct *process_mm=NULL;
struct task_struct *process_tasks=NULL;
int write_counter=0;
unsigned long address=0;
unsigned long target_address=0;
struct vm_area_struct *process_vma=NULL;
struct vm_area_struct *source_vma=NULL;
struct vm_area_struct *target_vma=NULL;
struct page *process_page;
unsigned int mappings_changed=0;
unsigned int total_pages_mapped=0;
unsigned int num_mapped_pages=0;
unsigned long *mapped_pages_list=NULL;

long current_pid;
struct mm_struct *current_mm;
unsigned int current_total_hints;
//char *is_page_mapped;

//KSM Exported variables

extern unsigned int total_hints;
extern int hints_available;
extern int (*get_gpu_hints)(unsigned long*);
extern struct ksm_scan_hints *scan_hints;

struct ksm_scan_hints *current_scan_hints_struct;

//struct ksm_scan_hints *ksm_hints_head;
//ksm_hints_head=scan_hints;
struct ksm_scan_hints *ksm_hints_tail;
extern unsigned int hints_shared_per_round;
//ksm_hints_tail=scan_hints;

/** @brief Devices are represented as file structure in the kernel. The file_operations structure from

 *  /linux/fs.h lists the callback functions that you wish to associated with your file operations
 *  using a C99 syntax structure. char devices usually implement open, read, write and release calls
 */

/*
	The following structure keeps the mapping of cuda variable's virtual address to vm pages physical address

 */
struct page_mappings {

	unsigned long cuda_virt_address; //virtual address of cuda process variable
	unsigned long vm_virt_address; //physical address of VM page mapped to above virtual address;
	struct page_mappings *next_mapping;
};

struct page_mappings *page_mapping_head=NULL;
struct page_mappings *page_mapping_tail=NULL;

struct pid_to_mm
{
	long process_id;
	struct mm_struct *mm;
	struct pid_to_mm *next_pid;
};

struct pid_to_mm *head_pid_to_mm=NULL;
struct pid_to_mm *tail_pid_to_mm=NULL;


//used to store the vm addresses that are actually mapped to physical address
//needed to remove the stale hints from the hint history list

void garbage_collection(void) {

	struct page_mappings *page_mapping_index,*temp;
	process_id=0;
	source_process_id=0;
	target_process_id=0;
	source_mm=NULL;
	target_mm=NULL;
	process_mm=NULL;
	process_tasks=NULL;
	address=0;
	target_address=0;
	process_vma=NULL;
	source_vma=NULL;
	target_vma=NULL;
	process_page=NULL;
	mappings_changed=0;
	total_pages_mapped=0;
	num_mapped_pages=0;
	//	total_hints=0;
	page_mapping_index=page_mapping_head;
	page_mapping_head=NULL;
	page_mapping_tail=NULL;

	while(page_mapping_index!=NULL) {
		temp=page_mapping_index->next_mapping;
		vfree(page_mapping_index);
		page_mapping_index=temp;

	}
	//	vfree(is_page_mapped);
	vfree(mapped_pages_list);



}


void update_pid_to_mm(struct mm_struct *source_mm, long vm_process_id)
{

	struct pid_to_mm *node;
	node = (struct pid_to_mm*)vmalloc(sizeof(struct pid_to_mm));
	node->process_id=vm_process_id;
	node->mm = source_mm;
	node->next_pid=NULL;
	if (head_pid_to_mm==NULL)
	{
		head_pid_to_mm = node;
		tail_pid_to_mm = node;
	}
	else
	{
		tail_pid_to_mm->next_pid=node;
		tail_pid_to_mm=node;
	}


}

struct mm_struct* pid_to_mm_lookup(long process_id)
{
	struct pid_to_mm *index;
	index = head_pid_to_mm;
	while(index!=NULL)
	{
		if(index->process_id == process_id)
			return index->mm;
		index=index->next_pid;
	}
	return NULL;
}

void free_pid_to_mm(void)
{
	struct pid_to_mm *index,*temp;
	index = head_pid_to_mm;
	head_pid_to_mm=NULL;
	tail_pid_to_mm=NULL;
	while(index!=NULL)
	{
		temp=index->next_pid;
	//	printk(KERN_INFO "Freeing process id= %ld \n",index->process_id);
		vfree(index);
		index=temp;

	}


}

int lookup_ksm_hints(struct mm_struct *mm)
{
	struct ksm_scan_hints *index;
	//index = ksm_hints_head;
	index=scan_hints;
	while(index!=NULL)
	{	
		if(index->mm==mm)
			return 1;
		//printk(KERN_INFO "KSM hint mm %p \n",index->mm);
		index=index->next_mm_to_scan;
		
	}
	return 0;
	
}

struct ksm_scan_hints* get_scan_hints_struct(struct mm_struct *mm)
{
	struct ksm_scan_hints *index;
	//index = ksm_hints_head;
	index=scan_hints;
	while(index!=NULL)
	{	
		if(index->mm==mm)
			return index;
		//printk(KERN_INFO "KSM hint mm %p \n",index->mm);
		index=index->next_mm_to_scan;

	}
	return NULL;
}

void traverse_ksm_hints(void)
{
	struct ksm_scan_hints *index;
	//index = ksm_hints_head;
	index=scan_hints;
	while(index!=NULL)
	{
		 if(atomic_read(&((index->mm)->mm_users)) == 0)
		 {
			 index=index->next_mm_to_scan;
			 continue;

		 }
			 
		// printk(KERN_INFO "KSM hint mm %p, num_hints=%u ,hint [0]= %lu,hint[400] = %lu\n",index->mm,index->num_hints,index->address[0],index->address[400]);
		 index=index->next_mm_to_scan;
		
	}
}

void addto_page_mapping_list(unsigned long scan_address, unsigned long change_address) {

	struct page_mappings *page_mapping;

	page_mapping = (struct page_mappings*)vmalloc(sizeof(struct page_mappings));
	page_mapping->cuda_virt_address=change_address;
	page_mapping->vm_virt_address=scan_address;
	page_mapping->next_mapping=NULL;

	if(page_mapping_head ==NULL) {
		page_mapping_head=page_mapping;
		page_mapping_tail=page_mapping;
	}

	else {
		page_mapping_tail->next_mapping=page_mapping;
		page_mapping_tail=page_mapping;
	}
	mappings_changed++; //This variable is used for accounting purpose when transferring mapping data to user space  (iovec)

}

void print_mappings(void) {
	struct page_mappings *temp;
	int count=0;
	temp=page_mapping_head;
	while(temp) {
		printk(KERN_INFO "Mapping number = %d, VM Address %lx , CUDA Address %lx \n",count,temp->vm_virt_address,temp->cuda_virt_address);
		temp=temp->next_mapping;
		count++;
	}
}


static struct file_operations fops =
{
		.open = dev_open,
		.read = dev_read,
		.write = dev_write,
		.unlocked_ioctl = dev_ioctl,
		.release = dev_release,

};

static inline pgoff_t linear_pindex(struct vm_area_struct *vma,
		unsigned long address)
{
	pgoff_t pgoff;
	pgoff = (address - vma->vm_start) >> PAGE_SHIFT;
	pgoff += vma->vm_pgoff;
	return pgoff >> (PAGE_CACHE_SHIFT - PAGE_SHIFT);
}




unsigned long is_set(unsigned long flag) {
	if(flag!=0)
		return 1;
	else
		return 0;
}


static inline struct page* follow_huge_PUD(struct mm_struct *mm, unsigned long address,
		pud_t *pud)
{
	struct page *page;
	page = pte_page(*(pte_t *)pud);
	if (page)
		page += ((address & ~PUD_MASK) >> PAGE_SHIFT);
	return page;
}
static inline struct page *follow_huge_PMD(struct mm_struct *mm, unsigned long address,
		pmd_t *pmd)
{
	struct page *page;
	page = pte_page(*(pte_t *)pmd);
	if (page)
		page += ((address & ~PMD_MASK) >> PAGE_SHIFT);
	return page;
}
static inline struct page *follow_trans_huge_PMD(struct mm_struct *mm, unsigned long address,
		pmd_t *pmd)
{
	struct page *page = NULL;

	assert_spin_locked(&mm->page_table_lock);
	page = pmd_page(*pmd);
	VM_BUG_ON(!PageHead(page));
	page += (address & ~HPAGE_PMD_MASK) >> PAGE_SHIFT;
	VM_BUG_ON(!PageCompound(page));
	return page;

}

static struct page* get_page_from_addr(struct vm_area_struct *vma,unsigned long address)
{
	unsigned long pfn;
	pgd_t *pgd;
	pud_t *pud;
	pmd_t *pmd;
	pte_t *ptep, pte;
	spinlock_t *ptl;
	struct page *page=NULL;
	struct mm_struct *mm = vma->vm_mm;
	pgd = pgd_offset(mm, address);
	if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
		goto out;
	pud = pud_offset(pgd, address);
	if (pud_none(*pud))
		goto out;
	if ((pud_val(*pud) & _PAGE_PSE) && vma->vm_flags & VM_HUGETLB) {
		page = follow_huge_PUD(mm, address, pud);
		goto out;
	}
	if (unlikely(pud_bad(*pud)))
		goto out;

	pmd = pmd_offset(pud, address);
	if (pmd_none(*pmd))
		goto out;
	if ((pmd_val(*pmd) & _PAGE_PSE) && vma->vm_flags & VM_HUGETLB) {
		page = follow_huge_PMD(mm, address, pmd);
		goto out;
	}
	if (pmd_trans_huge(*pmd)){
		spin_lock(&mm->page_table_lock);
		if (likely(pmd_trans_huge(*pmd))) {
			page = follow_trans_huge_PMD(mm, address,
					pmd);
			spin_unlock(&mm->page_table_lock);
			goto out;
		} else
			spin_unlock(&mm->page_table_lock);

	}
	if (unlikely(pmd_bad(*pmd)))
		goto out;

	ptep = pte_offset_map_lock(mm, pmd, address, &ptl);

	pte = *ptep;
	if (!pte_present(pte))
		goto no_page;
	pfn = pte_pfn(pte);
	if(!pfn_valid(pfn))
		goto bad_page;
	page = pfn_to_page(pfn);
	if (unlikely(!page)) {
		if (page_to_pfn(ZERO_PAGE(0)) == pte_pfn(pte))
			goto bad_page;
		page = pte_page(pte);
	}
	pte_unmap_unlock(ptep, ptl);
	out:
	return page;
	bad_page:
	pte_unmap_unlock(ptep, ptl);
	return NULL;
	no_page:
	pte_unmap_unlock(ptep, ptl);
	return page;


}

/*
   This function unmaps the virtual machine pages mapped to cuda virtual address space.
   You also need to restore the flags of VM mages which were made PG_mlocked while changing
   mappings.
 */
 static int clear_cuda_mappings(unsigned long change_address,unsigned long scan_address ) {
	   //unsigned long pfn;
	   pgd_t *pgd;
	   pud_t *pud;
	   pmd_t *pmd;
	   pte_t *ptep = NULL, pte;
	   spinlock_t *ptl;
	   //   struct page *scan_page=NULL;
	   struct page *change_page=NULL;
	   struct anon_vma *anon_vma,*temp_anon;
	   struct mm_struct *mm ;
	   struct vm_area_struct *scan_vma, *change_vma;
	   // *cuda_pmd = 0;
	   //struct mm_struct *mm = process_mm;
	   //  scan_vma=find_vma(source_mm,scan_address);
	   change_vma=find_vma(target_mm,change_address);

	   //  scan_page = get_page_from_addr(scan_vma,scan_address);
	   change_page = get_page_from_addr(change_vma,change_address);
	   mm=change_vma->vm_mm;
	   pgd = pgd_offset(mm, change_address);
	   if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
		   goto out;
	   pud = pud_offset(pgd, change_address);
	   if (pud_none(*pud))
		   goto out;
	   if ((pud_val(*pud) & _PAGE_PSE) && change_vma->vm_flags & VM_HUGETLB) {
		   goto out;
		   /*
                page = follow_huge_PUD(mm, address, pud);
                goto got_page;*/
	   }
	   if (unlikely(pud_bad(*pud)))
		   goto out;

	   pmd = pmd_offset(pud, change_address);
	   if (pmd_none(*pmd))
		   goto out;
	   if ((pmd_val(*pmd) & _PAGE_PSE) && change_vma->vm_flags & VM_HUGETLB) {
		   /*
                page = follow_huge_PMD(mm, address, pmd);
                goto got_page;*/
		   goto out;
	   }
	   if (pmd_trans_huge(*pmd)){
		   goto out;

		   /*    spin_lock(&mm->page_table_lock);
                if (likely(pmd_trans_huge(*pmd))) {
                                page = follow_trans_huge_PMD(mm, address,
                                                             pmd);
                                spin_unlock(&mm->page_table_lock);
                                goto got_page;
                } else
                        spin_unlock(&mm->page_table_lock);*/

	   }
	   if (unlikely(pmd_bad(*pmd)))
		   goto out;

	   ptep = pte_offset_map_lock(mm, pmd, change_address, &ptl);
	   if(!ptep || pte_none(*ptep))
		   goto out;
	   pte = *ptep;
	   if (!pte_present(pte))
		   goto out;
	   /*   pfn = pte_pfn(pte);
    if(!pfn_valid(pfn))
            goto out;
    page = pfn_to_page(pfn);
    if(!page)
	    goto out;
    printk(KERN_INFO "original pte with flags %lx\n", pte_flags(pte));*/
	   pte_clear(mm, change_address, ptep);
	   __flush_tlb();
	   pte_unmap_unlock(ptep, ptl);
	   // if(!scan_page)
	   //	continue;

	   /*TODO What if the page is reallocated to a process which
        needs LOCKED */

	   //XXX 21th mlock removed   change_page->flags &= ~__PG_MLOCKED;

	   /*XXX not needed

    if(scan_page==change_page) {
         scan_page->flags &= ~__PG_MLOCKED;

	 anon_vma = scan_vma->anon_vma;
	 anon_vma = (void*)( (unsigned long)anon_vma+ PAGE_MAPPING_ANON);
	 scan_page->mapping= (struct address_space *)anon_vma;
	 temp_anon=(void*)((unsigned long)scan_page->mapping & ~PAGE_MAPPING_FLAGS);
	 temp_anon=(void*)((unsigned long)anon_vma-PAGE_MAPPING_ANON);
	 temp_anon=(void*)((unsigned long)temp_anon|PAGE_MAPPING_ANON);
	 temp_anon=(void*)((unsigned long)temp_anon & ~PAGE_MAPPING_FLAGS);  
 	 atomic_dec(&scan_page->_mapcount);
	 put_page(scan_page);
    }


   if(scan_page!=change_page) {
	  printk(KERN_INFO "\n Scan page not equal to change page \n"); 
          change_page->flags &= ~__PG_MLOCKED;
	  atomic_dec(&change_page->_mapcount);
	  put_page(change_page);
   }
	    */
	   //  if(page_mapcount(scan_page)<1)
	   //	printk(KERN_INFO "within release mapping mapcount -1\n");
	   //	put_page(scan_page);
	   //   *cuda_pmd = (unsigned long)pmd;
	   //put_page(scan_page);
	   return 1;
	   out:
	   return 0;


 }

 static struct page* zap_cuda_mapping (struct vm_area_struct *vma, unsigned long address, 
		 unsigned long *cuda_pmd)
 {
	 unsigned long pfn;
	 pgd_t *pgd;
	 pud_t *pud;
	 pmd_t *pmd;
	 pte_t *ptep = NULL, pte;
	 spinlock_t *ptl;
	 struct page *page=NULL;
	 //   struct anon_vma *an_vma;
	 struct mm_struct *mm = vma->vm_mm;
	 *cuda_pmd = 0;
	 //struct mm_struct *mm = process_mm;
	 pgd = pgd_offset(mm, address);
	 if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
		 goto out;
	 pud = pud_offset(pgd, address);
	 if (pud_none(*pud))
		 goto out;
	 if ((pud_val(*pud) & _PAGE_PSE) && vma->vm_flags & VM_HUGETLB) {
		 goto out;
		 /*
                page = follow_huge_PUD(mm, address, pud);
                goto got_page;*/
	 }
	 if (unlikely(pud_bad(*pud)))
		 goto out;

	 pmd = pmd_offset(pud, address);
	 if (pmd_none(*pmd))
		 goto out;
	 if ((pmd_val(*pmd) & _PAGE_PSE) && vma->vm_flags & VM_HUGETLB) {
		 /*
                page = follow_huge_PMD(mm, address, pmd);
                goto got_page;*/
		 goto out;
	 }
	 if (pmd_trans_huge(*pmd)){
		 goto out;

		 /*    spin_lock(&mm->page_table_lock);
                if (likely(pmd_trans_huge(*pmd))) {
                                page = follow_trans_huge_PMD(mm, address,
                                                             pmd);
                                spin_unlock(&mm->page_table_lock);
                                goto got_page;
                } else
                        spin_unlock(&mm->page_table_lock);*/

	 }
	 if (unlikely(pmd_bad(*pmd)))
		 goto out;

	 *cuda_pmd = (unsigned long)pmd; 

	 ptep = pte_offset_map_lock(mm, pmd, address, &ptl);
	 if(!ptep)
		 goto out;
	 pte = *ptep;
	 if (pte_none(pte) || !pte_present(pte))
		 goto out_unlock;

	 pfn = pte_pfn(pte);
	 if(!pfn_valid(pfn))
		 goto out_unlock;

	 page = pfn_to_page(pfn);
	 if(!page)
		 goto out_unlock;
	 // printk(KERN_INFO "original pte with flags %lx\n", pte_flags(pte));
	 pte_clear(mm, address, ptep);
	 __flush_tlb();
	 pte_unmap_unlock(ptep, ptl);
	 //  *cuda_pmd = (unsigned long)pmd; XXX commented on 28th oct 2016
	 return page;
	 out_unlock:    
	 pte_unmap_unlock(ptep, ptl);
	 out:
	 return NULL;
 }


 int try_change_n_pages(struct vm_area_struct *to_scan, struct mm_struct *mm_to_change, unsigned long change_address, unsigned long *mapped_pages_list)
 {
	 struct page *target_page;
	 unsigned long scan_address = to_scan->vm_start;
	 struct vm_area_struct *tar_vma = NULL;
	 int done = 0;

	 // if(num <= 0)
	 //      return 0;
	 for( ;scan_address < to_scan->vm_end; scan_address += PAGE_SIZE) {
		 struct page *page;
		 unsigned long cuda_pmdp;
		 pmd_t *pmd;
		 pte_t *pte;
		 pte_t entry;
		 spinlock_t *ptl;
		 //	  struct anon_vma *anon_vma; 

		 target_page = get_page_from_addr(to_scan, scan_address);
		 if(!target_page){
			 // printk(KERN_INFO "target not mapped\n");
			 continue;
		 }

		 /*XXX 24thOct Commented


	 if(num_mapped_pages<525000)
	 {
		 mapped_pages_list[num_mapped_pages]=scan_address;
		 num_mapped_pages++;
	 }
		  */	 

		 if(page_mapcount(target_page)>1)
		 {
			 if(num_mapped_pages<525000)
			 {
				 mapped_pages_list[num_mapped_pages]=scan_address;
				 num_mapped_pages++;
			 }

			 continue; 
		 }
		 tar_vma = find_vma(mm_to_change, change_address);
		 if(!tar_vma)
			 return -1;

		 page = zap_cuda_mapping (tar_vma, change_address, &cuda_pmdp);

		 //XXX Below code added on 28th Oct 2016 the  if (page) part
		 if(page)
		 {
			 page->mapping = NULL;
			 atomic_dec(&page->_mapcount);
			 put_page(page); //At this point CUDA is free

		 }
		 if(!cuda_pmdp) {
			 printk(KERN_INFO  " NO cuda PMDP found\n");
			 return -1;
		 }

		 //   **21th      get_page(target_page);
		 // target_page->mapping = page->mapping;
		 //XXX 21th mlock removed	 target_page->flags |= __PG_MLOCKED;
		 //   **21th     atomic_inc(&target_page->_mapcount);
		 //	 anon_vma = tar_vma->anon_vma;
		 //	 anon_vma = (void*) anon_vma+ PAGE_MAPPING_ANON;
		 //	 target_page->mapping= (struct address_space *)anon_vma;
		 //	 target_page->index=linear_pindex(tar_vma,change_address);
		 //	 page->mapping = NULL;
		 //	 atomic_dec(&page->_mapcount);
		 //	 page->flags &= ~__PG_MLOCKED;
		 //	 put_page(page); //At this point CUDA is free
		 pmd = (pmd_t *) cuda_pmdp;


		 pte = pte_offset_map_lock (mm_to_change, pmd, change_address, &ptl); 
		 entry=mk_pte(target_page, tar_vma->vm_page_prot);	//here page_prot should be of CUDA Host Alloc pages. And if there is a mapping, you need to release the page.
		 // printk(KERN_INFO "setting pte with flags %lx\n", pte_flags(entry));
		 set_pte_at(mm_to_change, change_address, pte, entry);
		 pte_unmap_unlock(pte, ptl);
		 addto_page_mapping_list(scan_address,change_address); //function to store mappings
		 //printk(KERN_INFO "Setting for the source address  %lx and target address %lx \n", scan_address,change_address);
		 change_address += 4096;
		 done++;
		 //	 if(mappings_changed>2000)
		 //		return done;
		 //num--;

	 }
	 return done;
 }


 void change_mapping(struct mm_struct *source_mm, struct vm_area_struct *tar_vma, unsigned long tar_address,unsigned long *mapped_pages_list) {


	 struct vm_area_struct *source_vma = NULL;
	 //      int total_maps = 100;
	 mappings_changed =0; //This variable is used for accounting purpose when transferring mapping data to user space
	 WARN_ON(!source_mm || !source_mm->mmap);

	 for(source_vma=source_mm->mmap; source_vma; source_vma=source_vma->vm_next) {
		 if((source_vma->vm_flags & VM_MERGEABLE) && source_vma->anon_vma) {
			 int done = try_change_n_pages(source_vma, tar_vma->vm_mm, tar_address,mapped_pages_list);
			 WARN_ON(done < 0);     
			 tar_address += done * PAGE_SIZE;
			 // total_maps -= done;
			 total_pages_mapped+=done;
		 }

		 //	    if(total_pages_mapped>20000)
		 //	   if(mappings_changed>2000)
		 //		return;
	 } 

	 return;
 }	





 /*

	The "mapping" variable is struct page points to the anon_vma data structure which contain list of all
	processs' VMAs that share that page.
  */


 // Old code of change_mapping : commented by Anshuj

 int give_hints_to_ksm(unsigned long *input) {
	 return 0;
 }

 /** @brief The LKM initialization function
  *  The static keyword restricts the visibility of the function to within this C file. The __init
  *  macro means that for a built-in driver (not a LKM) the function is only used at initialization
  *  time and that it can be discarded and its memory freed up after that point.
  *  @return returns 0 if successful
  */
 static int __init ebbchar_init(void){

	 printk(KERN_INFO "EBBChar: Initializing the EBBChar LKM\n");

	 // Try to dynamically allocate a major number for the device -- more difficult but worth it
	 majorNumber = register_chrdev(0, DEVICE_NAME, &fops);
	 if (majorNumber<0){
		 printk(KERN_ALERT "EBBChar failed to register a major number\n");
		 return majorNumber;
	 }
	 printk(KERN_INFO "EBBChar: registered correctly with major number %d\n", majorNumber);

	 // Register the device class
	 ebbcharClass = class_create(THIS_MODULE, CLASS_NAME);
	 if (IS_ERR(ebbcharClass)){                // Check for error and clean up if there is
		 unregister_chrdev(majorNumber, DEVICE_NAME);
		 printk(KERN_ALERT "Failed to register device class\n");
		 return PTR_ERR(ebbcharClass);          // Correct way to return an error on a pointer
	 }
	 printk(KERN_INFO "EBBChar: device class registered correctly\n");

	 // Register the device driver
	 ebbcharDevice = device_create(ebbcharClass, NULL, MKDEV(majorNumber, 0), NULL, DEVICE_NAME);
	 if (IS_ERR(ebbcharDevice)){               // Clean up if there is an error
		 class_destroy(ebbcharClass);           // Repeated code but the alternative is goto statements
		 unregister_chrdev(majorNumber, DEVICE_NAME);
		 printk(KERN_ALERT "Failed to create the device\n");
		 return PTR_ERR(ebbcharDevice);
	 }
	 printk(KERN_INFO "EBBChar: device class created correctly\n"); // Made it! device was initialized
	 get_gpu_hints=give_hints_to_ksm;
	 //total_hints=10;
	 // hints_available=1;
	 return 0;
 }

 /** @brief The LKM cleanup function
  *  Similar to the initialization function, it is static. The __exit macro notifies that if this
  *  code is used for a built-in driver (not a LKM) that this function is not required.
  */
 static void __exit ebbchar_exit(void){
	 get_gpu_hints=NULL;
	 device_destroy(ebbcharClass, MKDEV(majorNumber, 0));     // remove the device
	 class_unregister(ebbcharClass);                          // unregister the device class
	 class_destroy(ebbcharClass);                             // remove the device class
	 unregister_chrdev(majorNumber, DEVICE_NAME);             // unregister the major number
	 printk(KERN_INFO "EBBChar: Goodbye from the LKM!\n");
 }

 /** @brief The device open function that is called each time the device is opened
  *  This will only increment the numberOpens counter in this case.
  *  @param inodep A pointer to an inode object (defined in linux/fs.h)
  *  @param filep A pointer to a file object (defined in linux/fs.h)
  */
 static int dev_open(struct inode *inodep, struct file *filep){
	 numberOpens++;
	 printk(KERN_INFO "EBBChar: Device has been opened %d time(s)\n", numberOpens);
	 return 0;
 }

 /** @brief This function is called whenever device is being read from user space i.e. data is
  *  being sent from the device to the user. In this case is uses the copy_to_user() function to
  *  send the buffer string to the user and captures any errors.
  *  @param filep A pointer to a file object (defined in linux/fs.h)
  *  @param buffer The pointer to the buffer to which this function writes the data
  *  @param len The length of the b
  *  @param offset The offset if required
  */
 static ssize_t dev_read(struct file *filep, char *buffer, size_t len, loff_t *offset){
	 int error_count = 0;
	 // copy_to_user has the format ( * to, *from, size) and returns 0 on success
	 error_count = copy_to_user(buffer, message, size_of_message);

	 if (error_count==0){            // if true then have success
		 printk(KERN_INFO "EBBChar: Sent %d characters to the user\n", size_of_message);
		 return (size_of_message=0);  // clear the position to the start and return 0
	 }
	 else {
		 printk(KERN_INFO "EBBChar: Failed to send %d characters to the user\n", error_count);
		 return -EFAULT;              // Failed -- return a bad address message (i.e. -14)
	 }
 }



 /** @brief This function is called whenever the device is being written to from user space i.e.
  *  data is sent to the device from the user. The data is copied to the message[] array in this
  *  LKM using the sprintf() function along with the length of the string.
  *  @param filep A pointer to a file object
  *  @param buffer The buffer to that contains the string to write to the device
  *  @param len The length of the array of data that is being passed in the const char buffer
  *  @param offset The offset if required
  */

 //dev_write no being used : Anshuj

 static ssize_t dev_write(struct file *filep, const char *buffer, size_t len, loff_t *offset){
	 //int process_id=0;
	 //struct task_struct* p_tasks;
	 // int ret;
	 sprintf(message, "%s(%d letters)", buffer, (int)len);   // appending received string with its length
	 size_of_message = strlen(message);                 // store the length of the stored message
	 printk(KERN_INFO "EBBChar: Received %u characters from the user\n", (unsigned)len);
	 return len;
 }
 /** @brief The device release function that is called whenever the device is closed/released by
  *  the userspace program
  *  @param inodep A pointer to an inode object (defined in linux/fs.h)
  *  @param filep A pointer to a file object (defined in linux/fs.h)
  */
 struct proc_info {
	 unsigned int source_process_id;
	 unsigned int target_process_id;
	 unsigned long target_address;
 };

 static long dev_ioctl(struct file *filep, unsigned int cmd,unsigned long arg) {
	 //	struct iovec *vm_address;

	 unsigned long *vm_addresses;
	 ssize_t bytes_written;
	 struct page_mappings *page_mapping;
	 struct proc_info *process_info;
	 struct proc_info *p_info;	
	 //	int count=0;
	 //	unsigned long *total_hints_ptr;
	 //	unsigned long total_hints;

	 unsigned long *hints;
	 int i;
	 switch(cmd) {

	 case WRITE_PROC_INFO:
		 //should be divided by 8 is_page_mapped and initialized with 0
		 //			is_page_mapped=(char*)vmalloc(525000*sizeof(char));			
		 mapped_pages_list=(unsigned long*)vmalloc(525000*sizeof(unsigned long));
		 process_info= (struct proc_info*)vmalloc(sizeof(struct proc_info));
		 p_info=process_info;
		 bytes_written=copy_from_user(process_info,(struct proc_info*)arg,sizeof(struct proc_info));
		 source_process_id=process_info->source_process_id;
		 target_process_id=process_info->target_process_id;
		 target_address=process_info->target_address;
		 process_tasks=pid_task(find_vpid(source_process_id), PIDTYPE_PID);
		 if(process_tasks->pid==source_process_id)
		 {
			// printk(KERN_INFO "pid=%d, state=%ld \n",process_tasks->pid,process_tasks->state);
			 source_mm=process_tasks->mm;
			 //scan_hints.mm = source_mm;
			 // XXX updating ksm scan list
			 	 
			 if(lookup_ksm_hints(source_mm)==0)
			 {
			 	 struct ksm_scan_hints *node,*head;
			 	 head=scan_hints;
				 node=(struct ksm_scan_hints *)vmalloc(sizeof(struct ksm_scan_hints));
				 node->next_mm_to_scan=NULL;
				 node->mm=source_mm;
				 if(head==NULL)
				 {
					 scan_hints=node;
					 //ksm_hints_head=node;
					 ksm_hints_tail=node;
					 node->prev_mm_to_scan=NULL;
				 }
				 else
				 {
					 node->prev_mm_to_scan=ksm_hints_tail;
					 ksm_hints_tail->next_mm_to_scan=node;
					 ksm_hints_tail=node;
				 }
				// printk(KERN_INFO "Source mm %p \n",source_mm);
				 				 
			 }
				 
			// XXX end of updating ksm scan list
			 process_mm=source_mm;
		 }
		 
		 update_pid_to_mm(source_mm,source_process_id);
		// printk(KERN_INFO "Source Process_Id %d \n",source_process_id);

		 process_tasks=pid_task(find_vpid(target_process_id), PIDTYPE_PID);
		 if(process_tasks->pid==target_process_id)
		 {
			// printk(KERN_INFO "pid=%d, state=%ld \n",process_tasks->pid,process_tasks->state);
			 target_mm=process_tasks->mm;
			 process_mm=target_mm;
		 }
		// printk(KERN_INFO "Target Process_Id %d \n",target_process_id);

		// printk(KERN_INFO "The target address passed is %lx",target_address);
		 process_vma=find_vma(target_mm,target_address);
		 target_vma=process_vma;
		 /*
		 if(process_vma!=NULL)
			 printk(KERN_INFO "The vma area struct of target virtual address is %p\n",process_vma);*/
		 //      		 process_page= get_page_from_addr(process_vma,target_address);

		 //printk(KERN_INFO "Address of page structure is %p, Value of flags = %lu \n",process_page,process_page->flags);
		 // print_page_flags(p_page);
		 // printk(KERN_INFO "Printing VMA flags of HOST Varitable \n");
		 // print_vma_flags(p_vma);
		 change_mapping(source_mm,target_vma,target_address,mapped_pages_list);
		// printk(KERN_INFO "Mapping changed Successfully\n");
		 //print_mappings();
		 wbinvd();
		 break;
	 case READ_NUM_MAPPINGS:
		 bytes_written=copy_to_user((unsigned int*)arg,&mappings_changed,sizeof(unsigned int));
		 break;
	 case READ_NUM_PAGES_MAPPED:
		 bytes_written=copy_to_user((unsigned int*)arg,&num_mapped_pages,sizeof(unsigned int));
		 break;
	 case READ_PAGES_MAPPED:
		// printk("First and last mapped page is %lu and %lu \n",mapped_pages_list[0],mapped_pages_list[num_mapped_pages-1]);
		 //divide it by 8 (52500)
		 bytes_written=copy_to_user((unsigned long*)arg,mapped_pages_list,num_mapped_pages*sizeof(unsigned long));
		 break;
	 case READ_MAPPINGS:
		 page_mapping=page_mapping_head;
		 //	vm_address = (struct iovec*)vmalloc(mappings_changed*sizeof(struct iovec))		;
		 vm_addresses= (unsigned long*)vmalloc(mappings_changed*sizeof(unsigned long));
		 for(i=0;i<mappings_changed && page_mapping!=NULL;i++) {
			 //			vm_address[i].iov_base =(unsigned long)page_mapping->vm_virt_address;
			 //			vm_address[i].iov_len = sizeof(unsigned long);
			 vm_addresses[i]=page_mapping->vm_virt_address;
			 page_mapping=page_mapping->next_mapping;

		 }
		 //		bytes_written=writev(filep,vm_address,mappings_changed);
		 //		bytes_written=copy_to_user((struct iovec *)arg,vm_address,mappings_changed*sizeof(struct iovec));
		 bytes_written=copy_to_user((unsigned long*)arg,vm_addresses,mappings_changed*sizeof(unsigned long));
		 //		kfree();
		 break;
	 case WRITE_HINTS_PID:
		 bytes_written=copy_from_user(&current_pid,(unsigned int*)arg,sizeof(unsigned int));
		 
		 break;
	 case WRITE_HINTS_NUM: //This also clears the page_mapping of cuda_process

		 //			copy_from_user(total_hints_ptr,(unsigned long*)arg,sizeof(unsigned long));
		 //			total_hints=*total_hints_ptr;
		 total_hints=arg;
		 current_total_hints = arg;
		 break;
	 case WRITE_HINTS:
		 if(total_hints<=0) {
			 printk(KERN_INFO "Error: No hints given");
			 break;
		 }
		 else {
			 //hints = (unsigned long*)vmalloc(total_hints*sizeof(unsigned long)); //vfree???
			// bytes_written=copy_from_user(hints,(unsigned long*)arg,total_hints*sizeof(unsigned long));
			 //printk(KERN_INFO "total_hints in module= %d\n",total_hints);			
			 //scan_hints.address= (unsigned long*)vmalloc(total_hints*sizeof(unsigned long)); 
			 //XXX dont forget to free above memory before vmallocing again : memory leak
			 //scan_hints.num_hints=total_hints;
			 /*
			 for(i=0;i<total_hints;i++) {
				 //	printk(KERN_INFO "The Hint number %d is %lx\n",i,hints[i]);
				 scan_hints.address[i]=hints[i];
				 //	printk(KERN_INFO "KSM scan Hints number %d is %lx\n",i,scan_hints.address[i]);
			 }*/
			 
			 //Remember to put sanity checks if needed
			 current_mm=pid_to_mm_lookup(current_pid);
			 current_scan_hints_struct = get_scan_hints_struct(current_mm);
			 current_scan_hints_struct->num_hints=current_total_hints;
			 current_scan_hints_struct->address= (unsigned long*)vmalloc(current_total_hints*sizeof(unsigned long));
			 bytes_written=copy_from_user(current_scan_hints_struct->address,(unsigned long*)arg,current_total_hints*sizeof(unsigned long));
		 			 
			// hints_available=1;

		 }

		 /*	while(count<5) {
				msleep(2000);
				if(hints_available==0) {
					hints_available=1;
				count++;
				}
			}
			count=0;*/
		 break;
	 case CLEAR_MAPPINGS:
		 page_mapping=page_mapping_head;
		 for(i=0;i<mappings_changed;i++) 
		 {
			 WARN_ON(clear_cuda_mappings(page_mapping->cuda_virt_address,page_mapping->vm_virt_address) == 0);
			 page_mapping=page_mapping->next_mapping;
		 }
		 break;
	 case START_SCAN:
		 hints_available=1;
		 break;
	 case GARBAGE_COLLECT:
		 garbage_collection();
		 break;
	 case CLEAR_PID_TO_MM:
		 traverse_ksm_hints();
		 free_pid_to_mm();		 
		 break;
	 case ARE_HINTS_EXHAUSTED:
		 bytes_written=copy_to_user((int*)arg,&hints_available,sizeof(int));
		 break;
    	 case HINTS_SHARED:
	 	 bytes_written=copy_to_user((unsigned int*)arg,&hints_shared_per_round,sizeof(unsigned int));
 		 hints_shared_per_round=0;
		 break;
	 default:
		 return -ENOTTY;
	 }
	 return 0;
 }


 static int dev_release(struct inode *inodep, struct file *filep){
	 printk(KERN_INFO "EBBChar: Device successfully \n");
	 write_counter=0;
	 return 0;
 }

 /** @brief A module must use the module_init() module_exit() macros from linux/init.h, which
  *  identify the initialization function at insertion time and the cleanup function (as
  *  listed above)
  */
 module_init(ebbchar_init);
 module_exit(ebbchar_exit);

