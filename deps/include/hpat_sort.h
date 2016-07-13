#ifndef HPAT_SORT_H_
#define HPAT_SORT_H_

#ifndef SORT_CMP
#define SORT_CMP(x, y)  ((x) < (y) ? -1 : ((x) == (y) ? 0 : 1))
#endif

#define SORT_SWAP(x,y) {int64_t __SORT_SWAP_t = (x); (x) = (y); (y) = __SORT_SWAP_t;}

typedef struct {
  uint64_t start;
  uint64_t length;
} TIM_SORT_RUN_T;

typedef struct {
  size_t alloc;
  int64_t *storage;
} TEMP_STORAGE_T;

int __hpat_quicksort_partition( int64_t ** arr,int64_t size ,int64_t* comp_arr , int low, int high) {
  int64_t pivot, t;
  int i,j;
  pivot = comp_arr[low];
  i = low;
  j = high+1;
  while(1){
      do ++i; while( comp_arr[i] <= pivot && i <= high );
      do --j; while( comp_arr[j] > pivot );
      if( i >= j ) break;
      for(int index =0 ; index< size ; index++){
	int64_t * curr_arr = arr[index];
	t = curr_arr[i]; curr_arr[i] = curr_arr[j]; curr_arr[j] = t;
      }
      t = comp_arr[i]; comp_arr[i] = comp_arr[j]; comp_arr[j] = t;
  }
  for(int index =0 ; index< size ; index++){
	int64_t * curr_arr = arr[index];
	t = curr_arr[low]; curr_arr[low] = curr_arr[j]; curr_arr[j] = t;
	//__hpat_quicksort_swap(curr_arr[i],curr_arr[j]);
  }
  t = comp_arr[low]; comp_arr[low] = comp_arr[j]; comp_arr[j] = t;
  return j;
}

void __hpat_quicksort( int64_t ** arr, int size, int64_t * comp_arr, int low, int high)
{
   int pivot;
   if( low < high )
   {
     pivot = __hpat_quicksort_partition( arr,size,comp_arr, low, high);
     __hpat_quicksort( arr,size, comp_arr, low, pivot-1);
     __hpat_quicksort( arr,size, comp_arr, pivot+1, high);
   }
}

#endif /* HPAT_SORT_H_ */
