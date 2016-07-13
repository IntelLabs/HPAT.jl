#ifndef HPAT_SORT_H_
#define HPAT_SORT_H_

int __hpat_sort_compare(int64_t x, int64_t y){
  if (x < y)
    return -1;
  else if (x == y)
    return 0;
  else
    return 1;
}


//Declarations
static void __hpat_binary_insertionsort_index(int64_t *comp_arr, const size_t start, const size_t size, int64_t ** all_arrs ,const size_t all_arrs_len);
static int64_t __hpat_binary_insertionsort_search(int64_t *comp_arr, const int64_t x, const size_t size);

// Basic binary search to find the right position
static __inline int64_t __hpat_binary_insertionsort_search(int64_t *comp_arr, const int64_t elem,const size_t size) {
  int64_t low, mid, high, pivot;
  low = 0;
  high = size - 1;
  mid = high >> 1;
  // If it is less than low
  if (__hpat_sort_compare(elem, comp_arr[0]) < 0) {
    return 0;
  } else if (__hpat_sort_compare(elem, comp_arr[high]) > 0) {
    return high;
  }
  pivot = comp_arr[mid];
  while (1) {
    if (__hpat_sort_compare(elem, pivot) < 0) {
      if (mid - low <= 1) {
        return mid;
      }
      high = mid;
    } else {
      if (high - mid <= 1) {
        return mid + 1;
      }
      low = mid;
    }
    mid = low + ((high - low) >> 1);
    pivot = comp_arr[mid];
  }
}

// Binary search with different starting index
static void __hpat_binary_insertionsort_index(int64_t *comp_arr, const size_t start, const size_t size, int64_t ** all_arrs ,const size_t all_arrs_len) {
  for (int64_t ind_start = start; ind_start < size; ind_start++) {
    int64_t ind_curr, elem, pivot;
    // Already sorted
    if (__hpat_sort_compare(comp_arr[ind_start - 1], comp_arr[ind_start]) <= 0) {
      continue;
    }
    elem = comp_arr[ind_start];
    int64_t temp_x[all_arrs_len];
    for (int k = 0 ; k < all_arrs_len ; k++){
      int64_t * cur_arr = all_arrs[k];
      temp_x[k]= cur_arr[ind_start];
    }
    pivot = __hpat_binary_insertionsort_search(comp_arr, elem, ind_start);
    for (ind_curr = ind_start - 1; ind_curr >= pivot; ind_curr--) {
      //std::cout << "swapping" << std::endl;
      comp_arr[ind_curr + 1] = comp_arr[ind_curr];
      for (int k = 0 ; k < all_arrs_len ; k++){
	int64_t * cur_arr = all_arrs[k];
	cur_arr[ind_curr + 1] = cur_arr[ind_curr];
      }
    }
    comp_arr[pivot] = elem;
    for (int k = 0 ; k < all_arrs_len ; k++){
      int64_t * cur_arr = all_arrs[k];
      cur_arr[pivot] = temp_x[k];
    }
  }
}

void TIM_SORT(int64_t *dst, const size_t size,int64_t ** all_arrs,const size_t all_arrs_len) {
  std::cout << "TIM SORT" << std::endl;
  if (size <= 1) {
    return;
  }
  if (size < 64) {
    __hpat_binary_insertionsort_index(dst, 1,size,all_arrs, all_arrs_len);
    return;
  }
}

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
