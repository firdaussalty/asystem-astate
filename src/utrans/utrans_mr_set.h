#ifndef INCLUDE_UTRANS_MR_SET_H_
#define INCLUDE_UTRANS_MR_SET_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct avl_node;

typedef struct avl_tree {
    struct avl_node* root;
    int size;
} mr_set_t;

typedef int (*item_oper_cb)(void* item_val, void* cb_arg);

/// init mr_ser_t, or simply memset it
void utrans_mr_set_init(mr_set_t* pmr_set);

/// insert one registered memory region
/// @note rejects memory region overlaps existed regions
/// @return 0 when success
int utrans_mr_set_insert(mr_set_t* pmr_set, void* addr, size_t len, void* pitem);

/// find corresponding memory region
/// @return target memory region contains input [addr, adrr+len],
/// will return null when addr range overlaps existed regions
void* utrans_mr_set_find(mr_set_t* pmr_set, void* addr, size_t len);

/// remove corresponding memory region
/// @return target memory region contains input [addr, adrr+len],
/// will return null and do nothing when addr range overlaps existed regions
void* utrans_mr_set_remove(mr_set_t* pmr_set, void* addr, size_t len);

/// get all registered memory regions array
/// @param psize returned array length
/// @return all registered memory regions array, shall free() after use
void** utrans_mr_set_getall(mr_set_t* pmr_set, int* psize);

/// get all registered memory regions array, with cb on each element
/// @param cb callback function on each element, cb shall return 0 when succ,
/// return others will stop traverse
/// @param cb_arg input param for cb
/// @return 0 for success, otherwise error happend
int utrans_mr_set_traverse(mr_set_t* pmr_set, item_oper_cb cb, void* cb_arg);

/// returns registered memory region number
int utrans_mr_set_getsize(mr_set_t* pmr_set);

/// destroy mr_set_t instance
/// @param cb callback function on each element, ret code is ignored,
///  so callback will always applied on each element
/// @param cb_arg input param for cb
/// @note has no effect on registered memory regions if cb is NULL,
/// better to traverse all items before release in-case-of mem-leak
void utrans_mr_set_destroy(mr_set_t* pmr_set, item_oper_cb cb, void* cb_arg);

#ifdef __cplusplus
}
#endif

#endif