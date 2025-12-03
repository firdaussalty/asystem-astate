#include "utrans_mr_set.h"

#include <stdlib.h>

#include "utrans.h"

struct avl_node {
    uint64_t addr;
    size_t len;
    void* pvalue;

    struct avl_node* pleft;
    struct avl_node* pright;
    int height;
};

__attribute__((always_inline)) inline int max(int a, int b) {
    return a < b ? b : a;
}

__attribute__((always_inline)) inline int avl_height(struct avl_node* pnode) {
    return pnode == NULL ? 0 : pnode->height;
}

__attribute__((always_inline)) inline void avl_copy(struct avl_node* pdst, struct avl_node* psrc) {
    pdst->addr = psrc->addr;
    pdst->len = psrc->len;
    pdst->pvalue = psrc->pvalue;
}

static struct avl_node*
avl_node_create(uint64_t key, size_t len, void* pval, struct avl_node* pleft, struct avl_node* pright) {
    struct avl_node* pnode = malloc(sizeof(struct avl_node));
    if (pnode) {
        pnode->addr = key;
        pnode->len = len;
        pnode->pvalue = pval;

        pnode->pleft = pleft;
        pnode->pright = pright;
        pnode->height = 0;
    }
    return pnode;
}

static struct avl_node* avl_ll_rotation(struct avl_node* proot) {
    struct avl_node* pnode;

    // rotate
    pnode = proot->pleft;
    proot->pleft = pnode->pright;
    pnode->pright = proot;

    // re-calculate height
    proot->height = max(avl_height(proot->pleft), avl_height(proot->pright)) + 1;
    pnode->height = max(avl_height(pnode->pleft), avl_height(pnode->pright)) + 1;

    // pnode became new root
    return pnode;
}

static struct avl_node* avl_rr_rotation(struct avl_node* proot) {
    struct avl_node* pnode;

    // rotate
    pnode = proot->pright;
    proot->pright = pnode->pleft;
    pnode->pleft = proot;

    // re-calculate height
    proot->height = max(avl_height(proot->pleft), avl_height(proot->pright)) + 1;
    pnode->height = max(avl_height(pnode->pleft), avl_height(pnode->pright)) + 1;

    // pnode became new root
    return pnode;
}

static struct avl_node* avl_lr_rotation(struct avl_node* proot) {
    proot->pleft = avl_rr_rotation(proot->pleft);
    return avl_ll_rotation(proot);
}

static struct avl_node* avl_rl_rotation(struct avl_node* proot) {
    proot->pright = avl_ll_rotation(proot->pright);
    return avl_rr_rotation(proot);
}

static struct avl_node* avl_insert(struct avl_node* ptree, uint64_t key, size_t len, void* pval, int* poper_fail) {
    if (*poper_fail == 1) {
        return ptree;
    }

    if (NULL == ptree) {
        ptree = avl_node_create(key, len, pval, NULL, NULL);
        if (!ptree) {
            *poper_fail = 1;
            return NULL;
        }
    } else if (key >= ptree->addr + ptree->len) {
        // insert to right branch of this tree
        ptree->pright = avl_insert(ptree->pright, key, len, pval, poper_fail);
        // check balance of current tree
        if (avl_height(ptree->pright) - avl_height(ptree->pleft) == 2) {
            // need re-balance due to height difference larger than 1
            if (key > ptree->pright->addr) {
                ptree = avl_rr_rotation(ptree);
            } else {
                ptree = avl_rl_rotation(ptree);
            }
        }
    } else if (key + len <= ptree->addr) {
        // insert to left branch of this tree
        ptree->pleft = avl_insert(ptree->pleft, key, len, pval, poper_fail);
        // check balance of current tree
        if (avl_height(ptree->pleft) - avl_height(ptree->pright) == 2) {
            // need re-balance due to height difference larger than 1
            if (key < ptree->pleft->addr) {
                ptree = avl_ll_rotation(ptree);
            } else {
                ptree = avl_lr_rotation(ptree);
            }
        }
    } else {
        *poper_fail = 1;
        // node with same key already exists, avoid insert
        return ptree;
    }
    ptree->height = max(avl_height(ptree->pleft), avl_height(ptree->pright)) + 1;
    return ptree;
}

static struct avl_node* avl_max(struct avl_node* ptree) {
    struct avl_node* pnode = ptree;
    if (ptree) {
        while (pnode->pright) {
            pnode = pnode->pright;
        }
    }
    return pnode;
}

static struct avl_node* avl_min(struct avl_node* ptree) {
    struct avl_node* pnode = ptree;
    if (ptree) {
        while (pnode->pleft) {
            pnode = pnode->pleft;
        }
    }
    return pnode;
}

static struct avl_node* avl_find(struct avl_node* ptree, uint64_t addr, size_t len) {
    struct avl_node* pnode = ptree;
    while (pnode != NULL) {
        if (addr + len <= pnode->addr) {
            // search left branch when addr < root's base addr
            pnode = pnode->pleft;
        } else if (addr >= pnode->addr + pnode->len) {
            // search right branch when addr > root's maximum addr
            pnode = pnode->pright;
        } else {
            // addr in mid-item's [0, reg_len], judge whether sub-region of mid-item
            if (addr >= pnode->addr && addr + len <= pnode->addr + pnode->len) {
                return pnode;
            }
            break;
        }
    }
    return NULL;
}

static void avl_destroy(struct avl_node* ptree, struct mem_region_registed** pitem, int* index) {
    if (ptree == NULL) {
        return;
    }
    avl_destroy(ptree->pleft, pitem, index);
    avl_destroy(ptree->pright, pitem, index);
    if (pitem && index) {
        pitem[(*index)++] = ptree->pvalue;
    }
    free(ptree);
}

static void avl_scan(struct avl_node* ptree, void** pitem, int* index) {
    if (ptree == NULL) {
        return;
    }
    avl_scan(ptree->pleft, pitem, index);
    pitem[(*index)++] = ptree->pvalue;
    avl_scan(ptree->pright, pitem, index);
}

static int avl_traverse(struct avl_node* ptree, int judge_cb_ret, item_oper_cb cb_func, void* cb_arg) {
    if (ptree == NULL) {
        return 0;
    }

    int ret;

    ret = avl_traverse(ptree->pleft, judge_cb_ret, cb_func, cb_arg);
    if (judge_cb_ret && ret != 0) {
        return ret;
    }

    if (cb_func) {
        ret = cb_func(ptree->pvalue, cb_arg);
        if (judge_cb_ret && ret != 0) {
            return ret;
        }
    }

    ret = avl_traverse(ptree->pright, judge_cb_ret, cb_func, cb_arg);
    if (judge_cb_ret && ret != 0) {
        return ret;
    }
    return 0;
}

struct avl_node* avl_delete_node(struct avl_node* ptree, struct avl_node* pnode, void** pitem_ret) {
    if (NULL == ptree || pnode == NULL) {
        return ptree;
    } else if (pnode->addr < ptree->addr) {
        // corresponding node locate at left branch
        ptree->pleft = avl_delete_node(ptree->pleft, pnode, pitem_ret);
        // judge balance stat of current tree
        if (avl_height(ptree->pright) - avl_height(ptree->pleft) == 2) {
            struct avl_node* pright = ptree->pright;
            if (avl_height(pright->pleft) > avl_height(pright->pright)) {
                ptree = avl_rl_rotation(ptree);
            } else {
                ptree = avl_rr_rotation(ptree);
            }
        }
    } else if (pnode->addr > ptree->addr) {
        // corresponding node locate at right branch
        ptree->pright = avl_delete_node(ptree->pright, pnode, pitem_ret);
        // judge balance stat of current tree
        if (avl_height(ptree->pleft) - avl_height(ptree->pright) == 2) {
            struct avl_node* pleft = ptree->pleft;
            if (avl_height(pleft->pright) > avl_height(pleft->pleft)) {
                ptree = avl_lr_rotation(ptree);
            } else {
                ptree = avl_ll_rotation(ptree);
            }
        }
    } else {
        if (pitem_ret) {
            *pitem_ret = ptree->pvalue;
        }
        // root node is the target node to delete
        if (ptree->pleft && ptree->pright) {
            // both left and right branch not empty
            if (avl_height(ptree->pleft) > avl_height(ptree->pright)) {
                // height of left branch bigger than right
                struct avl_node* pleft_max = avl_max(ptree->pleft);
                avl_copy(ptree, pleft_max);
                ptree->pleft = avl_delete_node(ptree->pleft, pleft_max, NULL);
            } else {
                struct avl_node* pright_min = avl_min(ptree->pright);
                avl_copy(ptree, pright_min);
                ptree->pright = avl_delete_node(ptree->pright, pright_min, NULL);
            }
        } else {
            // node to free or usless pleft_max/pright_min
            struct avl_node* pfree = ptree;
            ptree = ptree->pleft ? ptree->pleft : ptree->pright;
            free(pfree);
        }
    }
    return ptree;
}

void utrans_mr_set_init(mr_set_t* pmr_set) {
    if (pmr_set) {
        pmr_set->root = NULL;
        pmr_set->size = 0;
    }
}

int utrans_mr_set_insert(mr_set_t* pmr_set, void* addr, size_t len, void* pitem) {
    int op_fail = 0;
    if (!pmr_set || !len || !pitem) {
        return -1;
    }

    pmr_set->root = avl_insert(pmr_set->root, (uint64_t)addr, len, pitem, &op_fail);
    if (!op_fail) {
        pmr_set->size++;
    }
    return op_fail;
}

void* utrans_mr_set_find(mr_set_t* pmr_set, void* addr, size_t len) {
    if (!pmr_set || !len) {
        return NULL;
    }
    struct avl_node* pnode = avl_find(pmr_set->root, (uint64_t)addr, len);
    return pnode ? pnode->pvalue : NULL;
}

void* utrans_mr_set_remove(mr_set_t* pmr_set, void* addr, size_t len) {
    if (!pmr_set || !len) {
        return NULL;
    }
    void* pitem = NULL;
    struct avl_node* pnode = avl_find(pmr_set->root, (uint64_t)addr, len);
    if (pnode) {
        pmr_set->root = avl_delete_node(pmr_set->root, pnode, &pitem);
        if (pitem) {
            pmr_set->size--;
        }
    }
    return pitem;
}

void** utrans_mr_set_getall(mr_set_t* pmr_set, int* psize) {
    if (!pmr_set) {
        return NULL;
    }

    void** pitems = NULL;
    if (pmr_set->size > 0) {
        pitems = malloc(sizeof(void*) * pmr_set->size);
        if (pitems) {
            *psize = 0;
            avl_scan(pmr_set->root, pitems, psize);
        } else {
            *psize = pmr_set->size;
        }
    }
    return pitems;
}

int utrans_mr_set_traverse(mr_set_t* pmr_set, item_oper_cb cb_func, void* cb_arg) {
    if (!pmr_set) {
        return 0;
    }
    return avl_traverse(pmr_set->root, 1, cb_func, cb_arg);
}

int utrans_mr_set_getsize(mr_set_t* pmr_set) {
    if (pmr_set) {
        return pmr_set->size;
    }
    return 0;
}

void utrans_mr_set_destroy(mr_set_t* pmr_set, item_oper_cb cb_func, void* cb_arg) {
    if (!pmr_set) {
        return;
    }
    if (cb_func) {
        avl_traverse(pmr_set->root, 0, cb_func, cb_arg);
    }
    avl_destroy(pmr_set->root, NULL, NULL);
    pmr_set->root = NULL;
    pmr_set->size = 0;
}