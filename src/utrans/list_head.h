#ifndef INCLUDE_LIB2EASY_LIST_HEAD_H_
#define INCLUDE_LIB2EASY_LIST_HEAD_H_

#include <stddef.h>

#ifdef _MSC_VER
#    define inline __inline
#endif

#define container_of(obj, type, memb) ((type*)(((char*)(obj)) - offsetof(type, memb)))

#define LIST_POISON1 ((void*)0x0000100)
#define LIST_POISON2 ((void*)0x0000200)

/* 双向链表 */
struct list_head {
    struct list_head *next, *prev;
};

/* conflict with MAC */
#undef LIST_HEAD

#define LIST_HEAD_INIT(name) \
    { &(name), &(name) }
#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)

static inline void INIT_LIST_HEAD(struct list_head* list) {
    list->next = list;
    list->prev = list;
}

static inline void __list_add(struct list_head* new_head, struct list_head* prev, struct list_head* next) {
    next->prev = new_head;
    new_head->next = next;
    new_head->prev = prev;
    prev->next = new_head;
}

static inline void list_add(struct list_head* new_head, struct list_head* head) {
    __list_add(new_head, head, head->next);
}

static inline void list_add_tail(struct list_head* new_head, struct list_head* head) {
    __list_add(new_head, head->prev, head);
}

static inline void __list_del(struct list_head* prev, struct list_head* next) {
    next->prev = prev;
    prev->next = next;
}

static inline void __list_del_entry(struct list_head* entry) {
    __list_del(entry->prev, entry->next);
}

static inline void list_del(struct list_head* entry) {
    __list_del(entry->prev, entry->next);
    entry->next = (struct list_head*)LIST_POISON1;
    entry->prev = (struct list_head*)LIST_POISON2;
}

static inline int list_empty(const struct list_head* head) {
    return head->next == head;
}

static inline unsigned list_length(const struct list_head* phead) {
    unsigned item_num = 0U;
    struct list_head* pnext = phead->next;
    while (pnext != phead) {
        ++item_num;
        pnext = pnext->next;
    }
    return item_num;
}

static inline void list_del_init(struct list_head* entry) {
    __list_del(entry->prev, entry->next);
    INIT_LIST_HEAD(entry);
}

/* delete from one list and add as another's head */
static inline void list_move(struct list_head* list, struct list_head* head) {
    __list_del_entry(list);
    list_add(list, head);
}

static inline void list_move_tail(struct list_head* list, struct list_head* head) {
    __list_del(list->prev, list->next);
    list_add_tail(list, head);
}

static inline void list_rotate_left(struct list_head* head) {
    struct list_head* first;

    if (!list_empty(head)) {
        first = head->next;
        list_move_tail(first, head);
    }
}

static inline int list_is_singular(const struct list_head* head) {
    return !list_empty(head) && (head->next == head->prev);
}

/**
 * list_is_last - tests whether @list is the last entry in list @head
 * @list: the entry to test
 * @head: the head of the list
 */
static inline int list_is_last(const struct list_head* list, const struct list_head* head) {
    return list->next == head;
}

static inline void list_replace(struct list_head* old, struct list_head* new_head) {
    new_head->next = old->next;
    new_head->next->prev = new_head;
    new_head->prev = old->prev;
    new_head->prev->next = new_head;
}

static inline void list_replace_init(struct list_head* old, struct list_head* new_head) {
    list_replace(old, new_head);
    INIT_LIST_HEAD(old);
}

static inline void list_merge(struct list_head* head_keep, struct list_head* head_drop) {
    struct list_head* pkeep_head = head_keep->next;
    struct list_head* pdrop_tail = head_drop->prev;

    if (!list_empty(head_drop)) {
        if (!list_empty(head_keep)) {
            head_keep->next = head_drop->next;
            head_drop->next->prev = head_keep;
            pdrop_tail->next = pkeep_head;
            pkeep_head->prev = pdrop_tail;
        } else {
            head_keep->next = head_drop->next;
            head_keep->next->prev = head_keep;
            head_keep->prev = head_drop->prev;
            head_keep->prev->next = head_keep;
        }
        head_drop->next = head_drop;
        head_drop->prev = head_drop;
    }
}

#define list_entry(ptr, type, member) container_of(ptr, type, member)

#define list_first_entry(ptr, type, member) (list_empty((ptr)) ? NULL : list_entry((ptr)->next, type, member))

#define list_last_entry(list, type, member) (list_empty((list)) ? NULL : list_entry((list)->prev, type, member))

/**
 * list_next_entry - get the next element in list
 * @pos:	the type * to cursor
 * @member:	the name of the list_head within the struct.
 */
#define list_next_entry(pos, type, member) list_entry((pos)->member.next, type, member)

#define list_for_each(pos, head) for (pos = (head)->next; pos != (head); pos = pos->next)

#define list_for_each_reverse(pos, head) for (pos = (head)->prev; pos != (head); pos = pos->prev)

#define list_for_each_entry_safe(pos, n, head, member) \
    for (pos = list_entry((head)->next, typeof(*pos), member), n = list_entry(pos->member.next, typeof(*pos), member); \
         &pos->member != (head); \
         pos = n, n = list_entry(n->member.next, typeof(*n), member))

#define list_for_each_safe(pos, n, head) for (pos = (head)->next, n = pos->next; pos != (head); pos = n, n = pos->next)

#define list_for_each_reverse_safe(pos, p, head) \
    for (pos = (head)->prev, p = pos->prev; pos != (head); pos = p, p = pos->prev)

#define list_for_each_entry(pos, head, member) \
    for (pos = list_entry((head)->next, typeof(*pos), member); &pos->member != (head); \
         pos = list_entry(pos->member.next, typeof(*pos), member))

#define list_for_each_type_entry(pos, head, member, type) \
    for (pos = list_entry((head)->next, type, member); &pos->member != (head); \
         pos = list_entry(pos->member.next, type, member))

#define list_for_each_entry_reverse(pos, head, member) \
    for (pos = list_entry((head)->prev, typeof(*pos), member); &pos->member != (head); \
         pos = list_entry(pos->member.prev, typeof(*pos), member))

#define list_for_each_entry_safe_reverse(pos, n, head, member) \
    for (pos = list_entry((head)->prev, typeof(*pos), member), n = list_entry(pos->member.prev, typeof(*pos), member); \
         &pos->member != (head); \
         pos = n, n = list_entry(n->member.prev, typeof(*n), member))

#define list_for_each_type_entry_safe_reverse(pos, n, head, member, type) \
    for (pos = list_entry((head)->prev, type, member), n = list_entry(pos->member.prev, type, member); \
         &pos->member != (head); \
         pos = n, n = list_entry(n->member.prev, type, member))

#define list_prepare_entry(pos, head, member) ((pos) ?: list_entry(head, typeof(*pos), member))

#define list_for_each_entry_continue(pos, head, member, type) \
    for (pos = list_entry(pos->member.next, type, member); &pos->member != (head); \
         pos = list_entry(pos->member.next, type, member))

#define list_for_each_entry_continue_reverse(pos, head, member, type) \
    for (pos = list_entry(pos->member.prev, type, member); &pos->member != (head); \
         pos = list_entry(pos->member.prev, type, member))

#endif // INCLUDE_LIB2EASY_LIST_HEAD_H_