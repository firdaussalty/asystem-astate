#ifndef INCLUDE_ID_HASH_MAP_H_
#define INCLUDE_ID_HASH_MAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

// TODO: replace with lib2easy's maphash for better performance

typedef struct id_hash_map_t id_hash_map_t;

/**
 * @brief create an instance of hash map
 * 
 * @param bucket_count 
 * @param lock_count 
 * @param alloc_func 
 * @param free_func 
 * @return id_hash_map_t* NULL if failed
 */
id_hash_map_t*
idhm_create(size_t bucket_count, size_t lock_count, void* (*alloc_func)(size_t), void (*free_func)(void*));

/**
 * @brief destroy the hash map
 * 
 * @param map 
 */
void idhm_destroy(id_hash_map_t* map);

/**
 * @brief insert / overwirte key value
 * 
 * @param map 
 * @param key 
 * @param value new value to insert
 * @param pre_value previous value of the given key if exists
 * @return int -1 failed, 0-insert a bright new key, 1-update a key value
 */
int idhm_insert(id_hash_map_t* map, uint64_t key, uint64_t value, uint64_t* pre_value);

/**
 * @brief 
 * 
 * @param map 
 * @param key 
 * @param value 64bit-value returned if not null
 * @return int return 1 if found, 0 if not found
 */
int idhm_find(id_hash_map_t* map, uint64_t key, void* value);

/**
 * @brief return value of given key if exists, or insert a new key with value_if_not_found
 * 
 * @param map 
 * @param key 
 * @param value 64bit-value returned if not null
 * @param value_if_not_found 64bit-value to insert if not found
 * @return int return 1 if found, 0 if not found & insert, -1 if failed to insert
 */
int idhm_find_or_insert(id_hash_map_t* map, uint64_t key, void* value, uint64_t value_if_not_found);

/**
 * @brief 
 * 
 * @param map 
 * @param key 
 * @param value will be set to 64bit-value if key exists
 * @return int 0 if not key found, 1 if exists and removed
 */
int idhm_remove(id_hash_map_t* map, uint64_t key, void* value);

/**
 * @brief remove item with key's value part matches expected
 *
 * @return 0 if key-value pair not found, 1 for found and removed
 */
int idhm_remove_exists(id_hash_map_t* map, uint64_t key, uint64_t value);

/**
 * @brief get total number of items in the map
 * 
 * @param map 
 * @return long 
 */
long idhm_get_num_items(id_hash_map_t* map);

/**
 * @brief traverse the map
 */
long idhm_traverse(id_hash_map_t* map, int (*item_oper)(uint64_t, uint64_t, void*), void* oper_arg);

#ifdef __cplusplus
}
#endif

#endif