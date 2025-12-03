#ifndef INCLUDE_UTRANS_INTERNAL_H_
#define INCLUDE_UTRANS_INTERNAL_H_

#include <pthread.h>
#include <stdint.h>

#include <ucp/api/ucp.h>

#include "id_hash_map.h"
#include "list_head.h"
#include "utrans.h"
#include "utrans_mr_set.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // TODO: add cache update & remove
    id_hash_map_t* inst2info; // inst_id -> inst_info_t
    id_hash_map_t* addr2inst; // uint64(ip|port) -> inst_id

    pthread_mutex_t* lock_pool;
    uint32_t lock_num;
} peer_mgr_t;

typedef struct hosted_conns {
    int max_conns;
    volatile int updating;
    ucp_ep_h conn; // TODO: add more endpoints
} hosted_conns_t;

typedef struct {
    id_hash_map_t* id2ctx_map;
} conn_mgr_t;

typedef struct mem_region {
    void* addr;
    uint64_t len;
    int type; // ram, vram
} mem_region_t;

typedef struct mem_region_registed {
    mem_region_t mr;
    ucp_mem_h hdl;
    size_t rkey_size;
    void* rkey_cont;
} mem_region_registed_t;

typedef struct mem_region_registed_wire {
    mem_region_t mr;
    ucp_rkey_h rkey;
} mem_region_registed_wire_t;

typedef struct peer_mrs {
    int num_mrs;
    mem_region_registed_wire_t mrs_arr[0];
} peer_mrs_t;

typedef struct peer_mrs_mgr {
    int64_t last_update_time;
    peer_mrs_t* peer_mrs;

    pthread_mutex_t rpc_lock;
    pthread_rwlock_t mrs_lock;
    struct utrans_hash_map* peer_tags;
} peer_mrs_mgr_t;

typedef struct inst_info {
    host_info_t* host_info;
    peer_mrs_mgr_t mrs_mgr;
    hosted_conns_t* hosted_conns; // TODO: refine
} inst_info_t; // 远端实例信息

typedef struct {
    pthread_t thd;
    pthread_mutex_t lock;
    struct list_head in_queue;
    struct list_head work_queue;
    int num_in_queue;
    int num_work_queue;
    int monitor_epoll_fd;
    volatile int cleaner_exit;
} cleaner_ctx_t;

typedef struct ucx_ctx {
    struct utrans_ctx* putrz_ctx;
    ucp_context_h pucp_ctx;
    ucp_worker_h pucp_wrk;
    // ucp_ep_h
    ucp_listener_h conn_listener;

    // TODO: not initialized
    int num_pending_mrs;
    mr_set_t mr_set;
    pthread_rwlock_t mr_lock;
} ucx_ctx_t;

struct utrans_ctx {
    uint64_t inst_id; // current uniq inst id
    utrans_config_t* config;
    int listen_port;
    ucx_ctx_t ucx_ctx;
    peer_mgr_t peer_mgr;
};

typedef struct {
    uint64_t cli_node_id;
    uint64_t srv_node_id;
    uint64_t conn_id;
    int is_success;
} cli_finish_conn_info_t;

struct utrans_connect_config {
    char* server;
    int server_port;
};

struct utrans_req_info { // TODO: visible or not
    int64_t tm_start;
    int64_t tm_submit;
    int64_t tm_finish;
#if PERF_METRICS
    int64_t tm_route;
#endif
};

typedef struct wait_point {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int cond_waitting;
} wait_point_t;

typedef struct utrans_req_info_intl {
    utrans_req_info_t pub;
    struct list_head l_reuse;
    int st_in_reuse_queue;
    int st_reusable;
    wait_point_t* wait_point;
} utrans_req_info_intl_t;

utrans_req_info_intl_t* create_utrans_req_info_intl(int will_create_wait_point);
void release_utrans_req_info_intl(utrans_req_info_intl_t* info);

#ifdef __cplusplus
}
#endif
#endif