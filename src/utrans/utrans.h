#ifndef INCLUDE_UTRANS_H_
#define INCLUDE_UTRANS_H_

#include <linux/limits.h>
#include <sys/types.h>

#include "utrans_define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct utrans_ctx utrans_ctx_t;
typedef struct utrans_connect_config utrans_connect_config_t;

typedef struct rdma_dev_ctx rdma_dev_ctx_t;
typedef struct req_info req_info_t;

enum utrans_ret_code {
    UTRANS_RET_SUCC = 0,
    UTRANS_RET_UNKNOWN_ERR = -1,
    UTRANS_RET_INTERNAL_ERR = -2, // TODO: more detail
    UTRANS_RET_INVALID_ARGS = -3,
    UTRANS_RET_NO_MEM = -4,
    UTRANS_RET_UNSUPPORT = -5,
    UTRANS_RET_NO_DEVICES = -6,
    UTRANS_RET_DEVICE_ERR = -7, // TODO: refine
    UTRANS_RET_EXCHG_INFO_ERR = -8,
    UTRANS_RET_BRING_UP_QP_ERR = -9,
    UTRANS_RET_CONFIRM_INFO_ERR = -10,
    UTRANS_RET_CREATE_RPC_ERR = -11,
    UTRANS_RET_PROTOCOL_ERR = -12,
    UTRANS_RET_GET_HOST_INFO_ERR = -13,
    UTRANS_RET_CONN_REFUSED = -14,
    UTRANS_RET_RPC_SRV_ERR = -15,
    UTRANS_RET_POSIX_PTHREAD = -16,
    // TODO: refine
};

#define IS_UTRANS_SUCC(ret_code) ((ret_code) == UTRANS_RET_SUCC)
#define IS_UTRANS_FAIL(ret_code) (!IS_UTRANS_SUCC(ret_code))

typedef struct utrans_log_config {
    char log_dir[PATH_MAX];
    char log_name[NAME_MAX];
    unsigned long log_max_line;
    unsigned long log_max_file_count;
    unsigned long log_max_size;
    int self_delete;
} utrans_log_config_t;

typedef struct {
    char* valid_dev_patt;
    int num_pollers;
    int max_mr;
} rdma_config_t;

// TODO: hide
typedef struct utrans_config {
    int rpc_listen_port;
    char rdma_disabled;
    char reserved[3];
    utrans_log_config_t log_conf;
    rdma_config_t rdma_conf;
} utrans_config_t;

typedef struct {
    int dev_id;
    int port;
} dev_port_t;

typedef struct mem_region_registed mem_region_registed_t;

struct utrans_hash_map;

/** vv
 * @brief Setup utrans context. <utrans_setup> should be called before doing anything and only once
 * 
 * @param config
 * @param pret
 * @return utrans_ret_code 
 */
int utrans_setup(utrans_config_t* config, utrans_ctx_t** pret);

/** vv
 * @brief Clean utrans context. <utrans_setup> should be called after all tasks have been completed and can only be called once
 * 
 * @param ctx
 * @return utrans_ret_code
 */
int utrans_clean(utrans_ctx_t* ctx);

/** vv
 * @brief Setup Tcp Listener to handle Simple RPC
 */
int utrans_setup_rpcsrv(utrans_ctx_t* ctx);

/** vv
 * @brief
 * 
 * @param ctx 
 * @param addr start address of the memory region, auto malloced if NULL
 * @param len length of the memory region
 * @param numa_node -1 if not care, else valid numa node id
 * @return NULL if failed
 */
const mem_region_registed_t* utrans_regist_ram(utrans_ctx_t* ctx, void* addr, size_t len, int numa_node);

/** vv
 * @brief
 * 
 * @param ctx 
 * @param addr start address of the memory region, auto malloced if NULL
 * @param len length of the memory region
 * @param gpu_id 
 * @return NULL if failed
 */
const mem_region_registed_t* utrans_regist_vram(utrans_ctx_t* ctx, void* addr, size_t len, int gpu_id);

/** vv
 * @brief dereg all memory regions contained the given addr and length
 * 
 * @param ctx
 * @param addr start address of the memory
 * @param len length of the memory
 * @return 0 if success
 * @return 0 if success
 */
int utrans_dereg_mem(utrans_ctx_t* ctx, void* addr, size_t len);

typedef struct trans_conf {
    int max_num_slices; // max number of slices for multi-rail transfer, 0 means let utrans decide (disable multi-rail currently)
    uint32_t min_slice_size; // minimum slice size for multi-rail transfer, 0 means let utrans decide
    long wait_ms; // waitting in ms for the transfer to finish, < 0 means wait forever, 0 for asynchronous transfer
} trans_conf_t;

/** vv
 * @brief transfer data to the peer instance without specifying the connection to use.
 * 
 * @param ctx 
 * @param treq 
 * @param pconf transfer configuration, can not be NULL
 * @return utrans_req_info_t* NULL if param error
 */
utrans_req_info_t* utrans_exec_transfer(utrans_ctx_t* ctx, trans_req_t* treq, trans_conf_t* pconf);

/** vv
 * @brief Query request execution result: success or error info.
 *        Note: valid only when the request in finished or time-out state
 * 
 */
enum user_req_exec_result utrans_get_req_exec_result(utrans_req_info_t* req);

/** vv
 * @brief Release the reference utrans_req_info_t
 * 
 * @param req 
 */
void utrans_unref_req_info(utrans_req_info_t* req);

/** vv
 * @brief Get self utrans instance id
 *
 * @return self instance id if valid, otherwise INVALID_INST_ID
 */
uint64_t utrans_get_instid(utrans_ctx_t* ctx);

/** vv
 */
utrans_config_t* utrans_get_conf(utrans_ctx_t* ctx);

/** vv
 * @brief Get remote node instance id
 *
 * @param ctx utrans context
 * @param addr_ipv4 remote node's ipv4 address, shall not empty or null
 * @param addr_port remote node's listening port
 * @param peer_inst use INVALID_INST_ID if unknown, set to incorrect instance id which
 *   used to do data transfer with, will set to remote instance id if valid, otherwise
 *   INVALID_INST_ID
 * @return UTRANS_RET_SUCC when success
 */
int utrans_query_instid(utrans_ctx_t* ctx, const char* addr_ipv4, int addr_port, uint64_t* peer_inst);

/** vv
 * @brief Print current brief perf info to utrans log since last calling,
 *   only works when PERF_METRIC is enabled when compiling utrans lib,
 *   otherwise will do nothing and returns unsppoted
 * @note Hurts performance when system are busy or exists many peer instance
 * @return 0 for success
 */
int utrans_print_perf_info(utrans_ctx_t* pctx);

#ifdef __cplusplus
}
#endif
#endif // __UTRANS_H_INCLUDE__