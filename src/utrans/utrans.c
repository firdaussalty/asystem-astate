#include "utrans.h"

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <ucm/api/ucm.h>

#include "log.h"
#include "utils_helper.h"
#include "utrans_internal.h"

static int _proc_env(utrans_ctx_t* ctx) {
#define ON_ENV_VALUE(var, action) \
    do { \
        char* env_value = getenv(var); \
        if (env_value != NULL) { \
            action; \
            LOGI("[Environment] %s=%s\n", var, env_value); \
        } \
    } while (0)

    ON_ENV_VALUE("UTRANS_DISABLE_RDMA", ctx->config->rdma_disabled = 1);
    ON_ENV_VALUE("UTRANS_LOG_DIR",
                 strncpy(ctx->config->log_conf.log_dir, env_value, sizeof(ctx->config->log_conf.log_dir) - 1);
                 ctx->config->log_conf.log_dir[sizeof(ctx->config->log_conf.log_dir) - 1] = '\0');
    ON_ENV_VALUE("UTRANS_LOG_NAME",
                 strncpy(ctx->config->log_conf.log_name, env_value, sizeof(ctx->config->log_conf.log_name) - 1);
                 ctx->config->log_conf.log_name[sizeof(ctx->config->log_conf.log_name) - 1] = '\0');

#undef ON_ENV_VALUE
    return 0;
}

int _parse_config(utrans_ctx_t* ctx, utrans_config_t* config) {
    int ret = UTRANS_RET_UNKNOWN_ERR;
    utrans_config_t default_cfg = {.rpc_listen_port = 0,
                                   .log_conf = {.log_dir = ".",
                                                .log_name = "utrans",
                                                .log_max_line = LOG_DEFAULT_MAX_LINE,
                                                .log_max_file_count = LOG_DEFAULT_MAX_FILE_COUNT,
                                                .log_max_size = LOG_DEFAULT_MAX_SIZE,
                                                .self_delete = 1,
                                   },
                                   .rdma_disabled = 0,
                                   .rdma_conf = {.valid_dev_patt = NULL,
                                                 .num_pollers = 8,
                                                 .max_mr = 1024,
                                   }
    };
    if (!config) {
        memcpy(ctx->config, &default_cfg, sizeof(default_cfg));
        ret = UTRANS_RET_SUCC;
        goto end;
    }

#define NUM_ARG_CHECK(config_name, input, dst, valid_cond, def_cond, def_val, ret, invalid_action) \
    do { \
        if ((input)valid_cond) { \
            (dst) = (input); \
        } else if ((input)def_cond) { \
            (dst) = (def_val); \
        } else { \
            ret = UTRANS_RET_INVALID_ARGS; \
            LOGE("[Invalid Config] %s is invalid\n", config_name); \
            invalid_action; \
        } \
    } while (0)

#define UTRANS_NUM_ARG_CHECK(conf) \
    NUM_ARG_CHECK(#conf, config->conf, ctx->config->conf, > 0, == 0, default_cfg.conf, ret, goto end);

    UTRANS_NUM_ARG_CHECK(rpc_listen_port);
    if (strlen(config->log_conf.log_dir) != 0) {
        strncpy(ctx->config->log_conf.log_dir, config->log_conf.log_dir, sizeof(ctx->config->log_conf.log_dir));
    }
    if (strlen(config->log_conf.log_name) != 0) {
        strncpy(ctx->config->log_conf.log_name, config->log_conf.log_name, sizeof(ctx->config->log_conf.log_name));
    }
    UTRANS_NUM_ARG_CHECK(log_conf.log_max_line);
    UTRANS_NUM_ARG_CHECK(log_conf.log_max_file_count);
    UTRANS_NUM_ARG_CHECK(log_conf.log_max_size);
    ctx->config->log_conf.self_delete = 1;

    if (config->rdma_conf.valid_dev_patt) {
        ctx->config->rdma_conf.valid_dev_patt = (char*)malloc(strlen(config->rdma_conf.valid_dev_patt) + 1);
        if (!ctx->config->rdma_conf.valid_dev_patt) {
            ret = UTRANS_RET_NO_MEM;
            goto end;
        }
        strcpy(ctx->config->rdma_conf.valid_dev_patt, config->rdma_conf.valid_dev_patt);
    }

    UTRANS_NUM_ARG_CHECK(rdma_conf.max_mr);
    UTRANS_NUM_ARG_CHECK(rdma_conf.num_pollers);
    ctx->config->rdma_disabled = config->rdma_disabled;

    _proc_env(ctx);
    if (strlen(ctx->config->log_conf.log_dir) != 0 && strlen(ctx->config->log_conf.log_name) == 0) {
        strncpy(
            ctx->config->log_conf.log_name, default_cfg.log_conf.log_name, sizeof(ctx->config->log_conf.log_name) - 1);
        ctx->config->log_conf.log_name[sizeof(ctx->config->log_conf.log_name) - 1] = '\0';
    }
    if (strlen(ctx->config->log_conf.log_dir) == 0 && strlen(ctx->config->log_conf.log_name) != 0) {
        strncpy(ctx->config->log_conf.log_dir, default_cfg.log_conf.log_dir, sizeof(ctx->config->log_conf.log_dir) - 1);
        ctx->config->log_conf.log_dir[sizeof(ctx->config->log_conf.log_dir) - 1] = '\0';
    }
    ret = UTRANS_RET_SUCC;

#undef NUM_ARG_CHECK
#undef UTRANS_NUM_ARG_CHECK

end:
    if (IS_UTRANS_SUCC(ret)) {
        if (strlen(ctx->config->log_conf.log_dir) != 0 && strlen(ctx->config->log_conf.log_name) != 0) {
            struct log_config log_conf = {
                .log_max_line = ctx->config->log_conf.log_max_line,
                .log_max_file_count = ctx->config->log_conf.log_max_file_count,
                .log_max_size = ctx->config->log_conf.log_max_size,
                .self_delete = ctx->config->log_conf.self_delete,
            };
            utils_log_init(ctx->config->log_conf.log_dir, ctx->config->log_conf.log_name, &log_conf);
            // log_utrans_version_info();
            LOGI(
                "[CONF] log_name=%s/%s.log, log_max_line=%lu, log_max_file_count=%lu, log_max_size=%lu\n",
                ctx->config->log_conf.log_dir,
                ctx->config->log_conf.log_name,
                ctx->config->log_conf.log_max_line,
                ctx->config->log_conf.log_max_file_count,
                ctx->config->log_conf.log_max_size);
        } else {
            LOGI("[CONF] print log to console\n");
        }

        LOGI("[CONF] rpc_listen_port=%d\n", ctx->config->rpc_listen_port);
        LOGI("[CONF] rdma_disabled=%s\n", ctx->config->rdma_disabled ? "true" : "false");
        LOGI("[CONF] rdma conf:\n");
        LOGI(
            "[CONF]    valid_dev_patt=%s\n",
            ctx->config->rdma_conf.valid_dev_patt ? ctx->config->rdma_conf.valid_dev_patt : "null");
        LOGI("[CONF]    num_pollers=%d\n", ctx->config->rdma_conf.num_pollers);
        LOGI("[CONF]    max_mr=%d\n", ctx->config->rdma_conf.max_mr);
    }
    return ret;
}

int _ucp_worker_init(ucp_context_h ucp_ctx, ucp_worker_h* pucp_wrk, size_t inst_id) {
    ucp_worker_params_t worker_params = {0};
    ucs_status_t status;
    int ret = UTRANS_RET_SUCC;

    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    worker_params.client_id = inst_id;

    status = ucp_worker_create(ucp_ctx, &worker_params, pucp_wrk);
    if (status != UCS_OK) {
        LOGE("failed to ucp_worker_create (%s)\n", ucs_status_string(status));
        ret = UTRANS_RET_INTERNAL_ERR;
    }

    return ret;
}

void _ucx_ctx_destroy(ucp_context_h* pucp_ctx) {
    if (pucp_ctx && *pucp_ctx) {
        ucp_cleanup(*pucp_ctx);
        *pucp_ctx = NULL;
    }
}

int _ucx_ctx_init(ucx_ctx_t* pucx_ctx, rdma_config_t* rdma_conf) {
    int ret = UTRANS_RET_SUCC;
    ucs_status_t status;
    ucp_config_t* pucp_config;
    ucp_params_t ucp_params = {0};
    // rdma_ctx->conn_id_counter = 1;

    ucp_params.field_mask = UCP_PARAM_FIELD_NAME | UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS
        | UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;
    ucp_params.name = "utrans";
    ucp_params.features = UCP_FEATURE_RMA;
    ucp_params.estimated_num_eps = 8192;
    ucp_params.estimated_num_ppn = rdma_conf->num_pollers;

    status = ucp_config_read(NULL, NULL, &pucp_config);
    if (status != UCS_OK) {
        LOGE("failed to ucp_config_read (%s)\n", ucs_status_string(status));
        ret = UTRANS_RET_INTERNAL_ERR;
        goto end;
    }

    status = ucp_init(&ucp_params, pucp_config, &pucx_ctx->pucp_ctx);
    if (status != UCS_OK) {
        LOGE("failed to ucp_init (%s)\n", ucs_status_string(status));
        ret = UTRANS_RET_INTERNAL_ERR;
        goto end;
    }

    ret = _ucp_worker_init(pucx_ctx->pucp_ctx, &pucx_ctx->pucp_wrk, pucx_ctx->putrz_ctx->inst_id);
end:
    if (IS_UTRANS_FAIL(ret)) {
        _ucx_ctx_destroy(&pucx_ctx->pucp_ctx);
        pucx_ctx->pucp_ctx = NULL;
    }
    if (pucp_config) {
        ucp_config_release(pucp_config);
    }
    return ret;
}

int utrans_setup(utrans_config_t* putrz_conf, utrans_ctx_t** pputrz_ctx) {
    int ret = UTRANS_RET_SUCC;
    utrans_ctx_t* pctx;
    utrans_config_t* pconf;
    pctx = (utrans_ctx_t*)calloc(1, sizeof(utrans_ctx_t));
    pconf = (utrans_config_t*)calloc(1, sizeof(utrans_config_t));
    if (!pctx || !pconf) {
        ret = UTRANS_RET_NO_MEM;
        goto err;
    }
    pctx->config = pconf;
    if (IS_UTRANS_FAIL(ret = _parse_config(pctx, putrz_conf))) {
        goto err;
    }
    if (pctx->config->rdma_disabled) {
        LOGE("Only support rdma now");
        ret = UTRANS_RET_UNSUPPORT;
        goto err;
    }
    pctx->inst_id = get_inst_id();

    pctx->peer_mgr.lock_num = UTRANS_PEER_MGR_LOCK_NUM;
    pctx->peer_mgr.lock_pool = malloc(pctx->peer_mgr.lock_num * sizeof(pthread_mutex_t));
    if (!pctx->peer_mgr.lock_pool) {
        ret = UTRANS_RET_NO_MEM;
        goto err;
    }
    for (uint32_t i = 0; i < pctx->peer_mgr.lock_num; ++i) {
        if (0 != pthread_mutex_init(pctx->peer_mgr.lock_pool + i, NULL)) {
            for (uint32_t j = 0; j < i; ++j) {
                pthread_mutex_destroy(pctx->peer_mgr.lock_pool + j);
            }
            free(pctx->peer_mgr.lock_pool);
            pctx->peer_mgr.lock_pool = NULL;
            goto err;
        }
    }

    pctx->peer_mgr.inst2info = idhm_create(256, 32, NULL, NULL);
    if (!pctx->peer_mgr.inst2info) {
        ret = UTRANS_RET_INTERNAL_ERR;
        LOGE("can't create instinfo map\n");
        goto err;
    }
    pctx->peer_mgr.addr2inst = idhm_create(1024, 64, NULL, NULL);
    if (!pctx->peer_mgr.addr2inst) {
        ret = UTRANS_RET_INTERNAL_ERR;
        LOGE("create peer_mgr's addr2inst map failed\n");
        goto err;
    }
    pctx->ucx_ctx.putrz_ctx = pctx;

    if (IS_UTRANS_FAIL(ret = _ucx_ctx_init(&pctx->ucx_ctx, &pctx->config->rdma_conf))) {
        goto err;
    }

    if (pputrz_ctx) {
        *pputrz_ctx = pctx;
    }
    return ret;

err:
    if (pctx) {
        _ucx_ctx_destroy(&pctx->ucx_ctx.pucp_ctx);
        pctx->config = NULL;
        free(pctx);
        pctx = NULL;
    }
    if (pconf) {
        free(pconf);
        pconf = NULL;
    }
    utrans_clean(pctx); // TODO: add clean
    if (pputrz_ctx) {
        *pputrz_ctx = NULL;
    }
    return ret;
}

int utrans_clean(utrans_ctx_t* pucx_ctx) {
    return UTRANS_RET_SUCC; // TODO: add implement
}

/**
 * Server's application context to be used in the user's connection request
 * callback.
 * It holds the server's listener and the handle to an incoming connection request.
 */
typedef struct ucx_server_ctx {
    volatile ucp_conn_request_h conn_request;
    ucp_listener_h listener;
} ucx_server_ctx_t;


/**
 * The callback on the server side which is invoked upon receiving a connection
 * request from the client.
 */
void _ucx_srv_conn_handle_cb(ucp_conn_request_h conn_request, void* arg) {
    ucx_ctx_t* pucx_ctx = arg;
    ucp_conn_request_attr_t attr;
    char ip_str[INET_ADDRSTRLEN];
    ucs_status_t status;

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ID;

    status = ucp_conn_request_query(conn_request, &attr);
    if (status == UCS_OK) {
        LOGI(
            "Server received a connection request from client at address %s:%d, inst id %" PRIu64 "\n",
            sockaddr_get_ip_str(&attr.client_address, ip_str, sizeof(ip_str)),
            sockaddr_get_port_num(&attr.client_address),
            attr.client_id);
    } else if (status != UCS_ERR_UNSUPPORTED) {
        LOGE("failed to query the connection request (%s)\n", ucs_status_string(status));
    }

    // FIXME: NO good, shall process conn_request here
    ucp_listener_reject(pucx_ctx->conn_listener, conn_request);
}

static int _ucx_srv_listen(ucx_ctx_t* pucx_ctx, int listen_port) {
    int ret = UTRANS_RET_SUCC;
    struct sockaddr_storage listen_addr = {0};
    ucp_listener_params_t params;
    ucp_listener_attr_t attr;
    ucs_status_t status;

    struct sockaddr_in* sa_in = (struct sockaddr_in*)&listen_addr;
    sa_in->sin_addr.s_addr = INADDR_ANY;
    sa_in->sin_family = AF_INET;
    sa_in->sin_port = htons(listen_port);

    params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    params.sockaddr.addr = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen = sizeof(listen_addr);
    params.conn_handler.cb = _ucx_srv_conn_handle_cb;
    params.conn_handler.arg = pucx_ctx;

    // Create a listener on the server side to listen on the given address
    status = ucp_listener_create(pucx_ctx->pucp_wrk, &params, &pucx_ctx->conn_listener);
    if (status != UCS_OK) {
        LOGE("failed to listen on port %d ret (%s)\n", listen_port, ucs_status_string(status));
        ret = UTRANS_RET_RPC_SRV_ERR;
    } else {
        LOGI("server is listening on port %d\n", listen_port);
    }

    return ret;
}

int utrans_setup_rpcsrv(utrans_ctx_t* putrz_ctx) {
    if (putrz_ctx && putrz_ctx->ucx_ctx.pucp_wrk) {
        return _ucx_srv_listen(&putrz_ctx->ucx_ctx, putrz_ctx->listen_port);
    }
    return UTRANS_RET_INVALID_ARGS;
}

mem_region_registed_t* _regist_pub_buf(ucx_ctx_t* pucx_ctx, int mem_type, void* buf_addr, size_t buf_len) {
    pthread_rwlock_rdlock(&pucx_ctx->mr_lock);

    // first check whether already exist
    mem_region_registed_t* reged;
    if (buf_addr && (reged = utrans_mr_set_find(&pucx_ctx->mr_set, buf_addr, buf_len)) != NULL) {
        pthread_rwlock_unlock(&pucx_ctx->mr_lock);
        return reged;
    }

    const int WARN_MRS_NUM = pucx_ctx->putrz_ctx->config->rdma_conf.max_mr;
    // display how many mr is register-ing/ed
    int mr_num_reg = utrans_mr_set_getsize(&pucx_ctx->mr_set);
    int mr_num_fly = __atomic_fetch_add(&pucx_ctx->num_pending_mrs, 1, __ATOMIC_SEQ_CST);
    int current_mrs = mr_num_reg + mr_num_fly;
    if (current_mrs >= WARN_MRS_NUM) { // accept with warning
        LOGW(
            "accept new registered memory region, "
            "but pool size %d + pending %d already reach limit %d\n",
            mr_num_reg,
            mr_num_fly,
            WARN_MRS_NUM);
    }
    pthread_rwlock_unlock(&pucx_ctx->mr_lock);

    mem_region_registed_t* mrk;
    mrk = (mem_region_registed_t*)calloc(1, sizeof(mem_region_registed_t));
    if (!mrk) {
        LOGE("calloc memory for mem_region_registed_t failed\n");
        goto failed;
    }
    mrk->mr.addr = buf_addr;
    mrk->mr.len = buf_len;
    mrk->mr.type = mem_type;

    // second, register it to current worker
    ucs_status_t status;
    ucp_mem_map_params_t params = {0};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_PROT | UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH
        | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.memory_type = UCP_MEM_MAP_PROT_LOCAL_READ | UCP_MEM_MAP_PROT_LOCAL_WRITE | UCP_MEM_MAP_PROT_REMOTE_READ
        | UCP_MEM_MAP_PROT_REMOTE_WRITE;
    params.address = buf_addr;
    params.length = buf_len;
    params.memory_type = mem_type == MEM_TYPE_RAM ? UCS_MEMORY_TYPE_HOST : UCS_MEMORY_TYPE_CUDA;

    status = ucp_mem_map(pucx_ctx->pucp_ctx, &params, &mrk->hdl);
    if (status != UCS_OK) {
        LOGE(
            "ucp_mem_map register memory %s addr %p size %" PRIu64 " failed %s\n",
            utrans_get_mem_type_str(mem_type),
            buf_addr,
            buf_len,
            ucs_status_string(status));
        goto failed;
    }

    // generate rkey buffer to avoid redundant generate when sync
    status = ucp_rkey_pack(pucx_ctx->pucp_ctx, mrk->hdl, &mrk->rkey_cont, &mrk->rkey_size);
    if (status != UCS_OK) {
        LOGE("ucp_rkey_pack register memory rkey failed %s\n", ucs_status_string(status));
        goto failed;
    }

    // record to local mr_set
    pthread_rwlock_wrlock(&pucx_ctx->mr_lock);
    if (0 != utrans_mr_set_insert(&pucx_ctx->mr_set, mrk->mr.addr, mrk->mr.len, mrk)) {
        pthread_rwlock_unlock(&pucx_ctx->mr_lock);
        LOGE("record registered memory region failed, goto fail\n");
        goto failed;
    }
    __atomic_fetch_sub(&pucx_ctx->num_pending_mrs, 1, __ATOMIC_SEQ_CST);
    pthread_rwlock_unlock(&pucx_ctx->mr_lock);
    LOGI("Regist succ addr=%p len=%zu type=%s\n", buf_addr, buf_len, utrans_get_mem_type_str(mem_type));
    return mrk;

failed:
    LOGI("Regist fail addr=%p len=%zu type=%s\n", buf_addr, buf_len, utrans_get_mem_type_str(mem_type));
    __atomic_fetch_sub(&pucx_ctx->num_pending_mrs, 1, __ATOMIC_SEQ_CST);
    if (mrk && mrk->rkey_cont && mrk->rkey_size) {
        ucp_rkey_buffer_release(mrk->rkey_cont);
        mrk->rkey_cont = NULL;
        mrk->rkey_size = 0;
    }
    if (mrk && mrk->hdl) {
        ucp_mem_unmap(pucx_ctx->pucp_ctx, mrk->hdl);
        mrk->hdl = NULL;
    }
    if (mrk) {
        free(mrk);
    }
    return NULL;
}

const mem_region_registed_t* utrans_regist_ram(utrans_ctx_t* putrz_ctx, void* ram_addr, size_t ram_len, int numa_node) {
    if (!putrz_ctx || !ram_addr || !ram_len) {
        return NULL;
    }
    return _regist_pub_buf(&putrz_ctx->ucx_ctx, MEM_TYPE_RAM, ram_addr, ram_len);
}

const mem_region_registed_t* utrans_regist_vram(utrans_ctx_t* putrz_ctx, void* vram_addr, size_t vram_len, int gpu_id) {
    if (!putrz_ctx || !vram_addr || !vram_len) {
        return NULL;
    }
    int mem_gpu = utrans_get_vram_mem_type(gpu_id);
    if (mem_gpu == -1) {
        return NULL;
    }
    return _regist_pub_buf(&putrz_ctx->ucx_ctx, mem_gpu, vram_addr, vram_len);
}

static int _mr_reg_item_destroy(mem_region_registed_t* pmr_item, void* param) {
    ucx_ctx_t* pucx_ctx = (ucx_ctx_t*)param;
    if (!pmr_item) {
        return 0;
    }

    LOGI(
        "Deregist addr=%p len=%" PRIu64 " type=%s \n",
        pmr_item->mr.addr,
        pmr_item->mr.len,
        utrans_get_mem_type_str(pmr_item->mr.type));
    if (pmr_item->rkey_cont && pmr_item->rkey_size) {
        ucp_rkey_buffer_release(pmr_item->rkey_cont);
        pmr_item->rkey_cont = NULL;
        pmr_item->rkey_size = 0;
    }
    if (pmr_item->hdl) {
        ucp_mem_unmap(pucx_ctx->pucp_ctx, pmr_item->hdl);
        pmr_item->hdl = NULL;
    }
    free(pmr_item);
    return 0;
}

int utrans_dereg_mem(utrans_ctx_t* putrz_ctx, void* mem_addr, size_t mem_len) {
    if (!putrz_ctx || !mem_addr || !mem_len) {
        return UTRANS_RET_INVALID_ARGS;
    }

    ucx_ctx_t* pctx = &putrz_ctx->ucx_ctx;

    // use the write lock to prevent another thread visiting the mr which will be dereged
    pthread_rwlock_wrlock(&pctx->mr_lock);
    mem_region_registed_t* target_mr = utrans_mr_set_remove(&pctx->mr_set, mem_addr, mem_len);
    pthread_rwlock_unlock(&pctx->mr_lock);
    if (NULL == target_mr) {
        LOGW("Failed to find memory region at addr 0x%p len %zu\n", mem_addr, mem_len);
        goto end;
    }
    _mr_reg_item_destroy(target_mr, pctx);

end:
    return UTRANS_RET_SUCC;
}

uint64_t utrans_get_instid(utrans_ctx_t* putrz_ctx) {
    return putrz_ctx ? putrz_ctx->inst_id : UTRANS_INVALID_INST_ID;
}

utrans_config_t* utrans_get_conf(utrans_ctx_t* putrz_ctx) {
    return putrz_ctx ? putrz_ctx->config : NULL;
}

int utrans_query_instid(utrans_ctx_t* ctx, const char* addr_ipv4, int addr_port, uint64_t* peer_inst) {
    // TODO: add implementation
    return UTRANS_RET_SUCC;
}

int utrans_print_perf_info(utrans_ctx_t* pctx) {
    return UTRANS_RET_UNSUPPORT;
}

void utrans_unref_req_info(utrans_req_info_t* req) {
}

enum user_req_exec_result utrans_get_req_exec_result(utrans_req_info_t* req) {
    return URES_SUCCESS;
}

utrans_req_info_t* utrans_exec_transfer(utrans_ctx_t* ctx, trans_req_t* treq, trans_conf_t* pconf) {
    return NULL;
}