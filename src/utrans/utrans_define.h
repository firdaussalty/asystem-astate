#ifndef INCLUDE_UTRANS_DEFINE_H_
#define INCLUDE_UTRANS_DEFINE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UTRANS_PEER_MGR_LOCK_NUM (128)
#define UTRANS_INVALID_INST_ID (0xFFFFFFFFFFFFFFFFull)
#define UTRANS_DEF_MULTI_RAIL_MIN_SLICE_SIZE (2 * 1024 * 1024)
#define UTRANS_MAX_MESSAGE_SIZE ((uint32_t)2 << 30) // TODO: consider max_msg_sz
#define UTRANS_TRUE 1
#define UTRANS_FALSE 0

#ifndef UTRANS_PEER_REGINFO_TOLERANCE_MS
#    define UTRANS_PEER_REGINFO_TOLERANCE_MS (200)
#endif

typedef struct host_info {
    uint64_t inst_id;
    int8_t numa_aware;
    int8_t reserved[3];
    int device_num; // TODO: depart rdma dev info
    char ip[16]; // TODO: refine
    int port;
} host_info_t;

enum user_req_ops {
    USER_OP_INVALID,
    USER_OP_READ,
    USER_OP_WRITE,
    USER_OP_WRITE_IMM,
    USER_OP_SEND,
    USER_OP_SEND_IMM,
    USER_OP_RECV,
    USER_OP_GUTTER,
};

#pragma pack(push, 1)
typedef struct trans_buf {
    void* addr_beg;
    size_t trz_size;
} trans_buf_t;

struct mem_region_registed_wire;
typedef struct trans_req {
    uint64_t inst_id;
    int16_t opcode; // enum user_req_ops
    uint16_t num_seg; // please note external memory with (num_seg - 1) * sizeof(trans_buf) bytes is needed
    // remote buffer addr and related wire info
    void* rbuf;
    struct mem_region_registed_wire* rinfo;
    // local buffer segments
    struct trans_buf lbuf_seg[1];
} trans_req_t;

#pragma pack(pop)

enum user_req_state {
    USER_STATE_PENDING = 0,
    USER_STATE_RUNNING, // TODO: pending 和 running 两个状态差别不大，考虑删除掉running
    USER_STATE_SUBMITED,
    USER_STATE_FINISHED,
    USER_STATE_TIMEOUT,
    USER_STATE_INVALID, // req info == NULL case
};

enum user_req_exec_result {
    URES_SUCCESS = 0,
    URES_UNCERTAIN,
    URES_ERR_TIMEOUT,
    URES_ERR_INV_OPCODE,
    URES_ERR_INV_QP_STATE,
    URES_ERR_LKEY_NOT_FOUND,
    URES_ERR_INV_ARG,
    URES_ERR_SUBMIT_FAILED,
    URES_ERR_REF_CQ_RET,
    URES_ERR_COND_WAIT,
    URES_ERR_MR_NOT_FOUND,
    URES_ERR_PEER_INST_NOT_FOUND,
    URES_ERR_PEER_MR_NOT_FOUND,
    URES_ERR_PEER_NO_ROUTE,
    URES_ERR_NO_MEMORY,
    URES_ERR_FLOW_CONTROL,
    URES_ERR_NUMA_OPER,
};

typedef struct utrans_req_info utrans_req_info_t;


#ifdef __cplusplus
}
#endif

#endif // __UTRANS_DEFINE_H_INCLUDE__