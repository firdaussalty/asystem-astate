#ifndef INCLUDE_UTILS_H_INCLUDE_
#define INCLUDE_UTILS_H_INCLUDE_
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/socket.h>

#include "log.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IN2(var, v1, v2) (((var) == (v1)) || ((var) == (v2)))
#define IN3(var, v1, v2, v3) (((var) == (v1)) || ((var) == (v2)) || ((var) == (v3)))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ON_ENV_VALUE(var, action) \
    do { \
        char* env_value = getenv(var); \
        if (env_value != NULL) { \
            action; \
            LOGI("[Environment] %s=%s\n", var, env_value); \
        } \
    } while (0)
#define BIT(nr) (1UL << (nr)) //set nr-th bit as 1

int set_nonblocking(int fd);

int add_fd(int epoll_fd, int new_fd, void* ptr);

int remove_fd(int epoll_fd, int rm_fd);

/**
 * @return int 1 if matchs or 0 if not
 */
int match_patterns(const char* str, const char* patterns);

static inline int align_int_up(int value, int align) {
    return ((value + align - 1) / align) * align;
}

static inline uint64_t align_uint64_up(uint64_t value, long align) {
    return ((value + align - 1) / align) * align;
}

static inline uint64_t align_page(uint64_t addr, uint64_t page_size) {
    return addr & ~(page_size - 1);
}

int ram_numa_node_detect_rough(const uint64_t addr, size_t len, int* pret_numa);

void sort_strs(char** strs, int num_strs);

typedef struct {
    int num_ipv4;
    int num_ipv6;
    char* host_name;
    char* ips[0];
} host_net_info_t;

int get_local_ips(host_net_info_t** pret);

unsigned int ipstr_to_uint(const char* ip_str);

uint64_t ipstr_to_uint64(const char* ip_str, int port);

void uint_to_ipstr(uint32_t ip_bin, char* ip_str);

int pack_ipv4_addr(struct sockaddr* pdst, const char* addr, int port);

uint64_t get_inst_id();

uint32_t get_ipbin_from_inst(uint64_t inst_id);

uint64_t get_pid_from_inst(uint64_t inst_id);

#define UTRANS_MAX_NUM_GPUS 64
enum mem_type_en {
    MEM_TYPE_RAM,
    MEM_TYPE_SSD, // TODO: not support yet
    MEM_TYPE_VRAM_GPU0 = 8, // TODO: more topology info about VRAM
    MEM_TYPE_VRAM_GPU_MAX = MEM_TYPE_VRAM_GPU0 + UTRANS_MAX_NUM_GPUS,
};

typedef struct mem_info {
    void* addr;
    size_t len;
    int type; // enum mem_type_en
    int index; // index of numa or gpu
} mem_info_t;

void* malloc_best_on_numa_node(int numa_node, size_t size, int* is_on_numa_node);

void free_best_on_numa_node(void* mem, size_t size, int is_on_numa_node);

void print_numa_info();

void* malloc_vram(int gpu_id, size_t len);

void free_vram(void* ptr);

const char* utrans_get_mem_type_str(int mem_type);

static inline int utrans_is_vram(int mem_type) {
    return (mem_type >= MEM_TYPE_VRAM_GPU0 && mem_type < MEM_TYPE_VRAM_GPU_MAX);
}

static inline int utrans_get_vram_gpu_id(int mem_type) {
    if (mem_type < MEM_TYPE_VRAM_GPU0 || mem_type >= MEM_TYPE_VRAM_GPU_MAX) {
        return -1;
    }
    return mem_type - MEM_TYPE_VRAM_GPU0;
}

static inline int utrans_get_vram_mem_type(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= MEM_TYPE_VRAM_GPU_MAX - MEM_TYPE_VRAM_GPU0) {
        return -1;
    }
    return MEM_TYPE_VRAM_GPU0 + gpu_id;
}

int get_num_gpus();

/**
 * @brief get location of the given memory
 * 
 * @param addr memory address
 * @param len memory length
 * @param ret_mem_infos pointer to returned memory info. User should call free() to release when it is no longer needed
 * @param ret_mem_num pointer to returned memory num
 */
int get_memory_location(void* addr, size_t len, mem_info_t** ret_mem_infos, int* ret_mem_num);

/**
 * @brief Malloc memory in expected type with specified length
 *
 * @param mem_type expected memory type
 * @param plen allocate memory size, used as both input and output
 * @param pnuma_node allocate memory numa id for RAM, used as both input and output.
 * @return memory pointer
 */
void* utrans_malloc(int mem_type, size_t* plen, int* pnuma_node);

/**
 * @brief Free memory allocated from utrans_malloc
 */
void utrans_free(int mem_type, void* addr, size_t len, int numa_node);

// Introduce rdtsc register for low latency time calculation
// refer to https://github.com/ZhongUncle/TSC_Timer.
// Supported on X86 & ARM
int is_tsc_available();

uint64_t get_cpu_freq();

static inline uint64_t rdtsc() {
#ifdef __x86_64__
    uint64_t low, high;
    asm volatile("rdtsc" : "=a"(low), "=d"(high));
    // TSC is a MSR, so it saves data to EDX:EAX
    return low | (high << 32);
#elif defined(__aarch64__)
    // TODO: enable pmu counter later if needed
    uint64_t tsc;
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));
    return tsc;
#else
    return 0;
#endif
}

static inline uint64_t get_time_ns(uint64_t cycles, uint64_t cpu_hz, int is_tsc) {
    return is_tsc ? (uint64_t)(cycles * (1000000000.0) / cpu_hz) : cycles;
}

static void atomic_int32_update_max(int32_t* pmax, int val) {
    int32_t curr;
    do {
        curr = __atomic_load_n(pmax, __ATOMIC_SEQ_CST);
    } while (val > curr && !__atomic_compare_exchange_n(pmax, &curr, val, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
}

static char* sockaddr_get_ip_str(const struct sockaddr_storage* sock_addr, char* ip_str, size_t max_size) {
    struct sockaddr_in addr_in;

    switch (sock_addr->ss_family) {
        case AF_INET:
            memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
            inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_size);
            return ip_str;
        default:
            return "Invalid address family";
    }
}

static int sockaddr_get_port_num(const struct sockaddr_storage* sock_addr) {
    struct sockaddr_in addr_in;

    switch (sock_addr->ss_family) {
        case AF_INET:
            memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
            return ntohs(addr_in.sin_port);
        default:
            return -1;
    }
}

#ifdef __cplusplus
}
#endif
#endif