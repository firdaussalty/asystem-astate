#include "utils_helper.h"

#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <netdb.h>
#include <numa.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "log.h"
#include "utils.h"
#include "utrans_define.h"
#ifdef ENABLE_CUDA
#    include <cuda_runtime.h>
#endif

int set_nonblocking(int fd) {
    int flags, s;
    flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        LOGE("fcntl(F_GETFL): %s\n", strerror(errno));
        return -1;
    }
    flags |= O_NONBLOCK;
    s = fcntl(fd, F_SETFL, flags);
    if (s == -1) {
        LOGE("fcntl(F_SETFL): %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int add_fd(int epoll_fd, int new_fd, void* ptr) {
    if (set_nonblocking(new_fd) == -1) {
        LOGE("set_nonblocking: %s\n", strerror(errno));
        return -1;
    }

    struct epoll_event event;
    event.data.ptr = ptr;
    event.events = EPOLLIN | EPOLLET | EPOLLERR | EPOLLHUP | EPOLLRDHUP;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, new_fd, &event) == -1) {
        LOGE("epoll_ctl: add client: %s\n", strerror(errno));
        return -2;
    }
    return 0;
}

int remove_fd(int epoll_fd, int rm_fd) {
    if (epoll_ctl(epoll_fd, EPOLL_CTL_DEL, rm_fd, NULL) == -1) {
        LOGE("epoll_ctl: del client: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/**
 * @return int 1 if matchs or 0 if not
 */
int match_patterns(const char* str, const char* patterns) {
    char* pattern = (char*)alloca(strlen(patterns) + 1);
    const char* start = patterns;
    const char* end;
    int flags = 0;

    while ((end = strchr(start, ',')) != NULL) {
        size_t length = end - start;

        if (length > 0) {
            strncpy(pattern, start, length);
            pattern[length] = '\0';

            if (fnmatch(pattern, str, flags) == 0) {
                return 1;
            }
        }
        start = end + 1;
    }

    if (fnmatch(start, str, flags) == 0) {
        return 1;
    }

    return 0;
}

void sort_strs(char** strs, int num_strs) {
    int i, j;
    char* tmp;
    for (i = 0; i < num_strs - 1; i++) {
        for (j = i + 1; j < num_strs; j++) {
            if (strcmp(strs[i], strs[j]) > 0) {
                tmp = strs[i];
                strs[i] = strs[j];
                strs[j] = tmp;
            }
        }
    }
}

/**
 * Caller needs to free the struct returned
 *
 * @return num ips got, <0 if error
 */
int get_local_ips(host_net_info_t** pret) {
    char hostname[256], *pstrs;
    struct addrinfo hints, *res = NULL, *p;
    int status;
    int num_ipv4 = 0, num_ipv6 = 0;
    int buf_len;
    host_net_info_t* pi = NULL;
    int ret = 0;

    if (gethostname(hostname, sizeof(hostname)) == -1) {
        LOGE("gethostname");
        ret = -1;
        goto end;
    }

    LOGI("Hostname: %s\n", hostname);

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((status = getaddrinfo(hostname, NULL, &hints, &res)) != 0) {
        LOGE("getaddrinfo: %s\n", gai_strerror(status));
        ret = -2;
        goto end;
    }

    for (p = res; p != NULL; p = p->ai_next) {
        if (p->ai_family == AF_INET) {
            num_ipv4++;
        } else {
            num_ipv6++;
        }
    }

    buf_len = num_ipv4 * INET_ADDRSTRLEN + num_ipv6 * INET6_ADDRSTRLEN;
    buf_len += sizeof(host_net_info_t) + (num_ipv4 + num_ipv6) * sizeof(void*) + strlen(hostname) + 64;
    buf_len = align_int_up(buf_len, 64);
    pi = (host_net_info_t*)malloc(buf_len);
    if (!pi) {
        ret = -3;
        goto end;
    }
    pi->num_ipv4 = num_ipv4;
    pi->num_ipv6 = num_ipv6;

    int str_idx = 0;
    pstrs = (char*)pi + align_int_up(sizeof(host_net_info_t) + (num_ipv4 + num_ipv6) * sizeof(void*), 8);
    strcpy(pstrs, hostname);
    pi->host_name = pstrs;
    str_idx += strlen(pstrs) + 1;

    int idx_ipv4 = 0, idx_ipv6 = 0;
    for (p = res; p != NULL; p = p->ai_next) {
        void* addr;
        char* ipver;
        char* ipstr;

        if (p->ai_family == AF_INET) {
            struct sockaddr_in* ipv4 = (struct sockaddr_in*)p->ai_addr;
            addr = &(ipv4->sin_addr);
            ipver = "IPv4";
            ipstr = pstrs + str_idx;
            pi->ips[idx_ipv4++] = ipstr;
        } else {
            struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)p->ai_addr;
            addr = &(ipv6->sin6_addr);
            ipver = "IPv6";
            ipstr = pstrs + str_idx;
            pi->ips[num_ipv4 + idx_ipv6] = ipstr;
            idx_ipv6++;
        }

        inet_ntop(p->ai_family, addr, ipstr, INET6_ADDRSTRLEN);
        LOGI("%s hex: 0x%x\n", ipver, *((int*)addr));
        str_idx += strlen(ipstr) + 1;
        LOGI("%s: %s\n", ipver, ipstr);
    }
    sort_strs(pi->ips, num_ipv4);
    sort_strs(&pi->ips[pi->num_ipv4], num_ipv6);
    ret = num_ipv4 + num_ipv6;

end:
    if (pret) {
        *pret = pi;
    } else {
        free(pi);
    }
    if (res) {
        freeaddrinfo(res);
    }
    return ret;
}

unsigned int ipstr_to_uint(const char* ip_str) {
    struct sockaddr_in sa;
    inet_pton(AF_INET, ip_str, &(sa.sin_addr));
    return ntohl(sa.sin_addr.s_addr);
}

uint64_t ipstr_to_uint64(const char* ip_str, int port) {
    struct sockaddr_in sa;
    inet_pton(AF_INET, ip_str, &(sa.sin_addr));
    uint64_t packed = ntohl(sa.sin_addr.s_addr);
    return (packed << 32) | port;
}


void uint_to_ipstr(uint32_t ip_bin, char* ip_str) {
    struct in_addr addr;
    addr.s_addr = htonl(ip_bin);
    inet_ntop(AF_INET, &addr, ip_str, INET_ADDRSTRLEN);
}

uint64_t get_inst_id() {
    static int suffix = -1;
    if (suffix == -1) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        suffix = (int)(ts.tv_nsec & ((1 << 10) - 1));
    }
    uint64_t pid = (uint64_t)getpid();
    host_net_info_t* pips;
    get_local_ips(&pips);
    unsigned int ip1st = ipstr_to_uint(pips->ips[0]);
    free(pips);
    return (((uint64_t)ip1st) << 32) | ((pid & 0x3FFFFF) << 10) | suffix;
}

uint32_t get_ipbin_from_inst(uint64_t inst_id) {
    return (uint32_t)(inst_id >> 32);
}

uint64_t get_pid_from_inst(uint64_t inst_id) {
    return (uint64_t)((inst_id >> 10) & 0x3FFFFF);
}

void* malloc_best_on_numa_node(int numa_node, size_t size, int* is_on_numa_node) {
    void* pret;
    if (numa_node >= 0 && (pret = numa_alloc_onnode(size, numa_node)) != NULL) {
        *is_on_numa_node = 1;
    } else {
        *is_on_numa_node = 0;
        pret = lib2easy_malloc_align(size, 0);
    }
    return pret;
}

void free_best_on_numa_node(void* mem, size_t size, int is_on_numa_node) {
    if (is_on_numa_node) {
        numa_free(mem, size);
    } else {
        lib2easy_free_aligned(mem);
    }
}

void print_numa_info() {
    int max_node = numa_max_node(), i, cpu;
    printf("Max NUMA node: %d\n", max_node);

    for (i = 0; i <= max_node; ++i) {
        if (numa_bitmask_isbitset(numa_all_nodes_ptr, i)) {
            printf("NUMA node %d is present. ", i);

            struct bitmask* bm = numa_allocate_cpumask();
            if (numa_node_to_cpus(i, bm) == 0) {
                printf("CPUs: ");
                for (cpu = 0; cpu <= numa_num_possible_cpus(); ++cpu) {
                    if (numa_bitmask_isbitset(bm, cpu)) {
                        printf("%d ", cpu);
                    }
                }
            }
            numa_free_cpumask(bm);
            printf("\n");
        }
    }
}

int get_num_gpus() {
#ifdef ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        LOGE("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return device_count;
#else
    return 0;
#endif
}

int get_memory_location(void* addr, size_t len, mem_info_t** ret_mem_infos, int* ret_mem_num) {
    int i;
    mem_info_t* mem_infos = NULL;

#ifdef ENABLE_CUDA
    struct cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, addr);
    if (result != cudaSuccess) {
        LOGE("cudaPointerGetAttributes failed\n");
        return -1;
    }

    if (attributes.type == cudaMemoryTypeDevice) {
        mem_infos = (mem_info_t*)malloc(sizeof(mem_info_t));
        if (!mem_infos) {
            LOGE("failed to malloc for mem info\n");
            return -2;
        }
        mem_infos->addr = addr;
        mem_infos->len = len;
        mem_infos->type = utrans_get_vram_mem_type(attributes.device);
        mem_infos->index = attributes.device;
        *ret_mem_num = 1;
        *ret_mem_infos = mem_infos;
        return 0;
    }
    // 认为 cudaMemoryTypeUnregistered/cudaMemoryTypeHost 为CPU内存
    // 因为 cudaMemoryTypeManaged 不一定在显存中，所以也认为是CPU内存获取其numa信息
#endif // ENABLE_CUDA

    long PAGE_SIZE = sysconf(_SC_PAGESIZE);
    // 需要按页对齐
    uint64_t aligned_addr = align_page((uint64_t)addr, PAGE_SIZE);
    int page_num = ((uint64_t)addr - aligned_addr + len + PAGE_SIZE - 1) / PAGE_SIZE;
    void** pages = (void**)malloc(page_num * sizeof(void*));
    int* status = (int*)malloc(page_num * sizeof(int));

    for (i = 0; i < page_num; i++) {
        pages[i] = (void*)((char*)aligned_addr + i * PAGE_SIZE);
        ((char*)pages[i])[0] = ((char*)pages[i])[0]; // avoid lazy allocation
    }

    int ret = numa_move_pages(0, page_num, pages, NULL, status, 0);
    if (ret != 0) {
        LOGE("numa_move_pages failed");
        return -3;
    }

    int numa_node = status[0], mem_num = 1;
    for (i = 1; i < page_num; ++i) {
        if (numa_node != status[i]) { // 跨numa节点
            numa_node = status[i];
            ++mem_num;
        }
    }
    mem_infos = (mem_info_t*)malloc(mem_num * sizeof(mem_info_t));
    if (!mem_infos) {
        LOGE("failed to malloc for mem info\n");
        return -4;
    }
    numa_node = status[0];
    mem_num = 0;
    mem_info_t* cur_mem_info = NULL;
    uint64_t mem_start_addr = (uint64_t)addr, mem_len = PAGE_SIZE - ((uint64_t)addr - aligned_addr), cur_page_size;
    for (i = 1; i < page_num; ++i) {
        if (pages[i] + PAGE_SIZE >= addr + len) { // last page
            cur_page_size = (uint64_t)addr + len - (uint64_t)pages[i];
        } else {
            cur_page_size = PAGE_SIZE;
        }
        if (numa_node != status[i]) {
            cur_mem_info = &mem_infos[mem_num++];
            cur_mem_info->addr = (void*)mem_start_addr;
            cur_mem_info->len = mem_len;
            cur_mem_info->type = MEM_TYPE_RAM;
            cur_mem_info->index = numa_node;
            mem_start_addr = (uint64_t)pages[i];
            mem_len = cur_page_size;
            numa_node = status[i];
        } else {
            mem_len += cur_page_size;
        }
    }
    cur_mem_info = &mem_infos[mem_num++];
    cur_mem_info->addr = (void*)mem_start_addr;
    cur_mem_info->len = mem_len;
    cur_mem_info->type = MEM_TYPE_RAM;
    cur_mem_info->index = numa_node;

    *ret_mem_num = mem_num;
    *ret_mem_infos = mem_infos;
    free(pages);
    free(status);
    return 0;
}

int ram_numa_node_detect_rough(const uint64_t addr, size_t len, int* pret_numa) {
    // check whether numa supported
    if (!len || !addr || !pret_numa) {
        return URES_ERR_INV_ARG;
    }
    *pret_numa = -1;
    if (0 != numa_available()) {
        return URES_SUCCESS;
    }

    int ret;
    const long PAGE_SIZE = sysconf(_SC_PAGESIZE);
    const uint64_t addr_aligned = align_page((uint64_t)addr, PAGE_SIZE);
    const uint32_t page_num = ((uint64_t)addr - addr_aligned + len + PAGE_SIZE - 1) / PAGE_SIZE;

    void** pages = (void**)malloc(page_num * sizeof(void*));
    int* status = (int*)malloc(page_num * sizeof(int));

    if (!pages || !status) {
        ret = URES_ERR_NO_MEMORY;
        goto end;
    }

    // peek all pages
    for (int i = 0; i < page_num; i++) {
        pages[i] = (void*)(addr_aligned + i * PAGE_SIZE);
        ((char*)pages[i])[0] = ((char*)pages[i])[0]; // avoid lazy allocation
    }

    // peek numa status
    if (numa_move_pages(0, page_num, pages, NULL, status, 0) != 0) {
        LOGE("numa_move_pages get current status failed");
        ret = URES_ERR_NUMA_OPER;
        goto end;
    }

    // accumulate page numbers on each numa node
    uint32_t* pnuma_cntr = alloca((numa_max_node() + 1) * sizeof(uint32_t));
    memset(pnuma_cntr, 0, (numa_max_node() + 1) * sizeof(uint32_t));
    for (uint32_t pg = 0; pg < page_num; ++pg) {
        if (status[pg] < 0) {
            ret = URES_ERR_NUMA_OPER;
            goto end;
        }

        pnuma_cntr[status[pg]]++;
    }

    // find out numa node with maximum page number
    int numa_node = 0;
    uint32_t max_pgnum = pnuma_cntr[numa_node];
    for (int nn = 1; nn <= numa_max_node(); ++nn) {
        if (pnuma_cntr[nn] > max_pgnum) {
            max_pgnum = pnuma_cntr[nn];
            numa_node = nn;
        }
    }
    ret = URES_SUCCESS;
    // treat numa node with maximum pages as input buffer's numa node
    *pret_numa = numa_node;

end:
    if (pages) {
        free(pages);
    }
    if (status) {
        free(status);
    }
    return ret;
}

void* malloc_vram(int gpu_id, size_t len) {
#ifdef ENABLE_CUDA
    void* ptr = NULL;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        LOGE("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    if (gpu_id >= device_count) {
        LOGE("Invalid gpu_id=%d num_gpus=%d\n", gpu_id, device_count);
        return NULL;
    }
    err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        LOGE("cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    err = cudaMalloc(&ptr, len);
    if (err != cudaSuccess) {
        LOGE("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
#else
    LOGE("CUDA is not enabled, can't allocate VRAM\n");
    return NULL;
#endif
}

void free_vram(void* ptr) {
#ifdef ENABLE_CUDA
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            LOGE("cudaFree failed: %s\n", cudaGetErrorString(err));
        }
    }
#else
    LOGE("CUDA is not enabled, can't free VRAM\n");
#endif
}

const char* utrans_get_mem_type_str(int mem_type) {
    int id;
    static char strs[(MEM_TYPE_VRAM_GPU_MAX + 8) * 8] = {0};
    switch (mem_type) {
        case MEM_TYPE_RAM:
            return "RAM";
        case MEM_TYPE_SSD:
            return "SSD";
        default:
            id = utrans_get_vram_gpu_id(mem_type);
            if (id >= 0) {
                char* pret = strs + id * 8;
                if (*((long*)pret) == 0) {
                    snprintf(pret, 8, "GPU%d", id);
                }
                return pret;
            }
            return "UNKNOWN";
    }
}

void* utrans_malloc(int mem_type, size_t* plen, int* pnuma_node) {
    void* mem_ptr = NULL;
    int on_numa = 0;
    size_t len = align_uint64_up(*plen, sysconf(_SC_PAGESIZE));
    if (mem_type == MEM_TYPE_RAM) {
        mem_ptr = malloc_best_on_numa_node(*pnuma_node, len, &on_numa);
        if (!on_numa) {
            *pnuma_node = -1;
        }
    } else if (mem_type >= MEM_TYPE_VRAM_GPU0 && mem_type < MEM_TYPE_VRAM_GPU_MAX) {
        mem_ptr = malloc_vram(utrans_get_vram_gpu_id(mem_type), len);
    }
    if (mem_ptr) {
        *plen = len;
    }
    return mem_ptr;
}

void utrans_free(int mem_type, void* addr, size_t len, int numa_node) {
    len = align_uint64_up(len, sysconf(_SC_PAGESIZE));
    if (addr) {
        if (mem_type == MEM_TYPE_RAM) {
            free_best_on_numa_node(addr, len, numa_node >= 0);
        } else if (mem_type >= MEM_TYPE_VRAM_GPU0 && mem_type < MEM_TYPE_VRAM_GPU_MAX) {
            free_vram(addr);
        }
    }
}

int pack_ipv4_addr(struct sockaddr* pdst, const char* addr, int port) {
    struct sockaddr_in* packed = (struct sockaddr_in*)pdst;
    memset(packed, 0, sizeof(struct sockaddr_in));
    packed->sin_family = AF_INET;
    packed->sin_port = port;
    if (inet_pton(AF_INET, addr, &packed->sin_addr) == 1) {
        return sizeof(struct sockaddr_in);
    }
    return -1;
}

uint64_t get_cpu_freq() {
#ifdef __x86_64__
    // to filter other things like `CPU max MHz` and `CPU min MHz`...
    FILE* fd = popen("lscpu | grep 'CPU MHz' | awk {'print $3'}", "r");
    if (!fd)
        return 0;

    char cpu_mhz_str[64] = {0};
    fgets(cpu_mhz_str, 64, fd);
    fclose(fd);

    double mhz = atof(cpu_mhz_str);
    uint64_t freq = (uint64_t)(mhz * 1000 * 1000);

    return freq;
#elif defined(__aarch64__)
    // Notice that aarch64 == arm-v8, so we can safely invoke the asm
    uint64_t cntfrq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
    return cntfrq;
#else
    return 0;
#endif
}

int is_tsc_available() {
#ifdef __x86_64__
    unsigned long a = 0x1, b = 0, c = 0, d = 0;
    asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "b"(b), "c"(c), "d"(d));
    if ((d & BIT(4))) {
        // TSC exist!
        a = 0x80000007;
        asm volatile("cpuid\n\t" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "b"(b), "c"(c), "d"(d));
        if ((d & BIT(8))) {
            // we want a constant_tsc flag to provide a accurate counter
            // Invariant TSC available!
            return 1;
        }
    } else {
        // TSC not exist
        return 0;
    }
    return 0;
#elif defined(__aarch64__)
    // Notice that aarch64 == arm-v8, so we can safely invoke the asm
    uint64_t cntfrq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
    return cntfrq != 0;
#else
    // unsupported platform, using posix clock_gettime
    return 0;
#endif
}