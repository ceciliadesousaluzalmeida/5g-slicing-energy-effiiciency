GLOBAL_SEED = 42
MILP_TIME_LIMIT = 1800
MAX_MILP_SLICES = 10000
MAX_MILP_VNFS_TOTAL = 36000

PARAM_GRID = {
    "num_slices": [2, 4, 6, 8, 16, 32, 64],
    "num_vnfs_per_slice": [2, 3, 5, 4, 6],
    "seed": [1, 2, 3, 4, 5],
}

VNF_PROFILES = [
    {"cpu": 1, "throughput": 10, "latency": 20},
    {"cpu": 1, "throughput": 15, "latency": 30},
    {"cpu": 1, "throughput": 20, "latency": 45},

    {"cpu": 2, "throughput": 15, "latency": 40},
    {"cpu": 2, "throughput": 25, "latency": 60},
    {"cpu": 2, "throughput": 35, "latency": 80},

    {"cpu": 3, "throughput": 20, "latency": 70},
    {"cpu": 3, "throughput": 30, "latency": 90},
    {"cpu": 3, "throughput": 40, "latency": 110},

    {"cpu": 4, "throughput": 25, "latency": 90},
    {"cpu": 4, "throughput": 35, "latency": 120},
    {"cpu": 4, "throughput": 45, "latency": 140},

    {"cpu": 5, "throughput": 30, "latency": 110},
    {"cpu": 5, "throughput": 45, "latency": 140},
    {"cpu": 5, "throughput": 60, "latency": 170},

    {"cpu": 6, "throughput": 40, "latency": 130},
    {"cpu": 6, "throughput": 50, "latency": 160},
    {"cpu": 6, "throughput": 70, "latency": 200},

    {"cpu": 8, "throughput": 60, "latency": 180},
    {"cpu": 8, "throughput": 80, "latency": 220},
    {"cpu": 10, "throughput": 100, "latency": 250},
]