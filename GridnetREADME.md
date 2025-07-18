# GridNet 5G Slicing Simulation – Scenario Description

This simulation aims to model and evaluate energy-aware orchestration of 5G network slices under SLA constraints and varying user density. It is based on a simplified GridNet topology inspired by Zahraa El Attar's thesis, and supports multiple service types and virtualized network function (VNF) chains.

---

## 🧩 Slice Profiles and Service Classes

Three representative 5G service classes are implemented, inspired by 3GPP specifications:

| Slice Type | Description | Latency (ms) | Throughput (Mbps) | Users Range |
|------------|-------------|--------------|--------------------|-------------|
| **eMBB**   | Enhanced Mobile Broadband – High data traffic (e.g. video, AR/VR) | ≤ 50         | ≥ 100              | 20–100      |
| **URLLC**  | Ultra-Reliable Low Latency Communications (e.g. autonomous driving) | ≤ 10         | ≥ 10               | 5–20        |
| **mMTC**   | Massive Machine-Type Communications (e.g. sensors, IoT)            | ≤ 500        | ~0.1               | 100–300     |

Each slice instance is generated with:
- A number of users drawn from its respective range;
- A service function chain (VNF list);
- SLA constraints: max latency and min throughput.

---

## ⚙️ Virtual Network Functions (VNFs)

The following VNFs are used across slices:

| Function | Used By | Role |
|----------|---------|------|
| **RAN** (CU/DU) | All slices | Handles access layer, connects users |
| **UPF**         | eMBB, URLLC | Handles user data forwarding and QoS |
| **DN**          | eMBB        | Models the external data network |
| **IoT GW**      | mMTC        | Aggregates IoT traffic before processing |

Each VNF is instantiated per slice and mapped to a node. The number and type of VNFs vary per service class.

---

## 🔌 Energy Consumption Model

VNFs and infrastructure nodes consume energy based on usage:

### Per-VNF Consumption:

| VNF       | Base Power (W) | Power per User (W/user) |
|-----------|----------------|-------------------------|
| RAN       | 8              | 0.05                    |
| UPF       | 6              | 0.10                    |
| DN        | 4              | 0.02                    |
| IoT GW    | 3              | 0.01                    |

Total VNF energy = `base + users × per_user`.

### Node Energy:

- Each **active node** (hosting ≥1 VNF) consumes **100 W**, regardless of traffic.

---

## 🧠 Latency Estimation

For each slice, end-to-end latency is computed as:

```text
latency = hops × 5ms
