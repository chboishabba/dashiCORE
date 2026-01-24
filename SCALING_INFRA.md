Yes — and the “optimal” topology depends on whether you’re trying to optimize **(a) kernel semantics parity**, **(b) single-job throughput**, or **(c) scale-out training**. The nice thing is frontier hardware has converged on a very clear pattern:

* **Scale-up (intra-node)**: NVLink/NVSwitch domains (8 → 72 GPUs)
* **Scale-out (inter-node)**: InfiniBand (NDR 400 today; faster lanes emerging) or AI-tuned Ethernet fabrics (Spectrum-X)

Below is a deployment topology playbook that lines up with today’s frontier GPU/network stack.

---

## 1) The topology rule of thumb for dashiCORE

### If your algorithm needs frequent global sync/reductions

Prefer **scale-up** (bigger NVLink domain) over scale-out.

* NVLink/NVSwitch is built exactly for this (GPU-to-GPU bandwidth scales massively inside a “domain”). NVIDIA’s NVLink/NVSwitch specs and NVL72 aggregate bandwidth are explicit about the huge intra-domain fabric bandwidth. ([NVIDIA][1])

### If your algorithm is mostly local + occasional exchanges

Scale-out with **InfiniBand NDR 400** or **Spectrum-X Ethernet** is fine.

* NVIDIA Quantum-2 InfiniBand is positioned for 400Gb/s throughput, and the QM9700 class is 64 ports of 400Gb/s in 1U (51.2 Tb/s bidirectional aggregate). ([NVIDIA][2])
* Spectrum-X is marketed as an AI networking platform for Ethernet scale-out. ([NVIDIA][3])

For dashiCORE specifically (with deterministic/parity constraints), the key is: **avoid unordered reductions** and **keep “hard” global aggregations inside the most deterministic domain** (often a single node).

---

## 2) Practical deployment tiers (what I’d actually build)

### Tier A — “Semantic Truth” node (dev + CI truth)

**Goal:** prove invariants, parity, determinism.

* 1 node, 1–8 GPUs
* Prefer a tightly-coupled NVLink box if you’re doing multi-GPU

  * Example class: HGX / DGX-style 8-GPU node (or equivalent)
* Good candidates depending on budget/availability:

  * NVIDIA H200 (141GB HBM3e, 4.8 TB/s bandwidth) ([NVIDIA][4])
  * AMD Instinct MI325X (256GB HBM3E, ~6 TB/s stated peak) ([AMD][5])

**Why this tier matters:** it becomes your “CPU reference + GPU adapter parity gate” box. If Tier A can’t run bit-exact parity tests, *nothing else matters*.

---

### Tier B — “Scale-up pod” (fastest path to real throughput without fabric pain)

**Goal:** run larger experiments while keeping determinism manageable.

* 2–16 nodes, each 8 GPUs (or similar)
* Interconnect: InfiniBand NDR 400 (Quantum-2 + ConnectX) ([NVIDIA][2])
* Topology: fat-tree / folded-Clos (standard for IB/Ethernet pods)

**Why:** you get enough scale to stress modularity, but you can still enforce deterministic reductions by:

* doing reductions in a fixed tree order
* or reducing per-node and aggregating on CPU / a designated coordinator

---

### Tier C — “Frontier rack-scale” (when your workload is dominated by communication)

**Goal:** maximum scale-up (best for sync-heavy kernels).

* **GB200 NVL72** class systems: 72 GPUs tied together via NVLink/NVSwitch in one rack-scale domain ([NVIDIA][6])
* This is the “don’t make the network your bottleneck” option: you keep most comm inside the NVLink domain.

When you’re ready to be truly bleeding edge, NVIDIA has also publicly discussed a next platform (“Rubin NVL72”) with NVLink 6 and big rack-scale claims, but treat this as roadmap/announcement timing, not guaranteed deploy-now. ([Tom's Hardware][7])

---

## 3) Network fabric choices (how to decide quickly)

### InfiniBand (Quantum-2 / NDR 400)

Pick this if you want:

* lowest latency + tight collectives
* mature HPC/AI tuning
* predictable behavior for scale-out

Quantum-2 is explicitly framed around 400Gb/s throughput, and QM9700 class switches are designed for dense 400G ports. ([NVIDIA][2])

### Ethernet (Spectrum-X)

Pick this if you want:

* easier integration into “cloud-ish” environments
* scale-across / multi-site networking positioning (as NVIDIA is marketing it) ([NVIDIA Investor Relations][8])

For dashiCORE: Ethernet can be fine, but you’ll need to be extra strict about determinism around reductions/collectives.

---

## 4) A topology that specifically matches dashiCORE’s contracts

Given your **concurrency/determinism contract**, the safest deployment topology is:

1. **Single “truth” node** (CPU reference + 1 GPU backend)
2. **One scale-up node class** (8 GPUs NVLink) used for performance and parity
3. **Optional scale-out pod** only after you have:

   * deterministic reduction strategy
   * fixed-order collective plan
   * “nondeterminism detectors” in tests

This mirrors how frontier systems are built: maximize **intra-domain bandwidth** first (NVLink domains), then scale-out with IB/Ethernet as needed. ([NVIDIA][1])

---

## 5) Concrete “next sprint” deliverable for topology

If you want to make this actionable inside your repo, define a `docs/deployment_topology.md` with:

* **Supported topologies**: single-node / pod / NVL72-class rack
* **Backend tiers**: truth / throughput / scale-out
* **Determinism requirements** per tier (especially reductions)
* A minimal “cluster contract”:

  * how you pin thread/workgroup counts
  * how you order reductions
  * what metrics/logs must match

---

* [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026?utm_source=chatgpt.com)
* [Tom's Hardware](https://www.tomshardware.com/networking/nvidia-outlines-plans-for-using-light-for-communication-between-ai-gpus-by-2026-silicon-photonics-and-co-packaged-optics-may-become-mandatory-for-next-gen-ai-data-centers?utm_source=chatgpt.com)

[1]: https://www.nvidia.com/en-au/data-center/nvlink/?utm_source=chatgpt.com "NVLink & NVSwitch: Fastest HPC Data Center Platform"
[2]: https://www.nvidia.com/en-au/networking/infiniband-switching/?utm_source=chatgpt.com "NVIDIA Quantum InfiniBand Switches"
[3]: https://www.nvidia.com/en-au/networking/spectrumx/?utm_source=chatgpt.com "Spectrum-X | Ethernet Networking Platform for AI"
[4]: https://www.nvidia.com/en-au/data-center/h200/?utm_source=chatgpt.com "NVIDIA H200 Tensor Core GPU"
[5]: https://www.amd.com/en/products/accelerators/instinct/mi300.html?utm_source=chatgpt.com "AMD Instinct™ MI300 Series Accelerators"
[6]: https://www.nvidia.com/en-au/data-center/gb200-nvl72/?utm_source=chatgpt.com "GB200 NVL72 | NVIDIA"
[7]: https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026?utm_source=chatgpt.com "Nvidia launches Vera Rubin NVL72 AI supercomputer at CES - promises up to 5x greater inference performance and 10x lower cost per token than Blackwell, coming 2H 2026"
[8]: https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Spectrum-X-Ethernet-Switches-Speed-Up-Networks-for-Meta-and-Oracle/default.aspx?utm_source=chatgpt.com "NVIDIA Spectrum-X Ethernet Switches Speed Up Networks ..."




# dashiCORE — Deployment Topology & Cluster Contract

> **Status:** Normative  
> **Scope:** Hardware topology, backend tiers, and determinism guarantees  
> **Principle:** Scale must not weaken semantics.

---

## 0. Purpose

This document defines the **supported deployment topologies**, **backend tiers**, and
the **cluster-level determinism contract** for dashiCORE.

It answers three questions:

1. Where may dashiCORE run?
2. What guarantees must each deployment tier provide?
3. How do we prevent scale and concurrency from changing results?

If a deployment violates this document, it is **non-compliant**, even if it runs faster.

---

## 1. Supported Deployment Topologies

dashiCORE explicitly supports **three topology classes**.  
All others are out of scope unless added by amendment.

---

### 1.1 Single-Node Topology

**Definition**
- One host
- CPU only or CPU + local GPUs
- No inter-node communication

**Examples**
- CPU-only dev machine
- 1–8 GPU NVLink workstation / server

**Primary use**
- Semantic reference
- CI
- Parity validation
- Debugging and violation detection

**Guarantees**
- Strongest determinism
- Simplest reduction ordering
- Canonical “truth” execution

---

### 1.2 Pod Topology (Scale-Out)

**Definition**
- Multiple nodes
- Each node contains one or more GPUs
- Nodes connected via a high-speed fabric (e.g. InfiniBand / Ethernet)

**Examples**
- 2–32 nodes × 8 GPUs
- Fat-tree or Clos network

**Primary use**
- Throughput scaling
- Stress-testing modularity
- Large problem sizes

**Constraints**
- Inter-node communication must obey strict ordering rules
- Global reductions are restricted (see §4)

---

### 1.3 NVL72-Class Rack (Scale-Up)

**Definition**
- Rack-scale system
- Large NVLink/NVSwitch domain
- Treated as a *single coherence island*

**Examples**
- 72-GPU NVLink/NVSwitch systems
- Future rack-scale NVLink platforms

**Primary use**
- Communication-heavy workloads
- Deterministic multi-GPU execution without fabric ambiguity

**Guarantees**
- High bandwidth
- Low latency
- Easier determinism than scale-out pods

---

## 2. Backend Tiers

Every deployment must declare exactly **one backend tier**.

---

### 2.1 Truth Tier

**Role**
- Semantic authority
- Reference implementation

**Allowed topologies**
- Single-node only

**Requirements**
- CPU backend OR single deterministic GPU backend
- Float64 preferred
- No approximate math
- No atomics
- Fully deterministic reductions

**Obligation**
> If results differ from the truth tier, the other tier is wrong.

---

### 2.2 Throughput Tier

**Role**
- Faster execution with preserved semantics

**Allowed topologies**
- Single-node
- NVL72-class rack

**Requirements**
- Deterministic execution
- Fixed workgroup sizes
- Ordered reductions
- Backend parity with truth tier

**Allowed**
- Parallelism
- Vectorisation
- GPU execution

**Forbidden**
- Reduction reordering
- Atomic accumulation without proof
- Precision downgrades that change results

---

### 2.3 Scale-Out Tier

**Role**
- Maximum size / distribution
- Performance at scale

**Allowed topologies**
- Pod topology
- (Optionally) NVL72-class rack as a special case

**Requirements**
- Explicitly declared reduction strategy
- Deterministic per-node execution
- Restricted global reductions
- Strong observability

**Warning**
> Scale-out is where determinism is easiest to break.  
> This tier has the strictest contracts.

---

## 3. Determinism Requirements per Tier

| Tier        | Determinism Level | Notes |
|-------------|------------------|------|
| Truth       | Absolute          | Bit-exact reference |
| Throughput  | Absolute          | Must match truth tier |
| Scale-Out   | Absolute (with constraints) | Only allowed patterns |

There is **no tier** where “approximately equal” is acceptable by default.

---

## 4. Reduction Semantics (Critical)

Reductions are the primary source of nondeterminism.

---

### 4.1 Allowed Reduction Patterns

A reduction is allowed **only if**:

1. It is mathematically order-independent **and exact**, or
2. The reduction order is explicitly fixed and documented, or
3. The result is diagnostic only (not semantic)

---

### 4.2 Tier-Specific Rules

#### Truth Tier
- Single-thread or fixed-order reduction
- No parallel reduction ambiguity

#### Throughput Tier
- Parallel reductions allowed
- Reduction tree must be:
  - fixed
  - identical across runs
  - identical across devices

#### Scale-Out Tier
- Prefer:
  - per-node reductions
  - hierarchical fixed-order aggregation
- Global reductions must:
  - have a declared order
  - be test-verified for determinism
- Unordered collectives are forbidden

---

## 5. Minimal Cluster Contract

Every multi-GPU or multi-node deployment must satisfy the following.

---

### 5.1 Thread / Workgroup Pinning

Backends must explicitly pin:

- workgroup size
- grid dimensions
- thread-to-data mapping

No backend may rely on:
- “driver default” scheduling
- adaptive launch heuristics
- data-dependent launch geometry

If thread geometry changes, results must not.

---

### 5.2 Reduction Ordering

For every reduction used in semantics (defect, MDL):

- reduction order must be:
  - documented
  - fixed
  - testable

Forbidden:
- atomics as a substitute for ordering
- unspecified collective order

---

### 5.3 Metrics & Logs That Must Match

Across all backends and tiers, the following must be identical:

**Required**
- STATE logs
  - shape
  - support count
  - sign balance
- DEFECT logs
  - local nonzero count
  - aggregate defect value
- KERNEL logs
  - kernel name
  - idempotence / iteration index

**Optional but recommended**
- Visual parity:
  - support mask
  - defect heatmap

If any of these differ:
> The deployment is non-compliant.

---

## 6. Backend Admission Checklist (Topology-Aware)

A backend may run on a given topology iff:

- It declares required capabilities
- It satisfies the determinism rules of the tier
- It passes:
  - repeatability tests
  - backend parity tests
  - reduction stability tests

Topology does not weaken these requirements.

---

## 7. Non-Goals

This document does **not**:

- optimise cluster utilisation
- prescribe vendor hardware
- mandate specific network fabrics
- permit approximate math for speed

Those are operational concerns, not CORE semantics.

---

## 8. Relationship to Other Specs

This document is binding alongside:

- `docs/math_spec.md`
- `docs/typed_vocabulary.md`
- `docs/concurrency_spec.md`
- `docs/visual_log_spec.md`

In case of conflict:
**mathematical and determinism invariants take precedence.**

---

## Final Rule

> You may scale the hardware.  
> You may not scale away the guarantees.
