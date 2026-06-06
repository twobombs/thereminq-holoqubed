#!/bin/bash

# ==============================================================================
# ThereminQ-HPC Agentic Swarm Orchestrator
# 6x Qwen 3.6 9B MTP | 100% VRAM-Resident Pipeline
# Strict NUMA-to-PCIe Affinity Mapping
# ==============================================================================

# Model Configuration
MODEL="/media/aryan/nvme/models/Qwen3.5-9B-IQ4_XS.gguf"
LOG_DIR="./agent_logs"

# Ensure the log directory exists
mkdir -p $LOG_DIR

# Define the Swarm Topology: "Vulkan_ID  NUMA_Node  API_Port"
# Mapped directly from the physical sysfs PCIe tree
SWARM=(
  "0 0 8030"
  "1 0 8031"
  "2 1 8032"
  "3 1 8033"
  "4 6 8034"
  "5 6 8035"
)

# Graceful Shutdown: Catch Ctrl+C and kill all background servers
trap 'echo -e "\n[ThereminQ] Shutting down all agentic nodes..."; kill $(jobs -p); exit' SIGINT SIGTERM

echo "[ThereminQ] Initiating 6-Node Agentic Swarm..."

# Loop through the topology array and ignite each instance
for node_config in "${SWARM[@]}"; do
  # Read the variables from the current array string
  read -r VULKAN_ID NUMA_NODE PORT <<< "$node_config"

  echo "[+] Booting Instance -> Vulkan${VULKAN_ID} | NUMA Node ${NUMA_NODE} | Port ${PORT}"

  # Launch the server in the background (&)
  numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} ./build/bin/llama-server \
    -m $MODEL \
    -c 131072 \
    -np 1 \
    -ngl 999 \
    --device Vulkan${VULKAN_ID} \
    --kv-unified \
    -fa on \
    --split-mode none \
    --cache-type-k q8_0 \
    --cache-type-v q4_0 \
    -t 6 \
    -tb 6 \
    --no-mmap \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --chat-template-kwargs '{"preserve_thinking": true}' \
    --host 0.0.0.0 \
    --port ${PORT} \
    --tools all \
    --fit off > "${LOG_DIR}/vulkan${VULKAN_ID}_port${PORT}.log" 2>&1 &
done

echo "=============================================================================="
echo "[ThereminQ] Swarm is online."
echo "[ThereminQ] View individual initialization logs in: $LOG_DIR"
echo "[ThereminQ] Press [Ctrl+C] to gracefully terminate all instances."
echo "=============================================================================="

# Keep the script alive to hold the background jobs
wait
