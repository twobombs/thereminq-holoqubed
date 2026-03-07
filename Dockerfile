# ------------------------------------------------------------------------
# ThereminQ Holoqubed - Prototype Environment
# OS: Ubuntu 26.04 LTS (For native Mesa 25.x Rusticl OpenCL support)
# Hardware Target: AMD Radeon Pro V340 (gfx900 / Vega 10)
# ------------------------------------------------------------------------

FROM twobombs/thereminq-desktop

# Prevent tzdata and other interactive prompts from hanging the build
ENV DEBIAN_FRONTEND=noninteractive

# Update the package index and install required dependencies
# We install python3-pyopencl and python3-numpy from the repo to ensure
# they link cleanly against the system's OpenCL ICD.
RUN apt-get update && apt-get install -y --no-install-recommends \
    mesa-opencl-icd \
    clinfo \
    python3 \
    python3-pip \
    python3-pyopencl \
    && rm -rf /var/lib/apt/lists/*
# Python dependancies that need upgrades
RUN pip install --break-system-packages gguf numpy
# ------------------------------------------------------------------------
# OpenCL Driver Configuration (Mesa Rusticl)
# ------------------------------------------------------------------------

# 1. Force Mesa to use the modern Rusticl driver instead of legacy Clover
#    and bind it specifically to the 'radeonsi' Gallium driver for Vega.
ENV RUSTICL_ENABLE=radeonsi

# 2. Tell the OpenCL Installable Client Driver (ICD) loader to strictly 
#    prioritize Rusticl over any other OpenCL runtimes that might exist.
ENV OCL_ICD_VENDORS=rusticl.icd

# 3. Legacy ROCm override (Optional, but highly recommended). 
#    While Rusticl bypasses ROCm, some underlying AMD LLVM compilation 
#    paths still check the GFX version. This forces recognition of Vega 10.
ENV HSA_OVERRIDE_GFX_VERSION=9.0.0

# ------------------------------------------------------------------------
# Application Setup
# ------------------------------------------------------------------------

# Set the working directory inside the container
WORKDIR /app

# Copy your prototype Python engine into the container
# Ensure 'holoqubed_prototype.py' is in the same folder as this Dockerfile
COPY holoqubed_prototype.py /app/

# Set the default command to execute the Holoqubed prototype
CMD ["python3", "holoqubed_prototype.py"]
