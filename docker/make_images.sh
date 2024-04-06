#!/bin/bash
# run from the parent directory
# build all images

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLUSTER_MAIN_DIR="$( dirname "${CUR_DIR}" )"

#exit on any error
set -e

cd "${CLUSTER_MAIN_DIR}"

# Making Slurm RPMs
rm -rf "${CLUSTER_MAIN_DIR}/docker/RPMS/*"

# make image for building slurm binaries
docker build -t slurm_rpm_maker -f ./docker/MakeSlurmRPM.Dockerfile .

docker run --name slurm_rpm_maker -h slurm_rpm_maker \
           -v `pwd`/docker/RPMS:/RPMS:Z \
           --rm \
           -it slurm_rpm_maker

# Build Common Image
docker build -f docker/Common.Dockerfile -t slurm_common .

# Build Head-Node Image
docker build -f docker/HeadNode.Dockerfile -t slurm_head_node .

# # Build Compute-Node Image
docker build -f docker/ComputeNode.Dockerfile -t slurm_compute_node .

