FROM slurm_common

LABEL description="Compute Node Image for Slurm Virtual Cluster"

USER root

# copy slurm rpm
COPY ./docker/RPMS/aarch64/slurm-[0-9]*.rpm ./docker/RPMS/aarch64/slurm-slurmd-*.aarch64.rpm /root/

#install Slurm
RUN ls && yum update --assumeno || true && \
    yum -y install \
        slurm-[0-9]*.aarch64.rpm \
        slurm-slurmd-*.aarch64.rpm \
        && \
    rm slurm*.rpm  && \
    mkdir /var/log/slurm  && \
    chown -R slurm:slurm /var/log/slurm  && \
    mkdir /var/state  && \
    chown -R slurm:slurm /var/state  && \
    mkdir -p /var/spool/slurmd  && \
    chown -R slurm:slurm /var/spool/slurmd && \
    yum clean all && \
    rm -rf /var/cache/yum
EXPOSE 6818


# setup entry point
ENTRYPOINT ["/usr/local/sbin/cmd_start"]
CMD ["-loop", "/opt/cluster/vctools/init_system", "munged", "slurmd", "sshd", "/opt/cluster/vctools/init_slurm", "bash"]
