FROM slurm_common

LABEL description="HeadNode Image for Slurm Virtual Cluster"

USER root

# install dependencies
RUN yum update --assumeno || true && \
    yum -y install --setopt=tsflags=nodocs \
        vim tmux mc perl-Switch \
        iproute perl-Date* \
        mariadb-server mariadb-devel gcc curl-devel  python3 python3-devel sqlite-devel default-libmysqlclient-dev build-essential pkg-config && \
    yum clean all && \
    rm -rf /var/cache/yum

# COPY ./docker/install_python.sh /root/
# RUN /root/install_python.sh
# RUN python3 --version


#configure mysqld
RUN chmod g+rw /var/lib/mysql /var/log/mariadb /var/run/mariadb && \
    mysql_install_db && \
    chown -R mysql:mysql /var/lib/mysql && \
    cmd_start mysqld && \
    mysql -e 'DELETE FROM mysql.user WHERE user NOT LIKE "root";' && \
    mysql -e 'DELETE FROM mysql.user WHERE Host NOT IN ("localhost","127.0.0.1","%");' && \
    mysql -e 'GRANT ALL PRIVILEGES ON *.* TO "root"@"%" WITH GRANT OPTION;' && \
    mysql -e 'GRANT ALL PRIVILEGES ON *.* TO "root"@"localhost" WITH GRANT OPTION;' && \
    mysql -e 'DROP DATABASE IF EXISTS test;' && \
    mysql -e "CREATE USER 'slurm'@'%' IDENTIFIED BY 'slurm';" && \
    mysql -e 'GRANT ALL PRIVILEGES ON *.* TO "slurm"@"%" WITH GRANT OPTION;' && \
    mysql -e "CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'slurm';" && \
    mysql -e 'GRANT ALL PRIVILEGES ON *.* TO "slurm"@"localhost" WITH GRANT OPTION;' && \
    cmd_stop mysqld

# copy slurm rpm
COPY ./docker/RPMS/aarch64/slurm*.rpm /root/

#install Slurm
RUN yum update --assumeno || true && \
    yum -y install \
        slurm-[0-9]*.aarch64.rpm \
        slurm-perlapi-*.aarch64.rpm \
        slurm-slurmctld-*.aarch64.rpm \
        slurm-slurmdbd-*.aarch64.rpm  \
        slurm-pam_slurm-*.aarch64.rpm && \
    rm slurm*.rpm  && \
    mkdir /var/log/slurm  && \
    chown -R slurm:slurm /var/log/slurm  && \
    mkdir /var/state  && \
    chown -R slurm:slurm /var/state  && \
    mkdir -p /var/spool/slurmd  && \
    chown -R slurm:slurm /var/spool/slurmd && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    touch /bin/mail  && chmod 755 /bin/mail && \
    echo '/opt/cluster/vctools/start_head_node.sh' >> /root/.bash_history


RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install jupyter jupyterlab numpy Flask torch scikit-learn catboost PyMySQL


EXPOSE 6819
EXPOSE 6817
EXPOSE 8888

COPY ./docker/RPMS/job_submit_predict.so /usr/lib64/slurm
COPY ../server /opt/cluster/server

# setup entry point
ENTRYPOINT ["/usr/local/sbin/cmd_start"]
CMD ["-loop", "/opt/cluster/vctools/init_system", "munged", "mysqld", "slurmdbd", "slurmctld", "sshd", "/opt/cluster/vctools/init_slurm", "bash"]
