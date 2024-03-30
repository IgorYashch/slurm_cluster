# #!/bin/bash
# echo "Adding users to the system..."
# if [[ "`hostname`" == "headnode" ]]; then
#     echo "headnode."
#     USERADD_FLAG="-m"
# else
#     echo "computenode."
#     USERADD_FLAG="-M"
# fi
USERADD_FLAG="-N -g users"

# for i in range(100):
#     print(f"useradd ${{USERADD_FLAG}} -u {10000+i:d} -s /bin/bash user{i:03d} && echo 'user{i:03d}:user' |chpasswd")

create_user() {
  username=$1
  id=$2
  if ! id -u "$username" > /dev/null  2>&1; then
    useradd -M $USERADD_FLAG -u "$id" -s /bin/bash "$username" && echo "$username:user" |chpasswd
    mkdir -p "/home/$username"
  fi
}

create_user user1 10001
create_user user2 10002
create_user user3 10003
create_user user4 10004
create_user user5 10005

chown -R :users /home

# sudo useradd ${USERADD_FLAG} -u 10001 -s /bin/bash user1 && echo 'user1:user' |chpasswd
# sudo useradd ${USERADD_FLAG} -u 10002 -s /bin/bash user2 && echo 'user2:user' |chpasswd
# sudo useradd ${USERADD_FLAG} -u 10003 -s /bin/bash user3 && echo 'user3:user' |chpasswd
# sudo useradd ${USERADD_FLAG} -u 10004 -s /bin/bash user4 && echo 'user4:user' |chpasswd
# sudo useradd ${USERADD_FLAG} -u 10005 -s /bin/bash user5 && echo 'user5:user' |chpasswd

