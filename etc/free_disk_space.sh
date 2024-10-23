#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#
# The Azure provided machines typically have the following disk allocation:
# Total space: 85GB
# Allocated: 67 GB
# Free: 17 GB
# This script frees up 28 GB of disk space by deleting unneeded packages and
# large directories.
# The Flink end to end tests download and generate more than 17 GB of files,
# causing unpredictable behavior and build failures.
#
echo "=============================================================================="
echo "Freeing up disk space on CI system"
echo "=============================================================================="

if [ ! -z "${SHELL}" ]; then
    run() {
        "${SHELL}" -c "$*" || true
    }
else
    run() {
        /bin/sh -c "$*" || true
    }
fi

if [ ! -z "$CI" ]; then
    group() {
        echo "::group::$*"
        2>&1 run "$@"
        echo "::endgroup::"
    }
else
    group() { run "$@"; }
fi

set -x
echo "::group::Listing 100 largest packages"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100
echo "::endgroup::"
df -h
echo "Removing large packages"
group sudo apt-get update -y
# purge is like remove, but also removes configuration
group sudo apt-get purge -y '^ghc-.*'
group sudo apt-get purge -y '^dotnet-.*'
group sudo apt-get purge -y '^llvm-.*'
group sudo apt-get purge -y '^libclang-.*'
group sudo apt-get purge -y '^gcc-.*'
group sudo apt-get purge -y '^temurin-.*-jdk'
group sudo apt-get purge -y 'php.*'
group sudo apt-get purge -y 'google-cloud.*'
group sudo apt-get purge -y 'google-chrome.*'
group sudo apt-get purge -y azure-cli
group sudo apt-get purge -y hhvm
group sudo apt-get purge -y firefox
group sudo apt-get purge -y powershell
group sudo apt-get purge -y mono-devel
group sudo apt-get purge -y microsoft-edge-stable
group sudo apt-get purge -y '^openjdk-.*'
group sudo apt-get purge -y '^mysql-server.*'
group sudo apt-get purge -y '^libllvm.*'
group sudo apt-get purge -y 'linux-headers-*'
group sudo apt-get purge -y 'snapd'
group sudo apt-get autoremove -y
group sudo apt-get clean
group sudo apt-get update -y
echo "::group::Listing 100 largest remaining packages"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100
echo "::endgroup::"
group df -h
echo "Removing large directories"
# deleting 15GB
rm -rf /usr/share/dotnet/
group df -h
group 'du -sh /opt/hostedtoolcache/* | sort -h'
rm -rf /opt/hostedtoolcache/go /opt/hostedtoolcache/CodeQL /opt/hostedtoolcache/node
group 'du -sh /* | sort -h'
sudo rm -rf /lib/heroku # TODO: figure out how to remove not manually
sudo rm -rf /etc/skel
sudo rm -rf /var/cache
sudo mkdir -p /var/cache/apt/archives/partial
sudo touch /var/cache/apt/archives/lock
sudo chmod 640 /var/cache/apt/archives/lock
sudo rm -rf /usr/share/swift
sudo rm -rf /usr/share/miniconda
sudo rm -rf /usr/share/az_*
sudo rm -rf /usr/local/aws-*cli*
sudo rm -rf /usr/local/julia*
sudo rm -rf /usr/share/gradle*
sudo rm -rf /usr/share/sbt*
sudo swapoff /mnt/swapfile
sudo rm -rf /mnt/swapfile
sudo rm -rf /usr/local/share/chromium # 1.2G
sudo rm -rf /usr/local/share/powershell # 530M
sudo rm -rf /usr/local/share/vcpkg # 152M
sudo rm -rf /usr/local/lib/android # 7.6G
# sudo rm -rf /usr/local/lib/node_modules # 1.1G
for d in / /etc /snap /var /home /opt /mnt /usr \
    /var/lib /home/runner /opt/hostedtoolcache /usr/share /usr/local \
    /usr/local/share /usr/local/lib; do
# 163M	/home/linuxbrew
# 167M	/opt/actionarchivecache
# 209M	/opt/runner-cache
# 210M	/snap/core20
# 297M	/snap/lxd
# 431M	/opt/pipx
# 537M	/lib/x86_64-linux-gnu
# 552M	/lib/gcc
# 603M	/etc/skel
# 625M	/var/lib
# 1.4G	/home/runner
# 2.3G	/opt/hostedtoolcache
# 3.9G	/usr/share
# 4.1G	/mnt/swapfile
# 19G	/usr/local
    echo "::group::$d ($(du -sh $d 2>/dev/null | cut -f1))"
    du -sh $d/* 2>&1 | sort -h
    echo "::endgroup::"
done
group df -h
