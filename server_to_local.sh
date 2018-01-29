#!/usr/bin/env bash

local_root="/Users/pcf/PycharmProjects/yuntext/"
server_root="/home/xuhongbo/pcf/yuntext/"

scp -r xuhongbo@10.60.1.79:${server_root}${1} ${local_root}${1}
