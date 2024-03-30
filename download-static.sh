#!/bin/sh

set -exu

# Download the static files
dataset=$1

curl -o ${dataset}_static.tar.gz \
	https://lake.fmi.fi/cc_archive/neural-lam/${dataset}_static.tar.gz

