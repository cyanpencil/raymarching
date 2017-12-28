#!/bin/bash

cd $(dirname "$(readlink -f "$0")")

if [[ ./bin/fragment_shader.frag -nt ./apps/fragment_shader.frag ]]; then
  for i in $(seq 100); do 
  echo "The fragment shader in the apps folder is older than the fragment shader in the bin folder, did not compile."
  done
  exit 1
fi

cp -p ./apps/fragment_shader.frag ./bin/

cd build
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
cmake .. 
make -j 4

