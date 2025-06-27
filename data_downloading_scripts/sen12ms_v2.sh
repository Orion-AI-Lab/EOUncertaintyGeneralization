#!/bin/bash
if [ -z "$1" ]; then
  echo "No root path. Exiting!"
  exit 1
fi

root_folder_path=$1
full_path="$root_folder_path/Sen12MS"
mkdir $full_path
mkdir $full_path/ROIs

for season in 1158_spring 1868_summer 1970_fall 2017_winter
    do
        for source in lc s1 s2
        do
            wget -O "$full_path/${season}_${source}.tar.gz" "ftp://m1474000:m1474000@dataserv.ub.tum.de/ROIs${season}_${source}.tar.gz"
            tar xvzf "$full_path/${season}_${source}.tar.gz" --directory "$full_path/ROIs/"
        done
    done

    for split in train test
    do
        wget -O "$full_path/${split}_list.txt" "https://raw.githubusercontent.com/schmitt-muc/SEN12MS/3a41236a28d08d253ebe2fa1a081e5e32aa7eab4/splits/${split}_list.txt"
    done