#!/bin/bash

#defaults
port=8888
build=false

#arg parsing
while getopts bp: option
do 
    case "${option}" in
        p)port=${OPTARG};;
        b)build=true;;
        ?) echo "script usage: $(basename \$0) [-p] [-b]"
           exit 1
           ;; 
    esac
done

#set composer variables
dir_path=$(dirname $(pwd))
echo "dir_path=$dir_path">.env
_basename=$(basename $dir_path)
_basename="${_basename,,}"
echo "dir_name=$_basename">>.env
echo "port=$port">>.env
echo "USER=$USER">>.env

#run build or up depending on -b flag
if [[ $build = false ]]
then
    docker-compose -p $USER up
else
    docker-compose -p $USER build
fi