#!/bin/sh

TAG=casual_code_capability

arg0=$(basename "$0" .sh)
blnk=$(echo "$arg0" | sed 's/./ /g')

usage_info(){
    echo "Usage: $arg0 [{-b|--build} {dockerfile_path}] \\"
    echo "       $blnk [{-c|--create}] {GPU|''} \\"
    echo "       $blnk [{-s|--start}] {i|''} \\"
    echo "       $blnk [-p|--stop] \\"
    echo "       $blnk [-k|--clean] \\"
    echo "       $blnk [-h|--help]"
    exit 1
}

error(){
    echo "$arg0: $*"
    usage_info
    exit 1
}


help(){
    echo
    echo "  [{-b|--build} dockerfile_path] build-image    -- Build docker image from the specified dockerfile (default: Dockerfile)"
    echo "  [{-c|--create} {GPU|''}] create-container     -- Create the container from docker image, pass 'GPU' argument to use nvidia environment"
    echo "  [{-s|--start} {i|''}] start-container         -- Start the container from docker image, pass 'i' to use interactive mode"
    echo "  [-p|--stop] stop-container                    -- Stop the container from docker image"
    echo "  [-k|--clean] clean_all                        -- Delete the container and the image"
    echo "  [-h|--help] help                              -- Print this help message and exit"
    exit 0
}

build_image(){
  echo "> building image with name "$TAG"-img"
  docker build -t "$TAG-img" -f "$DOCKERFILE" .
}

create_container(){
  echo "> creating container with name "$TAG"-ctr"
  if [ "$GPU" == "true" ]
    then
    echo ">> NVIDIA runtime"
    docker create --runtime=nvidia --name "$TAG-ctr" -v $(pwd):/tf/main \
    -p 0.0.0.0:6008:6006 -p 8002:8888 "$TAG-img"
  else
    docker create --name "$TAG-ctr" -v $(pwd):/tf/main -p 8888:8888 "$TAG-img"
  fi
  exit 0
}

start_container(){
  echo "> starting container with name "$TAG"-ctr"
  if [ "$interactive_mode" == "true" ]
    then
      docker start -i "$TAG-ctr"
  else
    docker start "$TAG-ctr"
  fi
  exit 0
}

stop_container(){
  echo "> stopping container with name "$TAG"-ctr"
  docker stop "$TAG-ctr"
  exit 0
}

clean_all(){
  echo "> removing container with name "$TAG"-ctr"
  docker container rm "$TAG-ctr"
  echo "> removing image with name "$TAG"-img"
  docker image rm "$TAG-img"
  exit 0
}

flags(){
  while test $# -gt 0
    do
        case "$1" in
        (-b|--build)
            shift
            [ $# = 0 ] && error "No dockerfile path specified"
            export DOCKERFILE="$1"
            build_image
            shift;;
        (-c|--create)
            shift
            if [ -n "$1" ] && [ "$1" == "GPU" ]
              then
                export GPU="true"
            fi
            create_container
            shift;;
        (-s|--start)
            shift
            if [ -n "$1" ] && [ "$1" == "i" ]
              then
                export interactive_mode="true"
            fi
            start_container
            shift;;
        (-p|--stop)
          stop_container;;
        (-k|--clean)
            clean_all;;
        (-h|--help)
            help;;
        (*) usage_info;;
        esac
    done
}

flags "$@"