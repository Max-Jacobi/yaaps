#!/bin/bash
######################## load environment #######################
source ${HOME}/envs/conda.start

######################## parse arguments ########################
usage() {
    echo 'Usage: $(basename "$0") [-h] [-H] [-V] [-p] [-t] [var2 ...] [--args "extra-args"] -v <var1>'
    echo " -h       Help"
    echo " -H       Show help message from the plot script and exit [optional]"
    echo " -V       Show list of available vars and exit [optional]"
    echo " -p       Do a single plot instead of an animation [optional]"
    echo " -t       Time for the single plot [optional]"
    echo " --args   Extra arguments for the plotting script [optional]"
    echo " -v       List of variables to plot (required if not using -h, -H, -V)"
}

show_help_from_plot=false
do_list_vars=false
do_animate=true
pyexec=simple_anim.py
time=10
extra_args=()
vars=()
planes=("xy" "xz")

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h)
            usage
            return 0
            ;;
        -H)
            show_help_from_plot=true
            shift
            ;;
        -V)
            do_list_vars=true
            shift
            ;;
        -p)
            do_animate=false
            pyexec=simple_plot.py
            shift
            ;;
        -t)
            shift
            time=("$1")
            shift
            ;;
        --args)
            shift
            extra_args=("$1")
            shift
            ;;
        -v)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                vars+=("$1")
                shift
            done
            ;;
        -*)
            echo "Unknown option $1"
            return 1
            ;;
        *)
            echo "Unexpected argument $1"
            return 1
            ;;
    esac
done

if [[ ${#vars[@]} -eq 0 && $do_list_vars == false && $show_help_from_plot == false ]]; then
    echo "Error: -v is required if not using -h, -H, or -V"
    usage
    return 1
fi

if [ $do_animate = false ]; then
    extra_args+="--time $time"
    echo ${extra_args}
fi

######################## run analysis ########################
# path to directory containing all outputs
dir_base=$(dirname "$(pwd)")
# output directory for images: structure automatically created if missing
dir_out=$(pwd)/analysis
# sim tagname
simtag=$(basename "$(pwd)")
# path to directory containing *.athdf
dir_data=$(pwd)

cd ${DIR_REP}/yaaps/examples

if [ $show_help_from_plot = true ]; then
    python ${pyexec} -h
elif [ $do_list_vars = true ]; then
    python ${pyexec} -s ${dir_data}
else
  for var in "${vars[@]}"; do
    for plane in "${planes[@]}"; do

        exec="${pyexec}             \
            ${var}                  \
            --simdir ${dir_data}    \
            --outputpath ${dir_out} \
            --sampling ${plane}     \
            ${extra_args[@]}
        "
        echo
        echo Running ${exec}
        python ${exec}

    done
    echo
  done
fi

cd ${dir_data}
