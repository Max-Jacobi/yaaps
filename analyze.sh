#!/bin/bash
######################## load environment #######################
source ${HOME}/envs/conda.start

######################## parse arguments ########################
usage() {
    echo 'Usage: $(basename "$0") [-h] [-V] [-b <boundary>] [--fps <fps>] [-v <var1> ...]'
    echo " -h       Help"
    echo " -V       Show list of available vars and exit [optional]"
    echo " -b       Boundary of the plot [optional]"
    echo " --fps    Number of fps for the mp4 output [optional]"
    echo " -v       List of variables to plot [optional]"
}

vars1D="max_rho max_T tot-E "                       # hydro
vars1D+="m_ej_bern "                                # ejecta
vars1D+="min_alpha C-norm2 mass num_c2p_fail "      # GR evolution
vars1D+="max_B2 div_B "                             # magnetic fields
vars1D+="max_sc_nG_00,max_sc_nG_01,max_sc_nG_02 "   # neutrinos number densities
vars1D+="max_sc_E_00,max_sc_E_01,max_sc_E_02"       # neutrinos energy densities

boundary=""
fps=""
planes=("xy" "xz")
vars2D=("rho" "ye")
vars2D+=("B_x" "B_y" "B_z")             # magnetic fields
vars2D+=("m1_n_e" "m1_n_ae" "m1_n_x")   # neutrinos number densities
vars2D+=("m1_J_e" "m1_J_ae" "m1_J_x")   # neutrinos energy densities

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h)
            usage
            return 0
            ;;
        -b)
            shift
            boundary="--boundary $1"
            shift
            ;;
        --fps)
            shift
            fps="--fps $1"
            shift
            ;;
        -v)
            shift
            vars2D=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                vars2D+=("$1")
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

##############################################################
######################## Run analysis ########################

# path to directory containing all outputs
dir_base=$(pwd)
# path to directory containing the combined results
dir_comb=${dir_base}/combine
# output directory for images: structure automatically created if missing
dir_out=${dir_comb}/analysis

################### Instruction for Ctrl+C ###################

cleanup() {
    echo
    echo "Caught Ctrl+C or script exiting. Returning to base directory..."
    cd "$dir_base" || return 1
}
trap cleanup SIGINT SIGTERM EXIT

################### Combine restart folders ##################

# update combined results folder
combinepy="combine.py"
if [ -f $combinepy ]; then
    echo "Updating combined results with $(ls -la $combinepy)"
else
    echo "Linking combine script from ${DIR_REP}/yaaps"
    ln -s ${DIR_REP}/yaaps/$combinepy .
fi

python $combinepy .

##################### 1D diagnosys plots #####################

echo
echo "======================================================="
echo "Running 1D diagnosis plots"

cd ${DIR_REP}/yaaps/examples

exec="plot_hst.py           \
    ${vars1D}               \
    --simdir ${dir_comb}    \
    --outputpath ${dir_out}
"
python ${exec}

cd ${dir_base}

echo

###################### 2D analysis plots #####################

echo
echo "======================================================="
echo "Running 2D analysis plots"

cd ${DIR_REP}/yaaps/examples

for var in "${vars2D[@]}"; do
    echo "  Plotting ${var} ..."

    for plane in "${planes[@]}"; do

        exec="simple_anim.py        \
            ${var}                  \
            --simdir ${dir_comb}    \
            --outputpath ${dir_out} \
            --sampling ${plane}     \
            $boundary $fps
        "
        python ${exec}

    done
done

cd ${dir_base}

echo
