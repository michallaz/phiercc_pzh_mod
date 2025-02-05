#!/bin/bash
### Script downloads cgMLST profiles for Campylobacter, Salmonella and Ecoli
### and perform clustering using pHierCC methodology
### moves relevant data are strored in ../plepiseq_data/
### and commits chages to the repository
### The sctipt accepts two variables 
### --image image name with tag build using provided Dockerfile
### --output_dir path to the top level directory where intermediate calculations are carried out
### if direcotry does not exist it will be created
### if directory exists Salmonella/ Campylobacter/ and Escherichia subdirecories will be REMOVED
### Sctipt MUST be executed from main directory of the cloned repo, to commit all the changes
### Script will crash if machine has less than 600 Gb of RAM 

output_dir="" 
image_name=""
cpus=1


# Function to display help message
function show_help() {
    echo "Usage: $0 --output_dir <path> --image_name <string> --cpus <int>"
    echo "Script must be executed from main directory of the phiercc_pzh_mod repo, to commit all the changes"
    echo "Options:"

}


OPTIONS=$(getopt -o h --long output_dir:,image_name:,cpus:,help -- "$@")

eval set -- "$OPTIONS"

if [[ $# -eq 1 ]]; then
    echo "No parameters provided"
    show_help
    exit 1
fi


while true; do
    case "$1" in
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
	--cpus)
            cpus="$2"
            shift 2
            ;;
        --image_name)
            image_name="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if script is executed from within github repository	

if [ ! -d .git ]; then
	show_help
	exit 1
elif [ ! -d plepiseq_bin ]; then
	# in case user executes it from some other repository
	show_help
        exit 1
fi


## output path is provided
if [[ -z "$output_dir" ]]; then
    echo "Error: --output_dir is required."
    show_help
    exit 1
fi

### Create output directory if does not exists, this directory will be mounted by docker
output=$(realpath ${output_dir})
if [ ! -d "${output}" ]; then
    echo "Directory ${output} does not exist. Creating it..."
    mkdir -p "${output}/Salmonella"
    mkdir -p "${output}/Escherichia"
    mkdir -p "${output}/Campylobacter"
else
    echo "Directory ${output} exists. Removing data from selected subidrecotries..."
    rm -rf ${output}/Salmonella
    rm -rf ${output}/Escherichia
    rm -rf ${output}/Campylobacter

    mkdir -p "${output}/Salmonella"
    mkdir -p "${output}/Escherichia"
    mkdir -p "${output}/Campylobacter"
fi

### Check if the user-provided path has write permissions
if [ ! -w "$output" ]; then
    echo "Current user does not have write permissions to the directory $output"
    exit 1
fi

## image name is provided
if [[ -z "$image_name" ]]; then
    echo "Error: --image_name is required."
    show_help
    exit 1
fi

## image_name has a default, hence we only check if image name  is a valid docker image
tmp_name=`echo ${image_name} | cut -d ":" -f1`
tmp_tag=`echo ${image_name} | cut -d ":" -f2`

if [ $(docker images | grep "${tmp_name}" | grep "${tmp_tag}" | wc -l) -ne 1 ]; then
        echo "Provided docker image ${tmp_name}:${tmp_tag} does not exist. Provide valid image name"
        exit 1
fi


# download profiles
wget -O "${output}/Salmonella/profiles.list.gz"  "https://enterobase.warwick.ac.uk//schemes/Salmonella.cgMLSTv2/profiles.list.gz"
wget -O "${output}/Escherichia/profiles.list.gz" "https://enterobase.warwick.ac.uk//schemes/Escherichia.cgMLSTv1/profiles.list.gz"
python3 plepiseq_bin/download_profile_Campylo.py
mv profiles.list  "${output}/Campylobacter/"

echo "Running clustering for Campylobacter ~ 11 min"
docker run --rm \
       --volume "${output}/Campylobacter/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list" -n ${cpus} --clustering_method single

docker run --rm \
       --volume "${output}/Campylobacter/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list" --profile_distance0 "/dane/dist0.npy" --profile_distance1 "/dane/dist1.npy" -n 1 --clustering_method complete

echo "Finished calculations for Campylobacter"



echo "Running clustering for Ecoli ~ 11 min"
docker run --rm \
       --volume "${output}/Escherichia/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list.gz" -n ${cpus} --clustering_method single

docker run --rm \
       --volume "${output}/Escherichia/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list.gz" --profile_distance0 "/dane/dist0.npy" --profile_distance1 "/dane/dist1.npy" -n 1 --clustering_method complete

echo "Finished calculations for Ecoli"

echo "Running clustering for Salmonella ~8h "
# calculate profiles /For salmonella even when using 250 cores  it will take ~16h)
docker run --rm \
       --volume "${output}/Salmonella/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list.gz" -n ${cpus} --clustering_method single

docker run --rm \
       --volume "${output}/Salmonella/:/dane:rw" \
       --user $(id -u):$(id -g) \
       ${image_name} --profile "/dane/profiles.list.gz" --profile_distance0 "/dane/dist0.npy" --profile_distance1 "/dane/dist1.npy" -n 1 --clustering_method complete

echo "Finished calculations for Salmonella"

# moving results to plepiseq_data
# updating repo

if [ ! -d "plepiseq_data/Campylobacter" ]; then
	mkdir -p "plepiseq_data/Campylobacter"
fi
cp  ${output}/Campylobacter/*HierCC* plepiseq_data/Campylobacter


if [ ! -d "plepiseq_data/Salmonella" ]; then
        mkdir -p "plepiseq_data/Salmonella"
fi
cp  ${output}/Salmonella/*HierCC* plepiseq_data/Salmonella

if [ ! -d "plepiseq_data/Escherichia" ]; then
        mkdir -p "plepiseq_data/Escherichia"
fi
cp  ${output}/Escherichia/*HierCC* plepiseq_data/

# Update timestamp
date '+%D' > "plepiseq_data/timestamp"

git add plepiseq_data/*
git commit -m "Update on `date +%D`"
git push
