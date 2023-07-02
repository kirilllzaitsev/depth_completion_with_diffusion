#!/bin/bash


# GPU
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --gres=gpumem:10240m

# GENERAL
#SBATCH -A es_hutter
#SBATCH --cpus-per-task=25
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=6:00:00
#SBATCH --job-name="kbnet"
#SBATCH --open-mode=append
#SBATCH --output="/cluster/home/kzaitse/outputs/kbnet/kbnet-%j.txt"

export job_id=$SLURM_JOBID

export dest_dataset_dir=/cluster/scratch/kzaitse/extracted
export kitti_base_dir=$dest_dataset_dir/kitti_dataset
mkdir -p $kitti_base_dir

module load gcc/8.2.0 python_gpu/3.8.5

prepared_data_archives_base_dir=/cluster/scratch/kzaitse/kitti

echo Started extraction. `date`
echo "Extracting to $kitti_base_dir"

for data_archive in kitti_raw_data.tar.gz kitti_depth_completion.tar.gz kitti_depth_completion_kbnet.tar.gz train_val_test_file_paths.tar.gz
do
    tar xvzif "$prepared_data_archives_base_dir/$data_archive" -C $kitti_base_dir #> /dev/null
done
echo Finished extraction. `date`

base_kbnet_dir=/cluster/home/kzaitse/benchmarking/calibrated-backprojection-network
cd "$base_kbnet_dir" || exit
bash "./bash/kitti/train_kbnet_kitti.sh" "$kitti_base_dir/data" "$base_paper_dir/trained_kbnet/kitti"
bash "./bash/kitti/run_kbnet_kitti_validation.sh" "$kitti_base_dir/data" "$base_kbnet_dir/pretrained_models/kitti"
bash "./bash/kitti/run_kbnet_kitti_testing.sh" "$kitti_base_dir/data" "$base_kbnet_dir/pretrained_models/kitti"

echo Done. `date`