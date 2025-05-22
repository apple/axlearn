set -ex
job_dir=$(realpath artifacts/${1})
log_dir=$(realpath logs/)
profile_id=$2
s3_profile_path=s3://kaena-tempdata/huilgolr/fs-moe/profiles/$2

profile_dir=$(echo ${job_dir}/rt_profiles/*pid* | awk '{print $1}')
upload_dir=$job_dir/to_upload
mkdir -p $upload_dir

echo "Archiving penguin artifacts"
neff_path=$(ls -S ${job_dir}/neuron_dump/*program*/file.neff | head -n1)
cd $(dirname $neff_path)
tar -cf penguin-text.tar penguin-sg*
cp penguin-text.tar $upload_dir/
cp file.code $upload_dir/
cp file.neff $upload_dir/
cp log-neuron-cc.txt $upload_dir/

cd $log_dir
cp $log_dir/${1}*.out $upload_dir/

echo "Archiving HLOs"
cd $job_dir/hlo_dump
tar -cf hlo_dump.tar *
cp hlo_dump.tar $upload_dir/

cd $profile_dir
cp *vnc_0.ntff $upload_dir/


aws s3 sync --no-progress $upload_dir $s3_profile_path
set +x
echo "Profile uploaded to $s3_profile_path"
echo "profile-upload -F \"s3=$s3_profile_path\" -F name=$profile_id -F \"profiler-opts='--enable-memory-tracker'\""