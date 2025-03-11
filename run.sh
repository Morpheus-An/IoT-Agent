if [ -z "$1" ]
then
    # model="gpt3.5"
    # model="gemini-1.5-pro-002"
    # model="claude"
    # model="llama2"
    # model="Mistral"
    model="gpt4o-mini"
else
    model=$1
fi

if [ -z "$2" ]
then
    task_type="machine_detection"
    # task_type="ecg_detection"
    # task_type="imu_HAR"
else
    task_type=$2
fi 

if [ -z "$3" ]
then
    sample_num=50
else
    sample_num=$3
fi

if [ -z "$4" ]
then
    # output_file_path="results/output_baseline_gemini.log"
    # output_file_path="results/machine/baseline/gemini.log"
    # output_file_path="results/machine/baseline/gpt35.log"
    # output_file_path="results/machine/baseline/claude.log"
    # output_file_path="results/machine/baseline/4omini.log"
    # output_file_path="results/machine/baseline/llama.log"
    # output_file_path="results/machine/baseline/Mistral.log"
    # output_file_path="results/new_ablation_study/ecg/3.log"
    # output_file_path="results/new_ablation_study/HAR/1-3cls.log"
    # output_file_path="results/new-baseline4imu-3cls/gemini.log"
    # output_file_path="results/new-baseline4imu-3cls/claude.log"
    # output_file_path="results/new_ablation_study/HAR/3-2cls.log"
    output_file_path="results/new_ablation_study/machine/2.log"
    # output_file_path="results/new-baseline4imu-3cls/Mistral.log"

else
    output_file_path=$4
fi


if [ -z "$5" ]
then 
    cls_num=2
else
    cls_num=$5
fi

# if [ -z "$6" ]
# then 
#     grd="WALKING"
# else
#     grd=$6
# fi 

# python ./main.py --task_type $task_type --cls_num 2 --sample_num $sample_num --model $model --device "cuda"\
    # --no_demo_knowledge
# for grd in "WALKING" "STANDING"
for grd in "Pos" "Neg"
# for grd in "normal" "abnormal"
# for grd in "LAYING" "WALKING_UPSTAIRS" "LIE_TO_SIT"
do 
    python ./main.py --task_type $task_type --sample_num $sample_num --model $model --device "cuda" --no_demo_knowledge --output_file_path $output_file_path --cls_num $cls_num --grd $grd
done


