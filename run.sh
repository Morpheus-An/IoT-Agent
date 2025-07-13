if [ -z "$1" ]
then
    # model="gpt4o-mini"
    model="gemini-1.5-pro-002"
    # model="llama2"
    # model="gpt3.5"
    # model="Mistral"
    # model="claude"
else
    model=$1
fi
# 13
if [ -z "$2" ]
then
    task_type="imu_HAR"
    # task_type="ecg_detection"
    # task_type="machine_detection"
else
    task_type=$2
fi

if [ -z "$3" ]
then
    sample_num=1
else
    sample_num=$3
fi

if [ -z "$4" ]
then
    # output_file_path="results/new_4omini.log"
    # output_file_path="results/new_mistral-3cls.log"
    # output_file_path="results/new_claude.log"
    # output_file_path="results/new_llama2.log"
    # output_file_path="results/new_gemini.log"


    # output_file_path="results/ECG_heartbeat/enhanced_performance/gpt35.log"
    # output_file_path="results/ECG_heartbeat/enhanced_performance/Mistral.log"
    # output_file_path="results/ECG_heartbeat/enhanced_performance/claude.log"
    # output_file_path="results/ECG_heartbeat/enhanced_performance/4omini.log"
    # output_file_path="results/ECG_heartbeat/enhanced_performance/gemini.log"
    # output_file_path="results/ECG_heartbeat/enhanced_performance/llama2.log"
    # output_file_path="results/machine/enhanced_performace/4omini.log"
    # output_file_path="results/new-enhaced-imu-3cls.log"
    output_file_path="results/calculate_cost.log"
else
    output_file_path=$4
fi

if [ -z "$5" ]
then 
    cls_num=3
else
    cls_num=$5
fi

if [ -z "$6" ]
then 
    grd="None"
else
    grd=$6
fi 



# for grd in "Neg" "Pos"
# for grd in "LAYING"
# for grd in "STANDING"
# for grd in "WALKING" "STANDING"
# for grd in "LIE_TO_SIT"
for grd in "LAYING" "WALKING_UPSTAIRS" "LIE_TO_S IT"
# for grd in "normal" "abnormal"
do  
# python ./main.py --task_type $task_type --cls_num 2 --sample_num $sample_num --model $model --device "cuda"\
    # --no_demo_knowledge
    # output_file_path="results/ablation/mcls-${task_type}_${grd}_${model}-3.log"
    python ./main.py --task_type $task_type --sample_num $sample_num --model $model --device "cuda" --no_demo_knowledge --output_file_path $output_file_path --cls_num $cls_num --grd $grd --debug
done  

