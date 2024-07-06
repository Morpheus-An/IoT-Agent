if [ -z "$1" ]
then
    model="gpt4"
else
    model=$1
fi

if [ -z "$2" ]
then
    task_type="imu_HAR"
else
    task_type=$2
fi

if [ -z "$3" ]
then
    sample_num=2
else
    sample_num=$3
fi


python ./main.py --task_type $task_type --cls_num 2 --sample_num $sample_num --no_demo_knowledge --model $model --device "cuda" 
