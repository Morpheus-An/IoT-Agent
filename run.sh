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


if [ -z "$4" ]
then
    output_file_path="results/output.log"
else
    output_file_path=$4
fi

if [ -z "$3" ]
then
    sample_num=2
else
    sample_num=$3
fi

if [ -z "$5" ]
then 
    cls_num=2
else
    cls_num=$5
fi

if [ -z "$6" ]
then 
    grd="None"
else
    grd=$6
fi 

# python ./main.py --task_type $task_type --cls_num 2 --sample_num $sample_num --model $model --device "cuda"\
    # --no_demo_knowledge
python ./main.py --task_type $task_type --sample_num $sample_num --model $model --device "cuda" --no_demo_knowledge --output_file_path $output_file_path --cls_num $cls_num --grd $grd --no_domain_knowledge --debug
