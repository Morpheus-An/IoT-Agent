if [ -z "$1" ]
then
    # model="gpt3.5"
    # model="gemini-pro"
    # model="claude"
    # model="llama2"
    model="Mistral"
    # model="gpt4o-mini"
else
    model=$1
fi

if [ -z "$2" ]
then
    task_type="machine_detection"
else
    task_type=$2
fi 

if [ -z "$3" ]
then
    sample_num=100
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
    output_file_path="results/machine/baseline/Mistral.log"
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
# for grd in "LAYING" "WALKING_UPSTAIRS" "LIE_TO_SIT"
do 
    python ./main.py --task_type $task_type --sample_num $sample_num --model $model --device "cuda" --no_demo_knowledge --output_file_path $output_file_path --cls_num $cls_num --grd $grd --no_domain_knowledge
done
