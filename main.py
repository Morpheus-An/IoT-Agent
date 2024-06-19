from imports import *
from utils import * 


def chat_with_openai(data_dict, ground_ans: str="WALKING", contrast_ans: str="STANDING", answer_num: int=10, api_base: bool=True, model: str=MODEL["gpt3.5"], retrive=False, print_prompt=True):
    """
    return:
    results: list of str
    """
    results = []
    client = get_openAI_model(api_base, model)
    for i in range(answer_num):
        prompt = gen_prompt_template_without_rag(data_dict, ground_ans, contrast_ans, i, retrive)
        if print_prompt:
            print(prompt)
            return 
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = completion.choices[0].message.content
        if i % 5 == 0:
            print(f"第{i}次预测完成")
        results.append(result)
        print(result)
        print("-"*100)
    return results
        

if __name__ == "__main__":
    data_dict = read_raw_data_and_preprocess()
    filtered_data_dict = filter_data_dict_with_var(data_dict, thred=0.5, filter_by="body_acc", print_log=False)
    data_dict = filtered_data_dict

    ground_ans = "STANDING"
    contrast_ans = "WALKING"

    start_time = time.perf_counter()
    # without retrieval:
    # ans = chat_with_openai(data_dict, ground_ans=ground_ans, contrast_ans=contrast_ans, answer_num=10, api_base=True, model=MODEL["gpt3.5"], retrive=False, print_prompt=False)
    # print(eval_generated_ans(ans, ground_ans, contrast_ans))

    # with retrieval:
    # 首先，准备好document_store并写入:
    KB_paths = [
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Accelemeters.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Acceleration.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Activity-recognition.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Angular-Velocity.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Gyroscope.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/How-work.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Human Activity Recognition using Accelerometer.pdf",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/knowledge-from-model.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Activity Recognition Using Cell Phone Acceleromete.pdf",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Patterns of Bipedal Walking on Tri-axial Accelerat.pdf",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Sitting.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Standing.txt",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/Uni- and triaxial accelerometric signals agree during daily routine, but show differences between sports.pdf",
        "/home/ant/RAG/IMU_knowledge/domain-knowledge/walking.txt",
    ]
    ans = generate_with_rag(ground_ans, contrast_ans, KB_paths, data_dict, num_samples=5)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"总共耗时{elapsed_time}秒")
    print(eval_generated_ans(ans, ground_ans, contrast_ans))

     




    
