from imports import *
# from utils import read_raw_data_and_preprocess, filter_data_dict_with_var, get_openAI_model, gen_prompt_template_without_rag, eval_generated_ans, prepare_and_embed_documents, gen_prompt_template_with_rag, set_openAI_key_and_base, pretty_print_res_of_ranker, write_demo_knowledge, generate_with_rag
from utils import *
import logging
import sys

def chat_with_openai(data_dict, ground_ans: str="WALKING", contrast_ans: str="STANDING", answer_num: int=10, api_base: bool=True, model: str=MODEL["gpt3.5"], retrive=False, print_prompt=True):
    """
    return:
    results: list of str
    """
    results = []
    client = get_openAI_model(api_base, model)
    for i in range(answer_num):
        prompt = gen_prompt_template_without_rag(data_dict, ground_ans, contrast_ans, i)
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
    data_dict = read_raw_csi(subcarrier_dim=20, frames_num=20, frame_downsample=1)
    # filtered_data_dict = filter_data_dict_with_var(data_dict, thred=0.5, filter_by="body_acc", print_log=False)
    # data_dict = filtered_data_dict

    ground_ans = "no_person"
    contrast_ans = "have_person"

    start_time = time.perf_counter()
    # without retrieval:
    # ans = chat_with_openai(data_dict, ground_ans=ground_ans, contrast_ans=contrast_ans, answer_num=42, api_base=False, model=MODEL["gpt3.5"], retrive=False, print_prompt=False)
    # print(eval_generated_ans(ans, ground_ans, contrast_ans))

    # with retrieval:
    # 首先，准备好document_store并写入:
    KB_paths = [
        "domain-knowledge/dataset_introduction.txt",
        # "domain-knowledge/getPDF.pdf",
        "domain-knowledge/have_person.txt",
        "domain-knowledge/no_person.txt",
        "domain-knowledge/Value_description.txt",
        "domain-knowledge/Accurate_value.txt"
        # "domain-knowledge/None.txt"
    ]
    ans = generate_with_rag(ground_ans, contrast_ans, KB_paths, data_dict, num_samples=42)
    # with open(f"output_{ground_ans}.txt", "w") as file:
    #     # 将变量的内容写入文件
    #     for line in ans:
    #         file.write(line + "\n")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"总共耗时{elapsed_time}秒")
    ##print(eval_generated_ans(ans, ground_ans, contrast_ans)) ### 最好换成人工评估，不仅仅是看最终结果正确性，还需要关注大模型的分析能力，其是否表现出对物理规律的认识能力

     




    
