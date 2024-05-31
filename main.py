from imports import *
# from utils import read_raw_data_and_preprocess, filter_data_dict_with_var, get_openAI_model, gen_prompt_template_without_rag, eval_generated_ans, prepare_and_embed_documents, gen_prompt_template_with_rag, set_openAI_key_and_base, pretty_print_res_of_ranker, write_demo_knowledge, generate_with_rag
from utils import *
from dataset import *

def chat_with_openai(data_dict, K, answer_num: int=10, api_base: bool=True, model: str=MODEL["gpt3.5"], retrive=False, print_prompt=True):
    """
    return:
    results: list of str
    """
    results = []
    client = get_openAI_model(api_base, model)
    for i in range(answer_num):
        prompt = gen_prompt_template_without_rag(data_dict, K, i)
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
        
def mse(data1, data2):
    """
    计算两个二维数据集之间的均方误差（MSE）

    参数：
    - data1: 第一个二维数据集，numpy数组
    - data2: 第二个二维数据集，numpy数组

    返回值：
    - mse_value: 均方误差值
    """
    # 确保两个数据集形状相同
    assert data1.shape == data2.shape, "数据集形状不一致"

    # 计算差值
    diff = data1 - data2

    # 计算平方差值
    squared_diff = np.square(diff)

    # 计算均方误差
    mse_value = np.mean(squared_diff)

    return mse_value

if __name__ == "__main__":
    ntuiot_dataset = NTUIoTRSSI_Dataset('rssi_position_data.txt')
    ntuiot_dataset.raw_set = ntuiot_dataset.decrease_dataset(10)
    ntuiot_dataset.filter_outliers(quantile_threshold=0.1)
    data_dict = ntuiot_dataset.split_train_test(train_ratio=0.4, val_ratio=0.2, random_state=10)

    np.savetxt("domain-knowledge/database_rssi.txt", data_dict['database_rssi'])
    np.savetxt("domain-knowledge/database_position.txt", data_dict['database_position'])

    start_time = time.perf_counter()
    # without retrieval:
    # ans = chat_with_openai(data_dict, K=3, answer_num=len(data_dict['val_rssi']), api_base=False, model=MODEL["gpt3.5"], retrive=False, print_prompt=False)
    # error = []
    # i = 0
    # for an in ans:
    #     coordinates= np.array(extract_coordinates(an))
    #     gt = np.array(data_dict['val_position'][i, :])
    #     if gt.shape == coordinates.shape:
    #         error.append(mse(coordinates,gt))
    #
    #     i = i+1
    # error_mean = np.mean(error)
    # print(f"mse is {error_mean}")
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"总共耗时{elapsed_time}秒")

    # # with retrieval:
    # # 首先，准备好document_store并写入:
    KB_paths = [
        "domain-knowledge/WKNN.txt",
        "domain-knowledge/Code for WKNN localization.txt",
        "domain-knowledge/Dataset introduction.txt",
        # "domain-knowledge/None.txt",
        # "domain-knowledge/database_rssi.txt",
        # "domain-knowledge/database_position.txt",

    ]
    ans = generate_with_rag(KB_paths, data_dict, K=3, num_samples=len(data_dict['val_rssi']))
    # with open(f"output_{ground_ans}.txt", "w") as file:
    #     # 将变量的内容写入文件
    #     for line in ans:
    #         file.write(line + "\n")
    error = []
    i = 0
    for an in ans:
        coordinates= np.array(extract_coordinates(an))
        gt = np.array(data_dict['val_position'][i, :])
        if gt.shape == coordinates.shape:
            error.append(mse(coordinates,gt))

        i = i+1
    error_mean = np.mean(error)
    print(f"mse is {error_mean}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"总共耗时{elapsed_time}秒")
    # ##print(eval_generated_ans(ans, ground_ans, contrast_ans)) ### 最好换成人工评估，不仅仅是看最终结果正确性，还需要关注大模型的分析能力，其是否表现出对物理规律的认识能力


#5: 4488
#10: 4646
     




    
