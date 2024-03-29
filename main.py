from imports import *
from utils import read_raw_data_and_preprocess, filter_data_dict_with_var, get_openAI_model, gen_prompt_template, eval_generated_ans


def chat_with_openai(data_dict, ground_ans: str="WALKING", contrast_ans: str="STANDING", answer_num: int=10, api_base: bool=True, model: str=MODEL["gpt3.5"], retrive=False, print_prompt=True):
    """
    return:
    results: list of str
    """
    results = []
    client = get_openAI_model(api_base, model)
    for i in range(answer_num):
        prompt = gen_prompt_template(data_dict, ground_ans, contrast_ans, i, retrive)
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
    ans = chat_with_openai(data_dict, ground_ans=ground_ans, contrast_ans=contrast_ans, answer_num=10, api_base=True, model=MODEL["gpt3.5"], retrive=False, print_prompt=False)
    print(eval_generated_ans(ans, ground_ans, contrast_ans))




