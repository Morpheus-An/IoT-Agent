from imports import *

def format_number(number):
    # 将数值转换为字符串
    number_str = str(number)
    # 将字符串中的每个字符用空格分隔开
    # formatted_str = ' '.join(number_str)
    # # 去掉小数点两边的空格
    # formatted_str = formatted_str.replace(' . ', '.')
    # return formatted_str
    return number_str

def gen_content4retrive_domain(args, task_type, data_des=""):
    if args.task_type != "imu_HAR":
        return content4retrieve_domain[task_type] + data_des
    else:
        if args.cls_num == 2:
            return content4retrieve_domain[task_type]["2cls"] + data_des
        elif args.cls_num > 2:
            return content4retrieve_domain[task_type]["mcls"] + data_des
        else:
            raise ValueError("The number of classes should be greater than 2.")

def Role_Definition(args):
    """input: Task_Descriping(str), Preprocessed_Data(str), model(str)
    output: role_definition(str)"""
    return Role_des[args.task_type]

def gen_prompt_with_rag_ECG(args, data_dict, is_Pos=True, i: int=1):
    N_signals = data_dict["N_signals"][-i]
    N_spectral_centroid = data_dict["frequency_features"]["N_fe"]["spectral_centroid"][-i]
    N_max_psd = data_dict["frequency_features"]["N_fe"]["max_psd"][-i]


    V_signals = data_dict["V_signals"][-i]
    V_spectral_centroid = data_dict["frequency_features"]["V_fe"]["spectral_centroid"][-i]
    V_max_psd = data_dict["frequency_features"]["V_fe"]["max_psd"][-i]

    N_signals_str = ", ".join([f"{x[0]}mv" for x in N_signals])
    V_signals_str = ", ".join([f"{x[0]}mv" for x in V_signals])
    N_signals_demo = data_dict["N_signals"][-i-1]
    N_spectral_centroid_demo = data_dict["frequency_features"]["N_fe"]["spectral_centroid"][-i-1]
    N_max_psd_demo = data_dict["frequency_features"]["N_fe"]["max_psd"][-i-1]
    


    V_signals_demo = data_dict["V_signals"][-i-1]
    V_spectral_centroid_demo = data_dict["frequency_features"]["V_fe"]["spectral_centroid"][-i-1]
    V_max_psd_demo = data_dict["frequency_features"]["V_fe"]["max_psd"][-i-1]

    N_signals_demo_str = ", ".join([f"{x[0]}mv" for x in N_signals_demo])
    V_signals_demo_str = ", ".join([f"{x[0]}mv" for x in V_signals_demo])


    # print(N_signals_str)
    prompt = "Objective:\n{{ query }}"
    # prompt = """
# QUESTION: {{ query }}"""
    prompt += """\nExpert knowledge about Sensor data:
You will receive data from ECG sensors. Here's how to interpret this data:

The ECG data is collected from a patient's heart. The data consists of a series of electrical signals that represent the heart's electrical activity. The signals are measured in millivolts (mV) and are recorded over a period of time at the sampling frequency of 60Hz. This means there is an interval of 0.017 seconds between the two voltage values.  The data is divided into two categories: normal heartbeats (N) and ventricular ectopic beats (V). The normal heartbeats represent the regular electrical activity of the heart, while the ventricular ectopic beats represent abnormal electrical activity. The data is collected using a single-channel ECG device."""
    prompt += """{% for domain_doc in documents_domain %}
    {{ domain_doc.content }}
{% endfor %}
You can analyze whether the heartbeat is normal by considering a combination of factors such as the amplitude of peaks or valleys appearing in the electrocardiogram (ECG) time series, the time intervals between adjacent peaks or valleys, and the fluctuations in voltage values within the ECG data."""

    prompt += f"""\n
ECG data (electrocardiogram data) is the data that records the electrical activity of the heart. This data is typically presented in the form of a time series, reflecting the electrophysiological activity of the heart.

Pay attention to the following examples. Learn to utilize the features of ECG data to assist in your decision-making.
EXAMPLE1:
THE GIVEN ECG DATA:
{N_signals_str}
Features:
Spectral centroid: {N_spectral_centroid}
Max psd: {N_max_psd}
Ground Answer: Normal heartbeat (N)

EXAMPLE2:
THE GIVEN ECG DATA:
{V_signals_str}
Features:
Spectral centroid: {V_spectral_centroid}
Max psd: {V_max_psd}
Ground answer: Premature ventricular contraction (V)

Response Format:
Reasoning: Provide a comprehensive analysis of the sensor data.
Summary: Conclude with a brief summary of your findings.

Now give your reponse according to the following sensor data (Hint: learn to utilize the features of ECG data to assist in your decision-making):
"""
    if is_Pos:
        data_des = f"""Sensor data:
{N_signals_demo_str}
Features:
Spectral_centroid: {N_spectral_centroid_demo}
Max psd: {N_max_psd_demo}
"""
        prompt += data_des
    else:
        data_des = f"""Sensor data:
{V_signals_demo_str}
Features:
Spectral centroid: {V_spectral_centroid_demo}
Max psd: {V_max_psd_demo}
"""
        prompt += data_des
#     prompt += """
# Please analyze the data step by step to explain what it reflects, and then provide your final answer based on your analysis: "Is it a Normal heartbeat(N) or Premature ventricular contraction beat(V)?"
# ANALYSIS:
# ANSWER:
# """
    prompt += """Is it a Normal heartbeat(N) or not(abnormal)?\nReasoning:\nSummary:"""
    return prompt, data_des
def gen_prompt_template_with_rag_imu(args, label2ids, data_dict, ground_ans: str="WALKING", contract_ans: str="STANDING", i: int=0, candidates=None): 

    def create_data_des(i, is_ground=True, candidate=None):
        if is_ground:
            target_cls = ground_ans 
        else:
            i = 0
            if args.cls_num == 2:
                target_cls = contract_ans
            elif args.cls_num > 2:
                assert(candidate is not None)
                target_cls = candidate
            else:
                raise ValueError("The number of classes should be greater than 2.")

        # pdb.set_trace()
        acc_x = data_dict[label2ids[target_cls]]["total_acc"][i][0]
        acc_y = data_dict[label2ids[target_cls]]["total_acc"][i][1]
        acc_z = data_dict[label2ids[target_cls]]["total_acc"][i][2]
        gyr_x = data_dict[label2ids[target_cls]]["body_gyro"][i][0]
        gyr_y = data_dict[label2ids[target_cls]]["body_gyro"][i][1]
        gyr_z = data_dict[label2ids[target_cls]]["body_gyro"][i][2] 
        acc_x_str = ", ".join([f"{x}" for x in acc_x])
        acc_y_str = ", ".join([f"{x}" for x in acc_y])
        acc_z_str = ", ".join([f"{x}" for x in acc_z])
        gyr_x_str = ", ".join([f"{x}" for x in gyr_x])
        gyr_y_str = ", ".join([f"{x}" for x in gyr_y])
        gyr_z_str = ", ".join([f"{x}" for x in gyr_z])
        data_des = f"""
1. Triaxial acceleration signal: 
X-axis: {acc_x_str} 
Y-axis: {acc_y_str} 
Z-axis: {acc_z_str} 
2. Triaxial angular velocity signal: 
X-axis: {gyr_x_str} 
Y-axis: {gyr_y_str} 
Z-axis: {gyr_z_str}
"""
        return data_des
    if args.cls_num == 2:
        data_des = create_data_des(i)
        # demo_grd_data_des = create_data_des(i+1)
        # demo_con_data_des = create_data_des(i, is_ground=False)
       
    #     prompt = f"""
    # THE GIVEN DATA: 
    # {data_des}
    # """
    #     prompt += """QUESTION:
    # {{ query }}
    # """
    #     prompt += f"""[{ground_ans}, {contract_ans}]
    # ANALYSIS:
    # ANSWER:
    # """
        prompt = """Objective:
{{ query }}"""
        prompt += f"""[{ground_ans}, {contract_ans}]"""
        prompt += f"""Sensor Data and Expert Knowledge:
You will receive data from various sensors. Here's how to interpret this data: 
Three-axis acceleration data reflects the acceleration of the device in three orthogonal directions, while three-axis angular velocity data reflects the rotational speed of the device in three orthogonal directions.

Response Format:
Reasoning: Provide a comprehensive analysis of the sensor data.
Summary: Conclude with a brief summary of your findings.

Now give your response according to the following sensor data:
Sensor data:
{data_des}
Reasoning:
Summary:"""
    elif args.cls_num > 2:
        assert(candidates is not None)
        demo_data_desciptions = {}
        data_des = create_data_des(i)
        candidates_str = ", ".join(candidates)
        for candidate in candidates:
            if candidate != ground_ans:
                demo_data_desciptions[candidate] = create_data_des(i, False, candidate)
            else:
                demo_data_desciptions[candidate] = create_data_des(i+1)
#         prompt = f"""
# THE GIVEN DATA: 
# {data_des}
# """
#         prompt += """QUESTION:
# {{ query }}
# """
#         prompt += f"""[{candidates_str}]]
# ANALYSIS:
# ANSWER:
# """
        prompt = """Objective:
{{ query }}"""
        prompt += f"""[{candidates_str}]\n"""
        prompt += f"""Sensor Data and Expert Knowledge:
You will receive data from various sensors. Here's how to interpret this data: 
Three-axis acceleration data reflects the acceleration of the device in three orthogonal directions, while three-axis angular velocity data reflects the rotational speed of the device in three orthogonal directions.

Response Format:
Reasoning: Provide a comprehensive analysis of the sensor data.
Summary: Conclude with a brief summary of your findings.

Now give your response according to the following sensor data:
Sensor data:
{data_des}
Reasoning:
Summary:"""
    else:
        raise ValueError("The number of classes should be greater than 2.")
    
    return prompt, data_des

# EXAMPLE1:
# {% for d in grd_demo %}{{ d.content }}{% endfor %}

# EXAMPLE2:
# {% for d in con_demo %}{{ d.content }}{% endfor %}

#  to the previous EXAMPLES and compare the signal data, the mean data, and the var data in the EXAMPLES with those in the question,
# EXAMPLE1:
# {{ document_demo_grd.content }}
# EXAMPLE2:
# {{ document_demo_con.content }}
#     return """
# Given the following information, answer the question.

# Context:
# {% for document in documents %}
#     {{ document.content }}
# {% endfor %}

# Question: {{question}}
# Answer:
# """

def gen_prompt_tamplate_with_rag_machine(args, data_dict, label_dict, target, i: int=0, ground_truth="Pos"):
    assert target in label_dict.keys()
    if target == "Cooler condition %":
        Cooler_condition_3_data = {}
        Cooler_condition_100_data = {}
        for key in data_dict.keys():
            Cooler_condition_3_data[key] = data_dict[key][np.where(label_dict["Cooler condition %"] == 3)]
            Cooler_condition_100_data[key] = data_dict[key][np.where(label_dict["Cooler condition %"] == 100)]
        print(f"Cooler_condition_3_data: {Cooler_condition_3_data['PS1'].shape}")
        print(f"Cooler_condition_100_data: {Cooler_condition_100_data['PS1'].shape}")
        TS_neg_demo = Cooler_condition_3_data["TS1"][-i-1]
        CP_neg_demo = Cooler_condition_3_data["CP"][-i-1]
        CE_neg_demo = Cooler_condition_3_data["CE"][-i-1]
        TS_pos_demo = Cooler_condition_100_data["TS1"][-i-1]
        CP_pos_demo = Cooler_condition_100_data["CP"][-i-1]
        CE_pos_demo = Cooler_condition_100_data["CE"][-i-1]
        TS_pos_demo_str = ", ".join([f"{x}°C" for x in TS_pos_demo])
        CP_pos_demo_str = ", ".join([f"{x}KW" for x in CP_pos_demo])
        CE_pos_demo_str = ", ".join([f"{x}%" for x in CE_pos_demo])
        TS_neg_demo_str = ", ".join([f"{x}°C" for x in TS_neg_demo])
        CP_neg_demo_str = ", ".join([f"{x}KW" for x in CP_neg_demo])
        CE_neg_demo_str = ", ".join([f"{x}%" for x in CE_neg_demo])

        TS_pos = Cooler_condition_100_data["TS1"][-i]
        CP_pos = Cooler_condition_100_data["CP"][-i]
        CE_pos = Cooler_condition_100_data["CE"][-i]
        TS_neg = Cooler_condition_3_data["TS1"][-i]
        CP_neg = Cooler_condition_3_data["CP"][-i]
        CE_neg = Cooler_condition_3_data["CE"][-i]
        Ts_pos_str = ", ".join([f"{x}°C" for x in TS_pos])
        CP_pos_str = ", ".join([f"{x}KW" for x in CP_pos])
        CE_pos_str = ", ".join([f"{x}%" for x in CE_pos])
        TS_neg_str = ", ".join([f"{x}°C" for x in TS_neg])
        CP_neg_str = ", ".join([f"{x}KW" for x in CP_neg])
        CE_neg_str = ", ".join([f"{x}%" for x in CE_neg])

        prompt = """Objective:\n{{ query }}"""
        # prompt = """
# QUESTION: {{ query }}"""
        prompt += """\nSensor Data and Expert Knowledge:
You will receive data from various sensors. Here's how to interpret this data:
Temperature Change Sequence reflects changes of temperature during working; Cooling Power Change Sequence implies changes of machine's cooling power during working; Cooling Efficiency Change Sequence reflects changes of machine's cooling efficiency.

Response Format:
Reasoning: Provide a comprehensive analysis of the sensor data.
Summary: Conclude with a brief summary of your findings.

Now give your response according to the following sensor data:\n"""
        if ground_truth == "Pos":
            data_des = f"""Sensor data:
1. Temperature Change Sequence:
{Ts_pos_str}
2. Cooling Power Change Sequence:
{CP_pos_str}
3. Cooling Efficiency Change Sequence:
{CE_pos_str}
"""
            prompt += data_des
        else:
            data_des = f"""Sensor data:
1. Temperature Change Sequence:
{TS_neg_str}
2. Cooling Power Change Sequence:
{CP_neg_str}
3. Cooling Efficiency Change Sequence:
{CE_neg_str}
"""
            prompt += data_des
        prompt += """Reasoning:
Summary:"""
    elif target == "":
        pass 
    return prompt, data_des




# def prompt_template_generation(Task_Description, Preprocessed_Data):
#     """template中的变量为：domain_ks, demonstrations, question"""
#     Role_definition = Role_Definition(Task_Description, Preprocessed_Data)
#     base_template = f"""{Role_definition}.\n The following data has been collected from the devices worn by the subjects:\n {Preprocessed_Data}.\n When answering questions, you can refer to the knowledge of the experts below, as well as some demonstrations:\n\n Experts: """
#     domain_knowledge = """
#     {% for domain_k in domain_ks %}
#         {{ domain_ks.content }}
#     {% endfor %}"""
#     demonstrations = """
#     {% for demonstration in demonstrations %}
#         {{ demonstration.content }}
#     {% endfor %}"""
#     question = """Question: {{ question }}\nAnswer:"""
#     prompt_template = base_template + domain_knowledge + demonstrations + question 
#     return prompt_template

def gen_prompt_template_without_rag(data_dict, ground_ans: str="WALKING", contrast_ans: str="STANDING", i: int=0):
    # TODO
    acc_x = data_dict[label2ids[ground_ans]]["total_acc"][i][0]
    acc_y = data_dict[label2ids[ground_ans]]["total_acc"][i][1]
    acc_z = data_dict[label2ids[ground_ans]]["total_acc"][i][2]
    gyr_x = data_dict[label2ids[ground_ans]]["body_gyro"][i][0]
    gyr_y = data_dict[label2ids[ground_ans]]["body_gyro"][i][1]
    gyr_z = data_dict[label2ids[ground_ans]]["body_gyro"][i][2] 
    demo_grd_acc_x = data_dict[label2ids[ground_ans]]["total_acc"][i+1][0]
    demo_grd_acc_y = data_dict[label2ids[ground_ans]]["total_acc"][i+1][1]
    demo_grd_acc_z = data_dict[label2ids[ground_ans]]["total_acc"][i+1][2]
    demo_grd_gyr_x = data_dict[label2ids[ground_ans]]["body_gyro"][i+1][0]
    demo_grd_gyr_y = data_dict[label2ids[ground_ans]]["body_gyro"][i+1][1]
    demo_grd_gyr_z = data_dict[label2ids[ground_ans]]["body_gyro"][i+1][2]
    demo_con_acc_x = data_dict[label2ids[contrast_ans]]["total_acc"][i][0]
    demo_con_acc_y = data_dict[label2ids[contrast_ans]]["total_acc"][i][1]
    demo_con_acc_z = data_dict[label2ids[contrast_ans]]["total_acc"][i][2]
    demo_con_gyr_x = data_dict[label2ids[contrast_ans]]["body_gyro"][i][0]
    demo_con_gyr_y = data_dict[label2ids[contrast_ans]]["body_gyro"][i][1]
    demo_con_gyr_z = data_dict[label2ids[contrast_ans]]["body_gyro"][i][2]
    acc_x_str = ", ".join([f"{x}g" for x in acc_x])
    acc_y_str = ", ".join([f"{x}g" for x in acc_y])
    acc_z_str = ", ".join([f"{x}g" for x in acc_z])
    gyr_x_str = ", ".join([f"{x}rad/s" for x in gyr_x])
    gyr_y_str = ", ".join([f"{x}rad/s" for x in gyr_y])
    gyr_z_str = ", ".join([f"{x}rad/s" for x in gyr_z])
    demo_grd_acc_x_str = ", ".join([f"{x}g" for x in demo_grd_acc_x])
    demo_grd_acc_y_str = ", ".join([f"{x}g" for x in demo_grd_acc_y])
    demo_grd_acc_z_str = ", ".join([f"{x}g" for x in demo_grd_acc_z])
    demo_grd_gyr_x_str = ", ".join([f"{x}rad/s" for x in demo_grd_gyr_x])
    demo_grd_gyr_y_str = ", ".join([f"{x}rad/s" for x in demo_grd_gyr_y])
    demo_grd_gyr_z_str = ", ".join([f"{x}rad/s" for x in demo_grd_gyr_z])
    demo_con_acc_x_str = ", ".join([f"{x}g" for x in demo_con_acc_x])
    demo_con_acc_y_str = ", ".join([f"{x}g" for x in demo_con_acc_y])
    demo_con_acc_z_str = ", ".join([f"{x}g" for x in demo_con_acc_z])
    demo_con_gyr_x_str = ", ".join([f"{x}rad/s" for x in demo_con_gyr_x])
    demo_con_gyr_y_str = ", ".join([f"{x}rad/s" for x in demo_con_gyr_y])
    demo_con_gyr_z_str = ", ".join([f"{x}rad/s" for x in demo_con_gyr_z])
    prompt = f"""{Role_Definition()}

EXPERT: 
1. Triaxial acceleration signal: 
The provided three-axis acceleration signals contain acceleration data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of 26 data samples, measured at a fixed time interval with a frequency of 10Hz(10 samples is collected per second). The unit is gravitational acceleration (g), equivalent to 9.8m/s^2. It's important to note that the measured acceleration is influenced by gravity, meaning the acceleration measurement along a certain axis will be affected by the vertical downward force of gravity. 
2. Triaxial angular velocity signal: 
The provided three-axis angular velocity signals contain angular velocity data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of 26 data samples, measured at a fixed time interval with a frequency of 10Hz. The unit is radians per second (rad/s). 
​
You need to comprehensively analyze the acceleration and angular velocity data on each axis. For each axis, you should analyze not only the magnitude and direction of each sampled data (the direction is determined by the positive or negative sign in the data) but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding the subject's motion status. For example, signals with greater fluctuations in sample data in the sequence often indicate the subject is engaging in more vigorous activities like WALKING, whereas signals with smaller fluctuations in sample data often indicate the subject is engaged in calmer activities like STANDING. 
​
EXAMPLE1: 
1. Triaxial acceleration signal: 
X-axis: {demo_grd_acc_x_str} 
Y-axis: {demo_grd_acc_y_str} 
Z-axis: {demo_grd_acc_z_str} 
X-axis-mean={np.around(np.mean(demo_grd_acc_x), 3)}, X-axis-var={np.around(np.var(demo_grd_acc_x), 3)} 
Y-axis-mean={np.around(np.mean(demo_grd_acc_y), 3)}, Y-axis-var={np.around(np.var(demo_grd_acc_y), 3)} 
Z-axis-mean={np.around(np.mean(demo_grd_acc_z), 3)}, Z-axis-var={np.around(np.var(demo_grd_acc_z), 3)} 
2. Triaxial angular velocity signal: 
X-axis: {demo_grd_gyr_x_str} 
Y-axis: {demo_grd_gyr_y_str} 
Z-axis: {demo_grd_gyr_z_str} 
X-axis-mean={np.around(np.mean(demo_grd_gyr_x), 3)}, X-axis-var={np.around(np.var(demo_grd_gyr_x), 3)} 
Y-axis-mean={np.around(np.mean(demo_grd_gyr_y), 3)}, Y-axis-var={np.around(np.var(demo_grd_gyr_y), 3)} 
Z-axis-mean={np.around(np.mean(demo_grd_gyr_z), 3)}, Z-axis-var={np.around(np.var(demo_grd_gyr_z), 3)} 
ANSWER: {ground_ans} 
​
EXAMPLE2: 
1. Triaxial acceleration signal: 
X-axis: {demo_con_acc_x_str} 
Y-axis: {demo_con_acc_y_str} 
Z-axis: {demo_con_acc_z_str} 
X-axis-mean={np.around(np.mean(demo_con_acc_x), 3)}, X-axis-var={np.around(np.var(demo_con_acc_x), 3)} 
Y-axis-mean={np.around(np.mean(demo_con_acc_y), 3)}, Y-axis-var={np.around(np.var(demo_con_acc_y), 3)} 
Z-axis-mean={np.around(np.mean(demo_con_acc_z), 3)}, Z-axis-var={np.around(np.var(demo_con_acc_z), 3)} 
2. Triaxial angular velocity signal: 
X-axis: {demo_con_gyr_x_str} 
Y-axis: {demo_con_gyr_y_str} 
Z-axis: {demo_con_gyr_z_str} 
X-axis-mean={np.around(np.mean(demo_con_gyr_x), 3)}, X-axis-var={np.around(np.var(demo_con_gyr_x), 3)} 
Y-axis-mean={np.around(np.mean(demo_con_gyr_y), 3)}, Y-axis-var={np.around(np.var(demo_con_gyr_y), 3)} 
Z-axis-mean={np.around(np.mean(demo_con_gyr_z), 3)}, Z-axis-var={np.around(np.var(demo_con_gyr_z), 3)} 
ANSWER: {contrast_ans} 
​
​
QUESTION: Based on the given data, choose the activity that the subject is most likely to be performing from the following two options: 
{ground_ans} 
{contrast_ans} 
Before answering your question, you must refer to the previous examples and compare the signal data, the mean data, and the var data in the examples with those in the question, in order to help you make a clear choice. 
​
​
THE GIVEN DATA: 
1. Triaxial acceleration signal: 
X-axis: {acc_x_str} 
Y-axis: {acc_y_str} 
Z-axis: {acc_z_str} 
X-axis-mean={np.around(np.mean(acc_x), 3)}, X-axis-var={np.around(np.var(acc_x), 3)} 
Y-axis-mean={np.around(np.mean(acc_y), 3)}, Y-axis-var={np.around(np.var(acc_y), 3)} 
Z-axis-mean={np.around(np.mean(acc_z), 3)}, Z-axis-var={np.around(np.var(acc_z), 3)} 
2. Triaxial angular velocity signal: 
X-axis: {gyr_x_str} 
Y-axis: {gyr_y_str} 
Z-axis: {gyr_z_str} 
X-axis-mean={np.around(np.mean(gyr_x), 3)}, X-axis-var={np.around(np.var(gyr_x), 3)} 
Y-axis-mean={np.around(np.mean(gyr_y), 3)}, Y-axis-var={np.around(np.var(gyr_y), 3)} 
Z-axis-mean={np.around(np.mean(gyr_z), 3)}, Z-axis-var={np.around(np.var(gyr_z), 3)} 
ANSWER:""" 
    return prompt


def generate_prompt_template(args, data_dict, label_dict, target, i:int=0, ground_truth:str="Pos", **kwargs):
    if args.task_type == "imu_HAR":
        pass 
    elif args.task_type == "machine_detection":
        return gen_prompt_tamplate_with_rag_machine(args, data_dict, label_dict, target, i, ground_truth)
    elif args.task_type == "ecg_detection":
        pass
    elif args.task_type == "wifi_localization":
        pass
    elif args.task_type == "wifi_occupancy":    
        pass
