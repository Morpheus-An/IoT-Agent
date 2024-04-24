from imports import * 


def eval_by_gpt(ans, candidates, grd, con): 
    eval = []
    candidates_str = ", ".join(candidates)
    role_des = f"""You are an evaluator, please judge the attitude expressed in the given statement. You should provide one of the results from the candidates, and do not include any other content.
    candidates: [{candidates}]."""
    for an in ans:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": role_des},
                {"role": "user", "content": an},
            ]
        ).choices[0].message.content 
        eval.append(completion)
        print(completion)
    score = eval_generated_ans(eval, grd, con)
    print(score)
    return score
        
def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="thenlper/gte-large",
        meta_fields_to_embed=metadata_fields_to_embed,
        device=ComponentDevice.from_str("cuda:0"),
    )
    document_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)

    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    return indexing_pipeline

def prepare_and_embed_documents(document_store, source_paths: list[str], metadata_fields_to_embed=None, meta_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None, splitter_kwards: dict=None, draw: str=None, device: str="cuda:0"):
    # assert ((metadata_fields_to_embed is None and meta_data is None) or (metadata_fields_to_embed is not None and meta_data is not None)) 
    if type(meta_data) == list: 
        assert len(meta_data) == len(source_paths)

    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    pdf_converter = PyPDFToDocument()
    markdown_converter = MarkdownToDocument()
    document_jointer = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    if splitter_kwards is None:
        document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
    else:
        document_splitter = DocumentSplitter(**splitter_kwards)
    
    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDER_MODEL_LOCAL, meta_fields_to_embed=metadata_fields_to_embed, device=ComponentDevice.from_str(device)) 

    document_writer = DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE)

    preprocess_pipeline = Pipeline()
    preprocess_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocess_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocess_pipeline.add_component(instance=pdf_converter, name="pdf_converter")
    preprocess_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocess_pipeline.add_component(instance=document_jointer, name="document_jointer")
    preprocess_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocess_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocess_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocess_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocess_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocess_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
    preprocess_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocess_pipeline.connect("text_file_converter", "document_jointer")
    preprocess_pipeline.connect("pdf_converter", "document_jointer")
    preprocess_pipeline.connect("markdown_converter", "document_jointer")
    preprocess_pipeline.connect("document_jointer", "document_cleaner")
    preprocess_pipeline.connect("document_cleaner", "document_splitter")
    preprocess_pipeline.connect("document_splitter", "document_embedder")
    preprocess_pipeline.connect("document_embedder", "document_writer")
    if draw is not None:
        preprocess_pipeline.draw(draw)
    preprocess_pipeline.run(
        {
            "file_type_router": {
                "sources": source_paths
            },
            "text_file_converter": {
                "meta": meta_data
            },
            "pdf_converter": {
                "meta": meta_data 
            },
            "markdown_converter": {
                "meta": meta_data
            }
        }
    )
    return document_store

def gen_prompt_template_with_rag(data_dict, ground_ans: str="WALKING", contract_ans: str="STANDING", i: int=0):
    acc_x = data_dict[label2ids[ground_ans]]["total_acc"][i][0]
    acc_y = data_dict[label2ids[ground_ans]]["total_acc"][i][1]
    acc_z = data_dict[label2ids[ground_ans]]["total_acc"][i][2]
    gyr_x = data_dict[label2ids[ground_ans]]["body_gyro"][i][0]
    gyr_y = data_dict[label2ids[ground_ans]]["body_gyro"][i][1]
    gyr_z = data_dict[label2ids[ground_ans]]["body_gyro"][i][2] 
    acc_x_str = ", ".join([f"{x}g" for x in acc_x])
    acc_y_str = ", ".join([f"{x}g" for x in acc_y])
    acc_z_str = ", ".join([f"{x}g" for x in acc_z])
    gyr_x_str = ", ".join([f"{x}rad/s" for x in gyr_x])
    gyr_y_str = ", ".join([f"{x}rad/s" for x in gyr_y])
    gyr_z_str = ", ".join([f"{x}rad/s" for x in gyr_z])
    data_des = f"""1. Triaxial acceleration signal: 
X-axis: {acc_x_str} 
Y-axis: {acc_y_str} 
Z-axis: {acc_z_str} 
X-axis-mean={np.around(np.mean(acc_x), 3)}g, X-axis-var={np.around(np.var(acc_x), 3)} 
Y-axis-mean={np.around(np.mean(acc_y), 3)}g, Y-axis-var={np.around(np.var(acc_y), 3)} 
Z-axis-mean={np.around(np.mean(acc_z), 3)}g, Z-axis-var={np.around(np.var(acc_z), 3)} 
2. Triaxial angular velocity signal: 
X-axis: {gyr_x_str} 
Y-axis: {gyr_y_str} 
Z-axis: {gyr_z_str} 
X-axis-mean={np.around(np.mean(gyr_x), 3)}rad/s, X-axis-var={np.around(np.var(gyr_x), 3)} 
Y-axis-mean={np.around(np.mean(gyr_y), 3)}rad/s, Y-axis-var={np.around(np.var(gyr_y), 3)} 
Z-axis-mean={np.around(np.mean(gyr_z), 3)}rad/s, Z-axis-var={np.around(np.var(gyr_z), 3)}"""
    
    prompt = f"""{Role_Definition()}

EXPERT:
1. Triaxial acceleration signal: 
The provided three-axis acceleration signals contain acceleration data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of some data samples, measured at a fixed time interval with a frequency of 10Hz(10 samples is collected per second). The unit is gravitational acceleration (g), equivalent to 9.8m/s^2. It's important to note that the measured acceleration is influenced by gravity, meaning the acceleration measurement along a certain axis will be affected by the vertical downward force of gravity. 
2. Triaxial angular velocity signal: 
The provided three-axis angular velocity signals contain angular velocity data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of some data samples, measured at a fixed time interval with a frequency of 10Hz. The unit is radians per second (rad/s).
3. Other domain knowledge:
"""
    prompt += """
{% for domain_doc in documents_domain %}{{ domain_doc.content }}{% endfor %}


You need to comprehensively analyze the acceleration and angular velocity data on each axis. For each axis, you should analyze not only the magnitude and direction of each sampled data (the direction is determined by the positive or negative sign in the data) but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding the subject's motion status.
For example, when the signal consistently shows significant fluctuations, it indicates that the person may be engaged in continuous activities, such as WALKING_UPSTAIRS. On the other hand, when the signal consistently displays fewer fluctuations, it suggests that the person may be in a relatively calm state, such as LAYING. However, if there are differing patterns between segments of the signal sequence, and there are notable changes, particularly on certain axes during specific periods, it suggests that the person may be transitioning between activity states, such as in the case of LIE-TO-SIT.


QUESTION: {{ query }}
"""
    prompt += f"""
Before answering your question, you must refer to the EXPERT and make an analysis step by step.
​
​
THE GIVEN DATA: 
{data_des}
ANALYSIS:
ANSWER:""" 
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
def read_machine_data(sample_step=100):
    # 读入profile_labels文件，其中每行包括五列数据，数据之间由空格隔开
    labels = np.loadtxt('/home/ant/RAG/data/machine_detect/profile.txt', dtype=int)
    labels.shape
    label_dict = {
        "Cooler condition %": labels[:, 0],
        "Valve condition %": labels[:, 1],
        "Internal pump leakage": labels[:, 2],
        "Hydraulic accumulator": labels[:, 3],
        "Stable flag": labels[:, 4]
    } # 分别是五列数据表示的含义
    # print(label_dict["Cooler condition %"][:10])
    # 读取PS1-PS6.txt的数据：
    PS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS1.txt')
    # .shape
    PS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS2.txt')
    PS3 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS3.txt')
    PS4 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS4.txt')
    PS5 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS5.txt')
    PS6 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS6.txt')

    # 读取EPS1.txt的数据
    EPS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/EPS1.txt')
    # print(EPS1.shape)
    # EPS1[:10]
    # 读取FS1/FS2.txt的数据
    FS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/FS1.txt')
    FS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/FS2.txt')
    # print(FS1.shape)
    # FS1[:10]
    # 读取TS1-4.txt的数据
    TS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS1.txt')
    TS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS2.txt')
    TS3 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS3.txt')
    TS4 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS4.txt')
    # print(TS1.shape)
    # TS1[:10]
    # 读取VS1.txt的数据
    VS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/VS1.txt')
    # print(VS1.shape)
    # VS1[:10]
    SE = np.loadtxt('/home/ant/RAG/data/machine_detect/SE.txt')
    # print(SE.shape)   
    # SE[:10]
    CE = np.loadtxt('/home/ant/RAG/data/machine_detect/CE.txt')
    # print(CE.shape)
    # CE[:10]
    CP = np.loadtxt('/home/ant/RAG/data/machine_detect/CP.txt')
    # print(CP.shape)
    # CP[:10]
    print(PS1.shape)
    data_dict = {
        "PS1": PS1[:,::sample_step],
        "PS2": PS2[:,::sample_step],
        "PS3": PS3[:,::sample_step],
        "PS4": PS4[:,::sample_step],
        "PS5": PS5[:,::sample_step],
        "PS6": PS6[:,::sample_step],
        "EPS1": EPS1[:,::sample_step],
        "FS1": FS1[:,::sample_step//10],
        "FS2": FS2[:,::sample_step//10],
        "TS1": TS1,
        "TS2": TS2,
        "TS3": TS3,
        "TS4": TS4,
        "VS1": VS1,
        "SE": SE,
        "CE": CE,
        "CP": CP
    }
    print(f"machine_data loaded")
    return data_dict, label_dict
def gen_content4retrive_domain(data_des=""):
    return """
The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit which are connected via the oil tank [1], [2]. The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied.
    
Attributes are sensor data (all numeric and continuous) from measurements taken at the same point in time, respectively, of a hydraulic test rig's working cycle.

Temepurature sensors (TS) measure the temperature of the oil at different points in the hydraulic system. The sensors are named TS1, TS2, TS3, and TS4. The temperature is measured in degrees Celsius.
Efficiency factor sensors (SE) measure the efficiency of the cooler. The efficiency is calculated from the ratio of the actual cooling power to the ideal cooling power. The sensors are named SE. The efficiency factor is given in percentage.
Cooling power sensors (CP) measure the cooling power in kilowatts. The sensors are named CP1 and CP2. The cooling power is measured in kilowatts.

For each sensor, we collected 60 data points over a period of 60 seconds at a monitoring frequency of 1Hz (measuring sensor data once every second), forming a time series of length 60. We measured the following sequences using temperature sensors, Cooling power sensors, and Cooling efficiency sensors:

1. **Temperature Change Sequence**: Reflects the machine's temperature variation over 60 seconds, in degrees Celsius. By analyzing this sequence, you can assess whether the cooling equipment is operating normally. Typically, when the cooling system is functioning well, the machine's temperature is relatively low and does not fluctuate too significantly. If the temperature consistently remains at a high degrees Celsius or fluctuates significantly, it may indicate an abnormal issue with the cooling equipment.

2. **Cooling Power Change Sequence**: Reflects the variation in the cooling power of the machine's cooling equipment over 60 seconds, in kilowatts (KW). By analyzing this sequence, you can determine if the cooling equipment is operating normally. Generally, when the cooling system is functioning properly, the cooling power is relatively high and remains relatively stable throughout the period. If the power consistently stays low, it may suggest an abnormal issue with the cooling equipment.

3. **Cooling Efficiency Change Sequence**: Reflects the variation in the efficiency of the machine's cooling equipment over 60 seconds, in percentage (%). By analyzing this sequence, you can judge if the cooling equipment is operating normally. Typically, when the cooling system is working well, the cooling efficiency is relatively high, otherwise, it indicates that there may be an abnormal issue with the cooling equipment.

Please analyze the data step by step to explain what it reflects, and then provide your final answer based on your analysis: "Is the machine's cooling system functioning properly?"
""" + data_des 

def gen_prompt_tamplate_with_rag_machine(data_dict, label_dict, target, i: int=0, ground_truth="Pos"):
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
        prompt = f"""{Role_Definition()}

SENSOR DATA:
For each sensor, we collected 60 data points over a period of 60 seconds at a monitoring frequency of 1Hz (measuring sensor data once every second), forming a time series of length 60. We measured the following sequences using temperature sensors, Cooling power sensors, and Cooling efficiency sensors:

1. **Temperature Change Sequence**: Reflects the machine's temperature variation over 60 seconds, in degrees Celsius. By analyzing this sequence, you can assess whether the cooling equipment is operating normally. Typically, when the cooling system is functioning well, the machine's temperature is relatively low and does not fluctuate too significantly. If the temperature consistently remains at a high degrees Celsius or fluctuates significantly, it may indicate an abnormal issue with the cooling equipment.

2. **Cooling Power Change Sequence**: Reflects the variation in the cooling power of the machine's cooling equipment over 60 seconds, in kilowatts (KW). By analyzing this sequence, you can determine if the cooling equipment is operating normally. Generally, when the cooling system is functioning properly, the cooling power is relatively high and remains relatively stable throughout the period. If the power consistently stays low, it may suggest an abnormal issue with the cooling equipment.

3. **Cooling Efficiency Change Sequence**: Reflects the variation in the efficiency of the machine's cooling equipment over 60 seconds, in percentage (%). By analyzing this sequence, you can judge if the cooling equipment is operating normally. Typically, when the cooling system is working well, the cooling efficiency is relatively high, otherwise, it indicates that there may be an abnormal issue with the cooling equipment."""
        prompt += """
EXPERT: 
{% for domain_doc in documents_domain %}
    {{ domain_doc.content }}
{% endfor %}



"""
        prompt += f"""
EXAMPLE1:
1. Temperature Change Sequence:
{TS_neg_demo_str}
2. Cooling Power Change Sequence:
{CP_neg_demo_str}
3. Cooling Efficiency Change Sequence:
{CE_neg_demo_str}
ANSWER: not operating normally. 
EXAMPLE2:
1. Temperature Change Sequence:
{TS_pos_demo_str}
2. Cooling Power Change Sequence:
{CP_pos_demo_str}
3. Cooling Efficiency Change Sequence:
{CE_pos_demo_str}
ANSWER: operating normally."""
        prompt += """
QUESTION: {{ query }}"""
        if ground_truth == "Pos":
            data_des = f"""
THE GIVEN DATA:
1. Temperature Change Sequence:
{Ts_pos_str}
2. Cooling Power Change Sequence:
{CP_pos_str}
3. Cooling Efficiency Change Sequence:
{CE_pos_str}
"""
            prompt += data_des
        else:
            data_des = f"""
THE GIVEN DATA:
1. Temperature Change Sequence:
{TS_neg_str}
2. Cooling Power Change Sequence:
{CP_neg_str}
3. Cooling Efficiency Change Sequence:
{CE_neg_str}
"""
            prompt += data_des
        prompt += """
Please analyze the data step by step to explain what it reflects, and then provide your final answer based on your analysis: "Is the machine's cooling system functioning properly?"
ANALYSIS:
ANSWER:
"""
    elif target == "":
        pass 
    return prompt, data_des

    

def pretty_print_res_of_ranker(res):
    for doc in res["documents"]:
        print(doc.meta["file_path"], "\t", doc.score)
        print(doc.content)
        print("\n", "\n")

# 写入demo-knowledge：
def write_demo_knowledge(tgt_dir_path: str, data_dict, sample_num: int=5):
    file_paths = []
    # 为了不与test使用的demo重复，选用data_dict中的后面的数据作为范例知识
    for label_id in label2ids.values():
        for i in range(1, sample_num+1):
            acc_x = data_dict[label_id]["total_acc"][-i][0]
            acc_y = data_dict[label_id]["total_acc"][-i][1]
            acc_z = data_dict[label_id]["total_acc"][-i][2]
            gyr_x = data_dict[label_id]["body_gyro"][-i][0]
            gyr_y = data_dict[label_id]["body_gyro"][-i][1]
            gyr_z = data_dict[label_id]["body_gyro"][-i][2]
            acc_x_str = ", ".join([f"{x}g" for x in acc_x])
            acc_y_str = ", ".join([f"{x}g" for x in acc_y])
            acc_z_str = ", ".join([f"{x}g" for x in acc_z])
            gyr_x_str = ", ".join([f"{x}rad/s" for x in gyr_x])
            gyr_y_str = ", ".join([f"{x}rad/s" for x in gyr_y])
            gyr_z_str = ", ".join([f"{x}rad/s" for x in gyr_z])
            written_content = f"""1. Triaxial acceleration signal:
X-axis: {acc_x_str}
Y-axis: {acc_y_str}
Z-axis: {acc_z_str}
X-axis-mean={np.around(np.mean(acc_x), 3)}g, X-axis-var={np.around(np.var(acc_x), 3)}
Y-axis-mean={np.around(np.mean(acc_y), 3)}g, Y-axis-var={np.around(np.var(acc_y), 3)}
Z-axis-mean={np.around(np.mean(acc_z), 3)}g, Z-axis-var={np.around(np.var(acc_z), 3)}
2. Triaxial angular velocity signal:
X-axis: {gyr_x_str}
Y-axis: {gyr_y_str}
Z-axis: {gyr_z_str}
X-axis-mean={np.around(np.mean(gyr_x), 3)}rad/s, X-axis-var={np.around(np.var(gyr_x), 3)}
Y-axis-mean={np.around(np.mean(gyr_y), 3)}rad/s, Y-axis-var={np.around(np.var(gyr_y), 3)}
Z-axis-mean={np.around(np.mean(gyr_z), 3)}rad/s, Z-axis-var={np.around(np.var(gyr_z), 3)}
ANSWER: {id2labels[label_id]}"""
            file_path = tgt_dir_path + f"{id2labels[label_id]}_{i}.txt"
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.write(written_content)
                f.write("\n\n")
    return file_paths


        
def wikipedia_indexing(some_titles = ["Inertial measurement unit", ]):
    raw_docs = []
    for title in some_titles:
        page = wikipedia.page(title=title, auto_suggest=False)
        doc = Document(content=page.content, meta={"title": page.title, "url": page.url})
        raw_docs.append(doc)

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    indexing_pipeline = create_indexing_pipeline(document_store=document_store, metadata_fields_to_embed=["title"])
    indexing_pipeline.run({"cleaner": {"documents": raw_docs}})
    return document_store

def Role_Definition(Task_Description=None, Preprocessed_Data=None, model="chatgpt"):
    """input: Task_Descriping(str), Preprocessed_Data(str), model(str)
    output: role_definition(str)"""
#     return """You are an assistant sports scientist, specialized in analyzing sensor data to understand human movement and activity patterns. Your expertise in interpreting accelerometer sensor data makes you an expert in human activity recognition tasks. Your role is to assist users in determining the status of human activities by analyzing accelerometer data.
# Your training enables you to interpret and analyze the data collected by accelerometer sensors, thereby identifying different motion patterns. You understand the acceleration patterns generated by the human body in various activities and can determine the current activity status based on changes in the data.
# Your professional knowledge includes, but is not limited to:
# 1. Human Biomechanics: You understand the acceleration patterns generated by the human body in different activity modes and their relationship with specific activities.
# 2. Data Analysis and Pattern Recognition: You can utilize machine learning and pattern recognition techniques to analyze and process sensor data, accurately identifying human activities.
# 3. Exercise Physiology: You understand the physiological changes that occur in the human body during exercise, which can assist in activity recognition.
# As an assistant sports scientist, your task is to classify human activities based on the acceleration data you receive, helping users better understand and monitor their exercise activities."""
    return """As a seasoned machine evaluation expert with a profound understanding of hydraulic systems, you possess the following key abilities and knowledge:

- **System Comprehension**: You are well-versed in the inner workings of hydraulic systems, including the functions of key components such as coolers, pumps, valves, and accumulators, as well as their interplay.
- **Data Analysis**: You are adept at handling and analyzing complex datasets, employing advanced techniques like time series analysis, signal processing, and pattern recognition to uncover the underlying stories in the data.
- **Sensor Interpretation**: You have a deep understanding of various sensor data (e.g., pressure, temperature, flow, vibration) and can discern the operational status of machinery that these data represent.
- **Fault Diagnosis**: You are familiar with the types of malfunctions that can occur in hydraulic systems and can recognize the signs of these faults in sensor data, enabling you to monitor the machine's operating condition and diagnose potential issues.
- **System Synergy**: You understand how the different parts of a machine work together and how anomalies in one component can trigger a cascade effect throughout the system.

In the upcoming task of machine condition assessment, you will leverage your expertise and skills to conduct an in-depth analysis of the data collected by the hydraulic test rig. You will evaluate the operational status of the cooler and other critical components, providing professional insights and recommendations for machine performance optimization and preventive maintenance."""

def prompt_template_generation(Task_Description, Preprocessed_Data):
    """template中的变量为：domain_ks, demonstrations, question"""
    Role_definition = Role_Definition(Task_Description, Preprocessed_Data)
    base_template = f"""{Role_definition}.\n The following data has been collected from the devices worn by the subjects:\n {Preprocessed_Data}.\n When answering questions, you can refer to the knowledge of the experts below, as well as some demonstrations:\n\n Experts: """
    domain_knowledge = """
    {% for domain_k in domain_ks %}
        {{ domain_ks.content }}
    {% endfor %}"""
    demonstrations = """
    {% for demonstration in demonstrations %}
        {{ demonstration.content }}
    {% endfor %}"""
    question = """Question: {{ question }}\nAnswer:"""
    prompt_template = base_template + domain_knowledge + demonstrations + question 
    return prompt_template

def read_raw_data_and_preprocess(sample_step: int=5, raw_data_dir: str="/home/ant/RAG/data/IMU/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/", y_train_path: str="/home/ant/RAG/data/IMU/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"):
    """return :
    data_dict: dict[dict[list, list, list]]

    >>> data_dict[label_id]["body_acc"] = [[body_acc_x, body_acc_y, body_acc_z], ...]
    """
    signal_data_paths = {
        "body_acc_x_train_path" : raw_data_dir + "body_acc_x_train.txt",
        "body_acc_y_train_path" :  raw_data_dir + "body_acc_y_train.txt",
        "body_acc_z_train_path" :  raw_data_dir + "body_acc_z_train.txt",
        "body_gyro_x_train_path" :  raw_data_dir + "body_gyro_x_train.txt",
        "body_gyro_y_train_path" :  raw_data_dir + "body_gyro_y_train.txt",
        "body_gyro_z_train_path" :  raw_data_dir +  "body_gyro_z_train.txt",
        "total_acc_x_train_path" :  raw_data_dir + "total_acc_x_train.txt",
        "total_acc_y_train_path" :  raw_data_dir + "total_acc_y_train.txt", 
        "total_acc_z_train_path" :  raw_data_dir + "total_acc_z_train.txt",
    }
    signal_data = {}
    for signal_data_path in signal_data_paths.keys():
        with open(signal_data_paths[signal_data_path], "r") as f:
            signal_data[signal_data_path[:-5]] = np.array([list(map(float, line.split())) for line in f])
    with open(y_train_path, "r") as f:
        y_train = np.array([int(line) for line in f])
    print(Counter(y_train))
    data_dict: dict[dict[list, list, list]] = {}
    # 其中有6个key，分别代表六个活动类别，每个key中有三个list，分别代表三个传感器的数据

    for label_id in label2ids.values():
        data_dict[label_id] = {"body_acc": [], "body_gyro": [], "total_acc": []}

    for i in range(len(y_train)):
        data_dict[y_train[i]]["body_acc"].append([np.around(signal_data["body_acc_x_train"][i][::sample_step], 3), np.around(signal_data["body_acc_y_train"][i][::sample_step], 3), np.around(signal_data["body_acc_z_train"][i][::sample_step], 3)])

        data_dict[y_train[i]]["body_gyro"].append([np.around(signal_data["body_gyro_x_train"][i][::sample_step], 3), np.around(signal_data["body_gyro_y_train"][i][::sample_step], 3), np.around(signal_data["body_gyro_z_train"][i][::sample_step], 3)])

        data_dict[y_train[i]]["total_acc"].append([np.around(signal_data["total_acc_x_train"][i][::sample_step], 3), np.around(signal_data["total_acc_y_train"][i][::sample_step], 3), np.around(signal_data["total_acc_z_train"][i][::sample_step], 3)])
    return data_dict

def read_multicls_data_and_preprocess(labels, sample_step: int=50, raw_data_dir: str="/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/"):
    """return :
    data_dict: dict[dict[list, list, list]]

    >>> data_dict[label_id]["acc"] = [[acc_x, acc_y, acc_z], ...]
    """
    data_dict: dict[dict[list, list, list]] = {}
    # 其中有12个key，分别代表12个活动类别，每个key中有两个list，分别代表两个传感器的数据
    for label_id in id2labels.keys():
        data_dict[label_id] = {"total_acc": [], "body_gyro": []}
    for label in labels:
        exp, user, cls_id, begin, end = label
        acc_path = f"/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/acc_exp{exp:02d}_user{user:02d}.txt"
        gyr_path = f"/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/gyro_exp{exp:02d}_user{user:02d}.txt"
        acc_data = np.loadtxt(acc_path)
        gyr_data = np.loadtxt(gyr_path)
        if cls_id <= 6:
            raw_data_acc = acc_data[begin-1:end-1:sample_step]
            raw_data_gyr = gyr_data[begin-1:end-1:sample_step]
        else:
            raw_data_acc = acc_data[begin-1:end-1:sample_step//4]
            raw_data_gyr = gyr_data[begin-1:end-1:sample_step//4]
        data_dict[cls_id]["total_acc"].append([np.around(raw_data_acc[:, 0], 3), np.around(raw_data_acc[:, 1], 3), np.around(raw_data_acc[:, 2], 3)])

        data_dict[cls_id]["body_gyro"].append([np.around(raw_data_gyr[:, 0], 3), np.around(raw_data_gyr[:, 1], 3), np.around(raw_data_gyr[:, 2], 3)])
    for label_id in data_dict.keys():
        print(f"{id2labels[label_id]}: {len(data_dict[label_id]['total_acc'])}")

    return data_dict


def set_openAI_key_and_base(set_base=True, set_proxy=None):
    if set_proxy is not None:
        os.environ["http_proxy"] = PROXY
        os.environ["https_proxy"] = PROXY
    if set_base:
        os.environ["OPENAI_API_KEY"] = MY_API
        os.environ["OPENAI_BASE_URL"] = BASE_URL
        print("set OPENAI by my own key")
    else:
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        os.environ["OPENAI_API_KEY"] = TEACHER_API
        print("set OPENAI by teacher's key")
def get_openAI_model(api_base: bool=True,
                     model: str=MODEL["gpt3.5"]):
    set_openAI_key_and_base(api_base)
    if api_base:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def filter_data_dict_with_var(data_dict, thred: float=0.5, filter_by: str="body_acc", print_log: bool=True):
    """
    param:
    过滤掉方差百分数大于/小于thred的数据
    return:
    filtered_data_dict: dict[dict[list, list, list]]
    """
    var4cls = {
        label_id: {
            "x": [], 
            "y": [], 
            "z": []
        } for label_id in label2ids.values()
    }
    for label_id in label2ids.values():
        for i in range(len(data_dict[label_id][filter_by])):
            var4cls[label_id]["x"].append(np.var(data_dict[label_id][filter_by][i][0]))
            var4cls[label_id]["y"].append(np.var(data_dict[label_id][filter_by][i][1]))
            var4cls[label_id]["z"].append(np.var(data_dict[label_id][filter_by][i][2]))
    var4cls_sorted = {
        label_id: {
            "x": [], 
            "y": [], 
            "z": []
        } for label_id in label2ids.values()
    }
    for label_id in label2ids.values():
        var4cls_sorted[label_id]["x"] = sorted(var4cls[label_id]["x"])
        var4cls_sorted[label_id]["y"] = sorted(var4cls[label_id]["y"])
        var4cls_sorted[label_id]["z"] = sorted(var4cls[label_id]["z"])
        if print_log:
            print(f"{id2labels[label_id]} {filter_by}_x var {thred*100}% data is below {var4cls_sorted[label_id]['x'][int(len(var4cls_sorted[label_id]['x'])*thred)]}")
            print(f"{id2labels[label_id]} {filter_by}_y var {thred*100}% data is below {var4cls_sorted[label_id]['y'][int(len(var4cls_sorted[label_id]['y'])*thred)]}")
            print(f"{id2labels[label_id]} {filter_by}_z var {thred*100}% data is below {var4cls_sorted[label_id]['z'][int(len(var4cls_sorted[label_id]['z'])*thred)]}")
    # 过滤掉方差百分数大于/小于thred的数据 
    data_dict_filtered = {}
    for label_id in label2ids.values():
        data_dict_filtered[label_id] = {"body_acc": [], "body_gyro": [], "total_acc": []}
        for i in range(len(data_dict[label_id][filter_by])):
            if label_id >= 4:
                if np.var(data_dict[label_id][filter_by][i][0]) < var4cls_sorted[label_id]["x"][int(len(var4cls_sorted[label_id]["x"])*thred)] and np.var(data_dict[label_id][filter_by][i][1]) < var4cls_sorted[label_id]["y"][int(len(var4cls_sorted[label_id]["y"])*thred)] and np.var(data_dict[label_id][filter_by][i][2]) < var4cls_sorted[label_id]["z"][int(len(var4cls_sorted[label_id]["z"])*thred)]:
                    data_dict_filtered[label_id]["body_acc"].append(data_dict[label_id]["body_acc"][i])
                    data_dict_filtered[label_id]["body_gyro"].append(data_dict[label_id]["body_gyro"][i])
                    data_dict_filtered[label_id]["total_acc"].append(data_dict[label_id]["total_acc"][i])
            else:
                if np.var(data_dict[label_id][filter_by][i][0]) > var4cls_sorted[label_id]["x"][int(len(var4cls_sorted[label_id]["x"])*thred)] and np.var(data_dict[label_id][filter_by][i][1]) > var4cls_sorted[label_id]["y"][int(len(var4cls_sorted[label_id]["y"])*thred)] and np.var(data_dict[label_id][filter_by][i][2]) > var4cls_sorted[label_id]["z"][int(len(var4cls_sorted[label_id]["z"])*thred)]:
                    data_dict_filtered[label_id]["body_acc"].append(data_dict[label_id]["body_acc"][i])
                    data_dict_filtered[label_id]["body_gyro"].append(data_dict[label_id]["body_gyro"][i])
                    data_dict_filtered[label_id]["total_acc"].append(data_dict[label_id]["total_acc"][i])
        if print_log:
            print(f"{id2labels[label_id]} filtered data shape: {len(data_dict_filtered[label_id][filter_by])}")
    return data_dict_filtered


def gen_prompt_template_without_rag(data_dict, ground_ans: str="WALKING", contrast_ans: str="STANDING", i: int=0):
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

def eval_generated_ans(ans, grd, contrs):
    # 计算正确率
    # 首先将ans中所有字符串变成大写
    for i in range(len(ans)):
        ans[i] = ''.join([c.upper() if (c >= 'a' and c <= 'z') else c for c in ans[i]])
    correct = 0
    for an in ans:
        count_grd = an.count(grd)
        # count_contrs = an.count(contrs)
        if count_grd == 0:
            # print(f"fault:{an}", end="\n__\n")
            # print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
            # 把回答错误的an用红色字体打印出来:
            print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
            continue 
        # elif count_contrs == 0 and count_grd > 0:
        #     correct += 1
        # elif count_contrs > 0 and count_grd > 0:
        #     grd_begin = an.find(grd)
        #     contrs_begin = an.find(contrs)  
        #     if (grd_begin < contrs_begin and count_grd >= count_contrs) or grd_begin == 0:
        #         correct += 1
        #     else:
        #         print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
        #         print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
        else:
            # print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
            # print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
            correct += 1
    return correct / len(ans)
