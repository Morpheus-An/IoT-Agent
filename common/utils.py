from math import e
from imports import *
from common.read_data import *
from common.generate_prompt import *
from common.model import *
import random



def task_dependent_info(args, i, data_dict, label_dict, ground_ans):
    if args.task_type == "machine_detection":
        grd = "Pos"
        con = "Neg"
        template, data_des = generate_prompt_template(
            args,
            data_dict,
            label_dict,
            "Cooler condition %",
            i,
            grd
        ) # type: ignore
        query = """Is the machine's cooling system functioning properly?"""
    elif args.task_type == "imu_HAR":
        if args.cls_num == 2:
            grd = "STANDING"
            con = "WALKING"
            template, data_des = gen_prompt_template_with_rag_imu_2cls(args,label_dict, data_dict, grd, con, i) # type: ignore
            query = """
Based on the given data,choose the activity that the subject is most likely to be performing from the following two options:"""
        else:
            pass # TODO
    elif args.task_type == "ecg_detection":
        grd = True 
        con = False
        template, data_des = gen_prompt_with_rag_ECG(args, data_dict, grd, i)
        query = """Is the ECG heatbeat signal normal or abnormal?"""
    elif args.task_type == "wifi_localization":
        if args.model == "llama2" or args.model == "Mistral":
            K=3
            rssi = data_dict['val_rssi'][i,:]
            len_base = len(data_dict['database_rssi'])
            similarity = np.zeros(len_base)
            for j in range(0, len_base):
                similarity[j] = compute_similarity(rssi, data_dict['database_rssi'][j])
            idx_similarity = np.argsort(similarity, axis=-1, kind='quicksort', order=None)[::-1]
            similarity_ordered = similarity[idx_similarity]
            neighbor_position = data_dict['database_position'][idx_similarity[:K], :]
            neighbor_similarity = similarity[idx_similarity[:K]]

            data_des = f"""
            The rssi sample: {rssi}
            Based on the neighbor searching tools, the top-{K} position of the neighbors are: {neighbor_position},
                    there corresponding similarities are : {neighbor_similarity}.
            """

            prompt = f"""{Role_Definition_llama(args)}
            EXPERT:
            1. RSSI data: 
            The structure of rssi data is {rssi.shape}, consisting of measurements from six different locations. The intensity of the signal can be influenced by the location of human.
            2. K={K}, represents the number of nearest neighbors considered for estimating the target position.
            3. 
            """
            prompt += """
            {% for domain_doc in documents_domain %}
            {{ domain_doc.content }}
            {% endfor %}

            EXAMPLE:
            {% for d in demo %}{{ d.content }}{% endfor %}

            QUESTION: {{ query }}
            """
            prompt += f"""
            THE GIVEN DATA: 
            {data_des}
            Please analyze the data and conduct the algorithm step by step, and answer: "What is the estimated location?" The answer need to be the form of "[%d, %d]"
            ANSWER: 
            """
            query = """Based on the given data and the provided knowledge, estimate the x-y position:"""
            grd = None
            con = None
            template = prompt
        else:
            K=3
            rssi = data_dict['val_rssi'][i,:]
            len_base = len(data_dict['database_rssi'])
            similarity = np.zeros(len_base)
            for j in range(0, len_base):
                similarity[j] = compute_similarity(rssi, data_dict['database_rssi'][j])
            idx_similarity = np.argsort(similarity, axis=-1, kind='quicksort', order=None)[::-1]
            similarity_ordered = similarity[idx_similarity]
            neighbor_position = data_dict['database_position'][idx_similarity[:K], :]
            neighbor_similarity = similarity[idx_similarity[:K]]

            data_des = f"""
            The rssi sample: {rssi}
            Based on the neighbor searching tools, the top-{K} position of the neighbors are: {neighbor_position},
                    there corresponding similarities are : {neighbor_similarity}.
            """

            prompt = f"""{Role_Definition(args)}
            EXPERT:
            1. RSSI data: 
            The structure of rssi data is {rssi.shape}, consisting of measurements from six different locations. The intensity of the signal can be influenced by the location of human.
            2. The rssi database used for WKNN consists of position coordinates and corresponding rssi values for known locations, where the positions and the indices of the RSSI data correspond one-to-one.
            3. K={K}, represents the number of nearest neighbors considered for estimating the target position.
            4. Other domain knowledge:
            """
            prompt += """
            {% for domain_doc in documents_domain %}
            {{ domain_doc.content }}
            {% endfor %}

            You need to comprehensively analyze the rssi data and implement the WKNN algorithm to estimate the position of given rssi.

            EXAMPLE:
            {% for d in demo %}{{ d.content }}{% endfor %}

            QUESTION: {{ query }}
            """
            prompt += f"""
            THE GIVEN DATA: 
            {data_des}
            Before answering your question, you must refer to the provided knowledge and the previous examples to help you make a clear choice.
            Please analyze the data and conduct the algorithm step by step, and then provide your final answer based on your analysis: "What is the estimated location?" The answer need to be the form of "[%d, %d]"
            ANALYSIS:
            ANSWER:
            """
            query = """Based on the given data and the provided knowledge, estimate the x-y position:"""
            grd = None
            con = None
            template = prompt
    elif args.task_type == "wifi_occupancy":
        if ground_ans == "no_person":
            grd = "no_person"
            con = "have_person"
        else:
            grd = "have_person"
            con = "no_person"
        if args.model == "llama2":
            template, data_des = gen_prompt_with_rag_wifi_occupancy_llama(args, data_dict, grd, con, i)
        else:
            template, data_des = gen_prompt_with_rag_wifi_occupancy(args, data_dict, grd, con, i)
        query = """Based on the given data (the mean value and standard deviations for subcarriers and time steps) and the provided knowledge, determine whether there is a person or not from the following two options:"""
    else:
        assert(0) 
    return grd, con, template, data_des, query
    
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

def select_random_numbers(start, end, count):
    random.seed(42)  # 设置种子以确保每次生成的随机数相同
    return random.sample(range(start, end + 1), count)

def compute_similarity(point_query, point_support):
    rssi_err = point_query - point_support
    abs_err = np.linalg.norm(rssi_err)

    abs_err += 1e-4 if abs_err == 0 else 0
    similarity = 1 / abs_err
    return similarity

# 写入demo-knowledge：
def write_demo_knowledge(args, label2ids,tgt_dir_path: str, data_dict, sample_num: int=5):
    
    file_paths = []
    # 为了不与test使用的demo重复，选用data_dict中的后面的数据作为范例知识
    if args.task_type == "imu_HAR":
        id2labels = {
        id: label for label, id in label2ids.items()
        }
        if args.cls_num == 2:
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
                    file_path = tgt_dir_path + "/" + f"{id2labels[label_id]}_{i}.txt"
                    file_paths.append(file_path)
                    with open(file_path, 'w') as f:
                        f.write(written_content)
                        f.write("\n\n")

    elif args.task_type == "wifi_occupancy":
       
        classes = ['no_person', 'have_person']
        for label in classes:
            for i in range(1, sample_num+1):
                csi = data_dict[label][-i]
                written_content = f"""
                The mean value of CSI: {np.mean(csi)}
                The standard deviation across subcarriers for the mean CSI amplitude over time: {np.std(np.mean(csi, axis=1), axis=0)}
                The mean standard deviation across subcarriers for each time point: {np.mean(np.std(csi, axis=0))}
                ANSWER: {label}
                """
                file_path = tgt_dir_path + "/" + f"{label}_{i}.txt"
                file_paths.append(file_path)
                with open(file_path, 'w') as f:
                    f.write(written_content)
                    f.write("\n\n")

    elif args.task_type == "wifi_localization":
        K=3
        random.seed(42)
        sample_index = select_random_numbers(0, len(data_dict['test_rssi']), sample_num)

        for m in range(len(sample_index)):
            i = sample_index[m]
            rssi = data_dict['test_rssi'][i, :]

            len_base = len(data_dict['database_rssi'])
            similarity = np.zeros(len_base)
            for j in range(0, len_base):
                similarity[i] = compute_similarity(rssi, data_dict['database_rssi'][i])
            idx_similarity = np.argsort(similarity, axis=-1, kind='quicksort', order=None)[::-1]
            similarity_ordered = similarity[idx_similarity]
            neighbor_position = data_dict['database_position'][idx_similarity[:K], :]
            neighbor_similarity = similarity[idx_similarity[:K]]
            label = data_dict['test_position'][i, :]

            neighbor_weight = neighbor_similarity / sum(neighbor_similarity)
            estimate_position = np.average(neighbor_position, weights=neighbor_weight, axis=0)
            written_content = f"""
            The rssi sample: {rssi}
            Use WKNN to estimate the position with K = {K}.
            Based on the neighbor searching tools, the top-{K} position of the neighbors are: {neighbor_position},
            there corresponding similarities are : {neighbor_similarity}.
            Then, for these {K} nearest neighbors' location information, perform a weighted averaging calculation based on their similarity. 
            sum_similarity = sum({neighbor_similarity}) = {sum(similarity)}
            The weight of these neighbors are: {neighbor_weight} = {neighbor_similarity} / sum_similarity
            Estimate_position = Sum({neighbor_weight} * {neighbor_position})
            The final prediction is: {estimate_position}.
            """
            file_path = tgt_dir_path + f"/position_{i}.txt"
            file_paths.append(file_path)
            with open(file_path, 'w') as f:
                f.write(written_content)
                f.write("\n\n")
        
    return file_paths

def get_knowledge_paths(args, is_domain=True, label2id=None, data_dict=None):
    """
    return:
    file_paths: list of str
    """
    file_paths = []
    folder_path = args.knowledge_path + "/" + args.task_type
    if is_domain:
        folder_path += "/domain-knowledge"
    else:
        folder_path += "/demo-knowledge"
    assert os.path.exists(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".md"):
                file_paths.append(os.path.join(root, file))
    meta_data = None 
    if not is_domain:
        
        file_paths = write_demo_knowledge(args, label2id, folder_path, data_dict, 5)
        meta_data = [
            {
                "label": file_path.split("/")[-1][: -len("_i.txt")]
            }
            for file_path in file_paths
        ]
    return file_paths, meta_data




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
    
    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDER_MODEL, meta_fields_to_embed=metadata_fields_to_embed, device=ComponentDevice.from_str(device)) 

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


    






    






    

def pretty_print_res_of_ranker(res):
    for doc in res["documents"]:
        print(doc.meta["file_path"], "\t", doc.score)
        print(doc.content)
        print("\n", "\n")

        
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
