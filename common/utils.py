from math import e
from imports import *
from common.read_data import *
from common.generate_prompt import *
from common.model import *



def task_dependent_info(args, i, data_dict, label_dict):
    if args.task_type == "machine_detection":
        # grd = "Neg"
        grd = args.grd
        con = "Pos" if grd == "Neg" else "Neg"
        template, data_des = gen_prompt_tamplate_with_rag_machine(args, data_dict, label_dict, "Cooler condition %", i, grd)
        query = """Is the machine's cooling system functioning properly?"""
    elif args.task_type == "imu_HAR":
        if args.cls_num == 2:
            # grd = "WALKING"
            # con = "STANDING"
            grd = args.grd
            con = "STANDING" if grd == "WALKING" else "WALKING"
            template, data_des = gen_prompt_template_with_rag_imu(args, label_dict, data_dict, grd, con, i) # type: ignore
            query = """
Based on the given data,choose the activity that the subject is most likely to be performing from the following two options:"""
        elif args.cls_num > 2:
            candidates = ["LAYING", "WALKING_UPSTAIRS", "LIE_TO_SIT"]
            # grd = candidates[0]
            grd = args.grd
            template, data_des = gen_prompt_template_with_rag_imu(args, label_dict, data_dict, grd, None, i, candidates)
            query = """
Based on the given data, choose the activity that the subject is most likely to be performing from the following options:"""
            con = [c for c in candidates if c != grd]
        else:
            raise ValueError("cls_num must be greater than 2")
    elif args.task_type == "ecg_detection":
        # grd = False
        grd = True if args.grd == "normal" else False
        con = not grd
        template, data_des = gen_prompt_with_rag_ECG(args, data_dict, grd, i)
        query = """Is the ECG heatbeat signal normal or abnormal?"""
    elif args.task_type == "wifi_localization":
        pass # TODO
    elif args.task_type == "wifi_occupancy":
        pass # TODO
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

# 写入demo-knowledge：
def write_demo_knowledge(args, label2ids,tgt_dir_path: str, data_dict, sample_num: int=5):
    id2labels = {
        id: label for label, id in label2ids.items()
    }
    file_paths = []
    # 为了不与test使用的demo重复，选用data_dict中的后面的数据作为范例知识
    if args.task_type == "imu_HAR":
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
        if len(file_paths) == 0:
            file_paths = write_demo_knowledge(args, label2id, folder_path, data_dict, 9)
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
