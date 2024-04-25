from imports import *
from dataset import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pywt

def create_indexing_pipeline(document_store, metadata_fields_to_embed=None):

    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2)
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDER_MODEL,
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

def prepare_and_embed_documents(document_store, source_paths: List[str], metadata_fields_to_embed=None, meta_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]=None, splitter_kwards: dict=None, draw: str=None, device: str="cuda:0"):
    """将指定路径下的领域知识文档emebedding成向量库（支持pdf,markdown,txt格式）"""
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

def gen_prompt_template_with_rag(data_dict, ground_ans: str="no_person", contract_ans: str="have_person", i: int=0):
    csi = data_dict[ground_ans][i,:,:]
    data_des = f"""
    The mean value of CSI: {np.mean(csi)}
    The standard deviation across subcarriers for the mean CSI amplitude over time: {np.std(np.mean(csi, axis=1), axis=0)}
    The mean standard deviation across subcarriers for each time point: {np.mean(np.std(csi, axis=0))}
    """

    prompt = f"""{Role_Definition()}
    EXPERT:
    1. CSI data: 
    The structure of CSI data is {csi.shape}, where the first dimension means a time-series signal consisting of {csi.shape[0]} data samples and the second dimension means {csi.shape[1]} subcarriers of CSI data. It represents the amplitude of the signal, which can be reflected by the human presence.
    2. The mean value of CSI: 
    The mean value of CSI is a scalar that describe the average amplitude of the CSI data.
    3. The standard deviation across subcarriers for the mean CSI amplitude over time:
    It is a scalar which represents the variability of the mean CSI amplitude across different subcarriers over time.
    4. The mean std of CSI across the time axis:
    It is a scalar that describes the average std of CSI signals for each subcarrier over time. It reflects the overall degree of signal oscillation in time.
    5. Other domain knowledge:
    """
    prompt += """
    {% for domain_doc in documents_domain %}
    {{ domain_doc.content }}
    {% endfor %}

    You need to comprehensively analyze the CSI data along the two axis(time and subcarrier). For each axis, you should analyze not only the magnitude but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding the presence of human.

    EXAMPLE1:
    {% for d in grd_demo %}{{ d.content }}{% endfor %}

    EXAMPLE2:
    {% for d in con_demo %}{{ d.content }}{% endfor %}

    QUESTION: {{ query }}
    """
    prompt += f"""
    {ground_ans}
    {contract_ans}
    THE GIVEN DATA: 
    {data_des}
    Before answering your question, you must refer to the provided knowledge and the previous examples and compare the mean data, the standard deviation across subcarriers for the mean CSI amplitude over time and the mean std of CSI across the time axis in the examples with those in the question comprehensively, in order to help you make a clear choice.
    Please analyze the data step by step to explain what it reflects,.and then provide your final answer based on your analysis:"Is there is a person or not?"
    ANALYSIS:
    ANSWER:
    """
    return prompt, data_des

#  demo_path, device, splitter_kwargs_domain = {}
def generate_with_rag(
        ground_ans,
        contrast_ans,
        KB_path,
        data_dict,
        demo_dir_path = "demo-knowledge/demo-knowledge",
        splitter_kwargs_domain = {
            "split_by": "sentence",
            "split_length": 2,
        },
        splitter_kwargs_demo = {
            "split_by": "passage",
            "split_length": 1,
        },
        num_samples = 20,
        use_my_key = False,
        device="cuda:0",

):
    Demo_paths = write_demo_knowledge(
        demo_dir_path,
        data_dict,
    )
    meta_data = [
        {
            "label": file_path.split('/')[-1][len("demo-knowledge_"):-len("_i.txt")]
        }
        for file_path in Demo_paths
    ]
    # print(meta_data)
    document_store_domain = InMemoryDocumentStore()
    embedded_document_store_KB = prepare_and_embed_documents(document_store_domain, KB_path, draw=None, device=device, splitter_kwards=splitter_kwargs_domain)

    document_store_demo = InMemoryDocumentStore()
    embedded_document_store_DM = prepare_and_embed_documents(document_store_demo, Demo_paths, draw=None, device=device, splitter_kwards=splitter_kwargs_demo, meta_data=meta_data)

    ans = []
    for i in range(num_samples):
        text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(device))
        grd_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(device))
        con_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(device))

        embedding_retriever_domain = InMemoryEmbeddingRetriever(embedded_document_store_KB)
        grd_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)
        con_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)

        keyword_retriever_domain = InMemoryBM25Retriever(embedded_document_store_KB)
        grd_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)
        con_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)

        document_joiner_domain = DocumentJoiner()
        grd_document_joiner_demo = DocumentJoiner()
        con_document_joiner_demo = DocumentJoiner()

        ranker_domain = TransformersSimilarityRanker(model=RANKER_MODEL)
        grd_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)
        con_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)

        template, data_des = gen_prompt_template_with_rag(data_dict, ground_ans, contrast_ans, i)
        prompt_builder = PromptBuilder(template=template)

        set_openAI_key_and_base(use_my_key)
        generator = OpenAIGenerator(model=MODEL["gpt3.5"], api_base_url=os.environ["OPENAI_BASE_URL"] if use_my_key else None)

        rag_pipeline = Pipeline()
        # 1. for domain-knowledge:
        rag_pipeline.add_component("text_embedder_domain", text_embedder)
        rag_pipeline.add_component("embedding_retriever_domain", embedding_retriever_domain)
        rag_pipeline.add_component("keyword_retriever_domain", keyword_retriever_domain)
        rag_pipeline.add_component("document_joiner_domain", document_joiner_domain)
        rag_pipeline.add_component("ranker_domain", ranker_domain)
        # 2.1 for grd-demo knowledge:
        rag_pipeline.add_component("grd_demo_embedder", grd_demo_embedder)
        rag_pipeline.add_component("grd_embedding_retriever_demo", grd_embedding_retriever_demo)
        rag_pipeline.add_component("grd_keyword_retriever_demo", grd_keyword_retriever_demo)
        rag_pipeline.add_component("grd_document_joiner_demo", grd_document_joiner_demo)
        rag_pipeline.add_component("grd_ranker_demo", grd_ranker_demo)
        # 2.2 for con-demo knowledge:
        rag_pipeline.add_component("con_demo_embedder", con_demo_embedder)
        rag_pipeline.add_component("con_embedding_retriever_demo", con_embedding_retriever_demo)
        rag_pipeline.add_component("con_keyword_retriever_demo", con_keyword_retriever_demo)
        rag_pipeline.add_component("con_document_joiner_demo", con_document_joiner_demo)
        rag_pipeline.add_component("con_ranker_demo", con_ranker_demo)

        # 连接各个components
        # 1. for domain-knowledge:
        rag_pipeline.connect("text_embedder_domain", "embedding_retriever_domain")
        rag_pipeline.connect("embedding_retriever_domain", "document_joiner_domain")
        rag_pipeline.connect("keyword_retriever_domain", "document_joiner_domain")
        rag_pipeline.connect("document_joiner_domain", "ranker_domain")
        # # 2. for demo-knowledge:
        # # 2.1. for ground-truth demo knowledge:
        rag_pipeline.connect("grd_demo_embedder", "grd_embedding_retriever_demo")
        rag_pipeline.connect("grd_embedding_retriever_demo", "grd_document_joiner_demo")
        rag_pipeline.connect("grd_keyword_retriever_demo", "grd_document_joiner_demo")
        rag_pipeline.connect("grd_document_joiner_demo", "grd_ranker_demo")
        # # 2.2. for contrast demo knowledge:
        rag_pipeline.connect("con_demo_embedder", "con_embedding_retriever_demo")
        rag_pipeline.connect("con_embedding_retriever_demo", "con_document_joiner_demo")
        rag_pipeline.connect("con_keyword_retriever_demo", "con_document_joiner_demo")
        rag_pipeline.connect("con_document_joiner_demo", "con_ranker_demo")
        query = """Based on the given data and the provided knowledge, determine whether there is a person or not from the following two options:"""
        content4retrieval_domain = """1. CSI data: 
    The structure of CSI data is {csi.shape}, where the first dimension means a time-series signal consisting of {csi.shape[0]} data samples and the second dimension means {csi.shape[1]} subcarriers of CSI data. It represents the amplitude of the signal, which can be reflected by the human presence.
    2. The mean value of CSI: 
    The mean value of CSI is a scalar that describe the average amplitude of the CSI data.
    3. The standard deviation across subcarriers for the mean CSI amplitude over time:
    It is a scalar which represents the variability of the mean CSI amplitude across different subcarriers over time.
    4. The mean std of CSI across the time axis:
    It is a scalar that describes the average std of CSI signals for each subcarrier over time. It reflects the overall degree of signal oscillation in time.
    You need to comprehensively analyze the CSI data along the two axis(time and subcarrier). For each axis, you should analyze not only the magnitude but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding the subject's motion status.

"""
        content4retrieval_grd_demo = data_des 
        content4retrieval_con_demo = f"ANSWER: {contrast_ans}"

        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.connect("ranker_domain", "prompt_builder.documents_domain")
        rag_pipeline.connect("grd_ranker_demo", "prompt_builder.grd_demo")
        rag_pipeline.connect("con_ranker_demo", "prompt_builder.con_demo")
        #
        rag_pipeline.add_component("llm", generator)
        rag_pipeline.connect("prompt_builder", "llm")

        result = rag_pipeline.run(
            {
                "text_embedder_domain": {"text": content4retrieval_domain},
                "keyword_retriever_domain": {"query": content4retrieval_domain},
                "ranker_domain": {"query": content4retrieval_domain},
                "grd_demo_embedder": {"text": content4retrieval_grd_demo},
                "con_demo_embedder": {"text": content4retrieval_con_demo},
                "grd_embedding_retriever_demo": {
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": [ground_ans],
                    },
                    "top_k": 1,
                },
                "con_embedding_retriever_demo": {
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": [contrast_ans],
                    }
                },
                "grd_keyword_retriever_demo": {
                    "query": content4retrieval_grd_demo,
                    "top_k": 1,
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": [ground_ans],
                    },
                },
                "con_keyword_retriever_demo": {
                    "query": content4retrieval_con_demo,
                    "top_k": 1,
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": [contrast_ans],
                    },
                },
                "grd_ranker_demo": {
                    "query": content4retrieval_grd_demo,
                    "top_k": 1,
                },
                "con_ranker_demo": {
                    "query": content4retrieval_con_demo,
                    "top_k": 1,
                },             
                "prompt_builder": {"query": query},
            }
        )

        an = result["llm"]["replies"][0]
        print(an)
        ans.append(an)
        # assert(0)
        if i % 5 == 0:
            print(f"第{i}次预测完成")
    return ans

def pretty_print_res_of_ranker(res):
    for doc in res["documents"]:
        print(doc.meta["file_path"], "\t", doc.score)
        print(doc.content)
        print("\n", "\n")

# 写入demo-knowledge：
def write_demo_knowledge(tgt_dir_path: str, data_dict, sample_num: int=5):
    file_paths = []
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
            file_path = tgt_dir_path + f"_{label}_{i}.txt"
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
    return """You are a signal analysis scientist, specializing in analyzing Wi-Fi Channel State Information (CSI) data to understand human occupancy patterns. Your expertise in interpreting CSI data makes you proficient in human occupancy detection tasks. Your role is to assist users in determining the occupancy status of indoor spaces by analyzing Wi-Fi CSI data.
Your training enables you to interpret and analyze the data collected by Wi-Fi CSI sensors, thereby identifying different occupancy patterns. You understand the variations in Wi-Fi signal amplitude caused by human presence and movement, and can determine the current occupancy status based on changes in the data.
Your professional knowledge includes, but is not limited to:
1. Signal Processing: You understand how to preprocess and analyze Wi-Fi CSI data, extracting relevant features for occupancy detection.
2. Machine Learning and Pattern Recognition: You can utilize machine learning algorithms and pattern recognition techniques to classify occupancy status based on Wi-Fi CSI data.
3. Environmental Sensing: You are familiar with environmental factors that can affect Wi-Fi signal propagation and can account for these factors in occupancy detection.
4. Occupancy Behavior Analysis: You understand the patterns of human behavior in indoor environments and can incorporate this knowledge into occupancy detection algorithms.
As a signal analysis scientist, your task is to determine whether there is a person or not based on the Wi-Fi CSI data you receive, helping users better understand and manage indoor space utilization."""

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


def time_downsample(data, time_length, time_downsample):
    # 初始化存储序列的列表
    sequences = []

    # 获取数据的长度
    data_length = data.shape[0]

    # 计算数据可以划分成多少个序列
    num_sequences = data_length // time_downsample // time_length

    # 循环生成每个序列
    for i in range(num_sequences):
        start_index = i * time_downsample * time_length
        end_index = start_index + time_length * time_downsample
        sequence = data[start_index:end_index:time_downsample, :]
        sequences.append(sequence)

    return np.array(sequences)

def read_raw_csi(root="wifi_csi_har_dataset/room_2/1", subcarrier_dim=40, frames_num=10, frame_downsample=2):
    val_dataset = CSIDataset([
        root,
    ])
    no_person1 = val_dataset.amplitudes[5220:6179, :] #1000
    # no_person2= val_dataset.amplitudes[12265:12564, :] #300
    walking1 = val_dataset.amplitudes[2505: 3204, :]  #740
    # walking2 = val_dataset.amplitudes[40: 589, :] #540
    # getting_down = val_dataset.amplitudes[7170: 7289, :]  #120
    # getting_up = val_dataset.amplitudes[7650:7709, :]  #60
    # sitting = val_dataset.amplitudes[3800:4319, :]  #520
    # no_person = np.concatenate((no_person1, no_person2), axis=0)
    # walking = np.concatenate((walking1, walking2), axis=0)

    no_person_data = time_downsample(no_person1, frames_num, frame_downsample)
    walking_data = time_downsample(walking1, frames_num, frame_downsample)
    # getting_up_data = time_downsample(getting_up, frames_num, frame_downsample)
    # getting_down_data = time_downsample(getting_down, frames_num, frame_downsample)
    # sitting_data = time_downsample(sitting, frames_num, frame_downsample)
    # yes_person_data = np.concatenate((walking_data, getting_up_data, getting_down_data, sitting_data), axis=0)
    yes_person_data = walking_data

    downsample_interval = no_person_data.shape[2] // subcarrier_dim

    # 降采样数据
    no_person_data = no_person_data[:, :, ::downsample_interval]
    yes_person_data = yes_person_data[:, :, ::downsample_interval]

    # no_person_labels = np.zeros((no_person_data.shape[0],), dtype=int)
    # yes_person_labels = np.ones((yes_person_data.shape[0],), dtype=int)
    #
    # # 合并数据和标签
    # combined_data = np.concatenate((no_person_data, yes_person_data), axis=0)
    # combined_labels = np.concatenate((no_person_labels, yes_person_labels), axis=0)

    return {'no_person': no_person_data, 'have_person': yes_person_data}


def set_openAI_key_and_base(set_base=True):
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
    csi = data_dict[ground_ans][i]
    prompt = f"""{Role_Definition()}
    EXPERT:
    1. CSI data: 
    The structure of CSI data is {csi.shape}, where the first dimension means a time-series signal consisting of {csi.shape[0]} data samples and the second dimension means {csi.shape[1]} subcarriers of CSI data. It represents the amplitude of the signal, which can be reflected by the human presence.
    2. The mean value of CSI: 
    The mean value of CSI is a scalar that describe the average amplitude of the CSI data.
    3. The standard deviation across subcarriers for the mean CSI amplitude over time:
    It is a scalar which represents the variability of the mean CSI amplitude across different subcarriers over time.
    4. The mean std of CSI across the time axis:
    It is a scalar that describes the average std of CSI signals for each subcarrier over time. It reflects the overall degree of signal oscillation in time.
    
    You need to comprehensively analyze the CSI data along the two axis(time and subcarrier). For each axis, you should analyze not only the magnitude but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding whether there is a person or not.
    
    QUESTION: Based on the given data, determine whether there is a person or not from the following two options: 
    no_person, have_person

    ​
    THE GIVEN DATA: 
    The CSI data: {csi}
    The mean value of CSI: {np.mean(csi)}
    The standard deviation across subcarriers for the mean CSI amplitude over time: {np.std(np.mean(csi, axis=1), axis=0)}
    The mean standard deviation across subcarriers for each time point: {np.mean(np.std(csi, axis=0))}

    Please analyze the data step by step to explain what it reflects,.and then provide your final answer based on your analysis:"Is there is a person or not?"
    ANALYSIS:
    ANSWER:
    """

    return prompt

def eval_generated_ans(ans, grd, contrs):
    # 计算正确率
    # 首先将ans中所有字符串变成大写
    for i in range(len(ans)):
        ans[i] = ''.join([c.upper() for c in ans[i]])
    correct = 0
    for an in ans:
        count_grd = an.count(grd)
        count_contrs = an.count(contrs)
        if count_grd == 0:
            # print(f"fault:{an}", end="\n__\n")
            print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
            # 把回答错误的an用红色字体打印出来:
            print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
            continue 
        elif count_contrs == 0 and count_grd > 0:
            correct += 1
        elif count_contrs > 0 and count_grd > 0:
            grd_begin = an.find(grd)
            contrs_begin = an.find(contrs)  
            if (grd_begin < contrs_begin and count_grd >= count_contrs) or grd_begin == 0:
                correct += 1
            else:
                print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
                print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
        else:
            print(f"{grd}count: {count_grd}, {contrs}count: {count_contrs}")
            print(f"\033[1;31m fault: {an} \033[0m", end="\n__\n")
    return correct / len(ans)
