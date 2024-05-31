from imports import *
from dataset import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pywt
import re
import random

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

def gen_prompt_template_with_rag(data_dict, K, i: int=0):
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

    prompt = f"""{Role_Definition()}
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
    return prompt, data_des

def compute_similarity(point_query, point_support):
    rssi_err = point_query - point_support
    abs_err = np.linalg.norm(rssi_err)

    abs_err += 1e-4 if abs_err == 0 else 0
    similarity = 1 / abs_err
    return similarity

#  demo_path, device, splitter_kwargs_domain = {}
def generate_with_rag(
        KB_path,
        data_dict,
        K,
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
        K
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
        demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(device))

        embedding_retriever_domain = InMemoryEmbeddingRetriever(embedded_document_store_KB)
        embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)

        keyword_retriever_domain = InMemoryBM25Retriever(embedded_document_store_KB)
        keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)

        document_joiner_domain = DocumentJoiner()
        document_joiner_demo = DocumentJoiner()

        ranker_domain = TransformersSimilarityRanker(model=RANKER_MODEL)
        ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)

        template, data_des = gen_prompt_template_with_rag(data_dict, K, i)
        prompt_builder = PromptBuilder(template=template)

        set_openAI_key_and_base(use_my_key)
        generator = OpenAIGenerator(model=MODEL["gpt4"], api_base_url=os.environ["OPENAI_BASE_URL"] if use_my_key else None)

        rag_pipeline = Pipeline()
        # 1. for domain-knowledge:
        rag_pipeline.add_component("text_embedder_domain", text_embedder)
        rag_pipeline.add_component("embedding_retriever_domain", embedding_retriever_domain)
        rag_pipeline.add_component("keyword_retriever_domain", keyword_retriever_domain)
        rag_pipeline.add_component("document_joiner_domain", document_joiner_domain)
        rag_pipeline.add_component("ranker_domain", ranker_domain)
        # 2.1 for grd-demo knowledge:
        rag_pipeline.add_component("demo_embedder", demo_embedder)
        rag_pipeline.add_component("embedding_retriever_demo", embedding_retriever_demo)
        rag_pipeline.add_component("keyword_retriever_demo", keyword_retriever_demo)
        rag_pipeline.add_component("document_joiner_demo", document_joiner_demo)
        rag_pipeline.add_component("ranker_demo", ranker_demo)

        # 连接各个components
        # 1. for domain-knowledge:
        rag_pipeline.connect("text_embedder_domain", "embedding_retriever_domain")
        rag_pipeline.connect("embedding_retriever_domain", "document_joiner_domain")
        rag_pipeline.connect("keyword_retriever_domain", "document_joiner_domain")
        rag_pipeline.connect("document_joiner_domain", "ranker_domain")
        # # 2. for demo-knowledge:
        # # 2.1. for ground-truth demo knowledge:
        rag_pipeline.connect("demo_embedder", "embedding_retriever_demo")
        rag_pipeline.connect("embedding_retriever_demo", "document_joiner_demo")
        rag_pipeline.connect("keyword_retriever_demo", "document_joiner_demo")
        rag_pipeline.connect("document_joiner_demo", "ranker_demo")

        query = """Based on the given data and the provided knowledge, estimate the x-y position:"""
        content4retrieval_domain = """1. RSSI data: 
    The structure of rssi data is {rssi.shape}, consisting of measurements from six different locations. The intensity of the signal can be influenced by the location of human.
    2. The rssi database used for WKNN consists of position coordinates and corresponding rssi values for known locations, where the positions and the indices of the RSSI data correspond one-to-one.
    3. K={K}, represents the number of nearest neighbors considered for estimating the target position.
    You need to comprehensively analyze the rssi data and implement the WKNN algorithm to estimate the position of given rssi.
"""
        content4retrieval_demo = data_des

        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.connect("ranker_domain", "prompt_builder.documents_domain")
        rag_pipeline.connect("ranker_demo", "prompt_builder.demo")
        #
        rag_pipeline.add_component("llm", generator)
        rag_pipeline.connect("prompt_builder", "llm")

        result = rag_pipeline.run(
            {
                "text_embedder_domain": {"text": content4retrieval_domain},
                "keyword_retriever_domain": {"query": content4retrieval_domain},
                "ranker_domain": {"query": content4retrieval_domain},
                "demo_embedder": {"text": content4retrieval_demo},
                "embedding_retriever_demo": {
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": ['position'],
                    },
                    "top_k": 1,
                },
                "keyword_retriever_demo": {
                    "query": content4retrieval_demo,
                    "top_k": 1,
                    "filters": {
                        "field": "meta.label",
                        "operator": "in",
                        "value": ['position'],
                    },
                },
                "ranker_demo": {
                    "query": content4retrieval_demo,
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


def select_random_numbers(start, end, count):
    random.seed(42)  # 设置种子以确保每次生成的随机数相同
    return random.sample(range(start, end + 1), count)

# 写入demo-knowledge：
def write_demo_knowledge(tgt_dir_path: str, data_dict, K, sample_num: int=5):
    file_paths = []
    random.seed(42)
    sample_index = select_random_numbers(0, len(data_dict['val_rssi']), sample_num)

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
        file_path = tgt_dir_path + f"_position_{i}.txt"
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
    return """You are tasked with estimating the position using the WKNN (Weighted k-Nearest Neighbors) algorithm, leveraging Wi-Fi Received Signal Strength Indicator (RSSI) data for indoor positioning. 
    Your expertise lies in analyzing Wi-Fi RSSI data to accurately determine the location of individuals within indoor spaces. You understand the operation of RSSI signals.
    Your professional knowledge includes, but is not limited to:
    Algorithm Implementation: Develop and implement the WKNN algorithm to estimate the location of individuals based on Wi-Fi RSSI data.
    Signal Processing: You understand how to preprocess and analyze Wi-Fi rssi data, extracting relevant features for localization.
    As a signal analysis scientist, your task is to estimate the location based on the Wi-Fi CSI data you receive, helping users better understand and manage indoor space utilization."""

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



def gen_prompt_template_without_rag(data_dict, K, i: int=0):
    rssi = data_dict['val_rssi'][i, :]
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

    # prompt = f"""{Role_Definition()}
    #     EXPERT:
    #     1. RSSI data:
    #     The structure of rssi data is {rssi.shape}, consisting of measurements from six different locations. The intensity of the signal can be influenced by the location of human.
    #     2. The rssi database used for WKNN consists of position coordinates and corresponding rssi values for known locations, where the positions and the indices of the RSSI data correspond one-to-one.
    #     3. K={K}, represents the number of nearest neighbors considered for estimating the target position.
    #     4. It provides the guidance to estimate the location of a new rssi sample. So when conducting WKNN, it is necessary to calculate the similarity between the input sample and the samples in database.
    #        The description of using WKNN (Weighted k-Nearest Neighbors) algorithm for RSSI localization: Initialization: When creating an instance of the WKNN class, you need to provide a database containing the known positions\' location information (database_position) and their corresponding RSSI information (database_rssi). Compute Similarity: For the RSSI information to be localized (input_rssi), the first step is to compute its similarity with the RSSI information of each position point in the database.
    #             As all the information provided (including database, input rssi and WKNN algorithm), you can implement the algorithm by code to calculate it more correctly.
    #                database_position = database_position self.database_rssi = database_rssi def compute_similarity(self, point_query, point_support): rssi_err = point_query - point_support abs_err = np.
    #                     Select Nearest Neighbors: Based on the computed similarities, select the K data points from the database that are most similar to the RSSI information being localized. These K nearest neighbors\' location information will be used for the subsequent weighted averaging calculation.
    #                         The dataset contains 6-dimension rssi collected by 6 APs and the corresponding 2D positions. It is important to note that the database containing several samples for each position as reference.
    #                           import numpy as np ###you can implement the code with the given database_position and database_rssi class WKNN: def __init__(self, database_position, database_rssi): super(WKNN, self).__init__() self.
    #                           Return Estimated Position: The ultimate goal of the WKNN algorithm is to estimate the position of the RSSI information being localized. The position obtained through the weighted averaging calculation is returned as the estimated position.
    #                            num_best = K len_base = len(self.database_rssi) similarity = np.\n    \n     Weighted Averaging: For these K nearest neighbors\' location information, perform a weighted averaging calculation based on their similarity to the RSSI information being localized. Generally, data points with higher similarity are assigned higher weights to improve the accuracy of the estimated position.
    #                             You need to comprehensively analyze the rssi data and implement the WKNN algorithm to estimate the position of given rssi.\n\n
    #
    #     """
    prompt = f"""{Role_Definition()}
            EXPERT:
            1. RSSI data: 
            The structure of rssi data is {rssi.shape}, consisting of measurements from six different locations. The intensity of the signal can be influenced by the location of human.
            2. The rssi database used for WKNN consists of position coordinates and corresponding rssi values for known locations, where the positions and the indices of the RSSI data correspond one-to-one.
            3. K={K}, represents the number of nearest neighbors considered for estimating the target position.
            """
    prompt += """

        You need to comprehensively analyze the rssi data and implement the WKNN algorithm to estimate the position of given rssi.

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



def extract_coordinates(text):
    matches = re.findall(r"\s*[\[\(]\s*(-?\d+(\.\d*)?),\s*(-?\d+(\.\d*)?)\s*[\]\)]", text)
    if matches:
        last_match = matches[-1]
        x = float(last_match[0]) if '.' in last_match[0] else int(last_match[0])
        y = float(last_match[2]) if '.' in last_match[2] else int(last_match[2])
        return [x, y]
    else:
        return None
