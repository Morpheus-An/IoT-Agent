from imports import *
# from utils import read_raw_data_and_preprocess, filter_data_dict_with_var, get_openAI_model, gen_prompt_template_without_rag, eval_generated_ans, prepare_and_embed_documents, gen_prompt_template_with_rag, set_openAI_key_and_base, pretty_print_res_of_ranker, write_demo_knowledge,eval_by_gpt, read_multicls_data_and_preprocess
from utils import * 

def get_all_file_paths(folder_path: str):
    """
    return:
    file_paths: list of str
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".md"):
                file_paths.append(os.path.join(root, file))
    return file_paths
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
    data_dict, label_duct = read_machine_data()
    start_time = time.perf_counter()
    config = "machine-domain-knowledge & handed demo-knowledge"
    # with rerieval:
    # 首先，准备好document_store并写入:
    # 得到指定文件夹下所有文件的路径
    KB_paths = get_all_file_paths("/home/ant/RAG/Machine-knowledge/domain-knowledge")
    device = "cuda:0"
    document_store_domain = InMemoryDocumentStore()
    splitter_kwags_domain = {"split_by": "sentence", "split_length": 2}
    embedded_document_store_KB = prepare_and_embed_documents(document_store_domain, KB_paths, draw=None, device=device, splitter_kwards=splitter_kwags_domain)

    # document_store_demo = InMemoryDocumentStore()
    # splitter_kwags_demo = {"split_by": "passage", "split_length": 1}
    # embedded_document_store_DM = prepare_and_embed_documents(document_store_demo, Demo_paths, draw=None, device=device, splitter_kwards=splitter_kwags_demo, meta_data=meta_data)

    
    ans = []
    with open("output_details-machine.log", "a") as f:
        for i in range(1, 51):
            # 建立一个rag的pipeline，使用hybrid的retrieval方法进行检索
            # first, 定义components:
            text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL_LOCAL, device=ComponentDevice.from_str(device))
            # grd_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL_LOCAL, device=ComponentDevice.from_str(device))
            # con_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL_LOCAL, device=ComponentDevice.from_str(device))

            embedding_retriever_domain = InMemoryEmbeddingRetriever(embedded_document_store_KB)
            # grd_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)
            # con_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)

            keyword_retriever_domain = InMemoryBM25Retriever(embedded_document_store_KB)
            # grd_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)
            # con_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)

            document_joiner_domain = DocumentJoiner()
            # grd_document_joiner_demo = DocumentJoiner()
            # con_document_joiner_demo = DocumentJoiner()


            ranker_domain = TransformersSimilarityRanker(model=RANKER_MODEL_LOCAL)
            # grd_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL_LOCAL)
            # con_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL_LOCAL)
            grd = "Pos"
            template, data_des = gen_prompt_tamplate_with_rag_machine(data_dict, label_duct, "Cooler condition %", i, grd)
            prompt_builder = PromptBuilder(template=template)
            set_openAI_key_and_base(False, set_proxy=PROXY)
            generator = OpenAIGenerator(model=MODEL["gpt4"])

            # seconde: 创建pipeline:
            rag_pipeline = Pipeline()
            # 1. for domain-knowledge:
            rag_pipeline.add_component("text_embedder_domain", text_embedder)
            rag_pipeline.add_component("embedding_retriever_domain", embedding_retriever_domain)
            rag_pipeline.add_component("keyword_retriever_domain", keyword_retriever_domain)
            rag_pipeline.add_component("document_joiner_domain", document_joiner_domain)
            rag_pipeline.add_component("ranker_domain", ranker_domain)
            # 2. for demo-knowledge:
            # 2.1 for ground-truth demo knowledge:
            # rag_pipeline.add_component("grd_demo_embedder", grd_demo_embedder)
            # rag_pipeline.add_component("grd_embedding_retriever_demo", grd_embedding_retriever_demo)
            # rag_pipeline.add_component("grd_keyword_retriever_demo", grd_keyword_retriever_demo)
            # rag_pipeline.add_component("grd_document_joiner_demo", grd_document_joiner_demo)
            # rag_pipeline.add_component("grd_ranker_demo", grd_ranker_demo)
            # # 2.2 for contrast demo knowledge:
            # rag_pipeline.add_component("con_demo_embedder", con_demo_embedder)
            # rag_pipeline.add_component("con_embedding_retriever_demo", con_embedding_retriever_demo)
            # rag_pipeline.add_component("con_keyword_retriever_demo", con_keyword_retriever_demo)
            # rag_pipeline.add_component("con_document_joiner_demo", con_document_joiner_demo)
            # rag_pipeline.add_component("con_ranker_demo", con_ranker_demo)
            
            # third: 将components连接起来
            # 1. for domain-knowledge:
            rag_pipeline.connect("text_embedder_domain", "embedding_retriever_domain")
            rag_pipeline.connect("embedding_retriever_domain", "document_joiner_domain")
            rag_pipeline.connect("keyword_retriever_domain", "document_joiner_domain")
            rag_pipeline.connect("document_joiner_domain", "ranker_domain")
            # # 2. for demo-knowledge:
            # # 2.1. for ground-truth demo knowledge:
            # rag_pipeline.connect("grd_demo_embedder", "grd_embedding_retriever_demo")
            # rag_pipeline.connect("grd_embedding_retriever_demo", "grd_document_joiner_demo")
            # rag_pipeline.connect("grd_keyword_retriever_demo", "grd_document_joiner_demo")
            # rag_pipeline.connect("grd_document_joiner_demo", "grd_ranker_demo")
            # # 2.2. for contrast demo knowledge:
            # rag_pipeline.connect("con_demo_embedder", "con_embedding_retriever_demo")
            # rag_pipeline.connect("con_embedding_retriever_demo", "con_document_joiner_demo")
            # rag_pipeline.connect("con_keyword_retriever_demo", "con_document_joiner_demo")
            # rag_pipeline.connect("con_document_joiner_demo", "con_ranker_demo")
            # rag_pipeline.draw("retriver_pipeline2.png")
            # print("draw1 done")
            query = """Is the machine's cooling system functioning properly?"""
            content4retrieval_domain = gen_content4retrive_domain(data_des)
        #     content4retrieval_grd_demo = None
            # content4retrieval_con_demo = None
            # retrieved = rag_pipeline.run(
            #     {
            #         "text_embedder_domain": {"text": content4retrieval_domain},
            #         # "grd_demo_embedder": {"text": content4retrieval_grd_demo},
            #         # "con_demo_embedder": {"text": content4retrieval_con_demo},

            #         # "grd_embedding_retriever_demo": {
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [ground_ans]
            #         #     },
            #         #     "top_k": 2,
            #         # },
            #         # "con_embedding_retriever_demo": {
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [contrast_ans]
            #         #     }
            #         # },

            #         "keyword_retriever_domain": {"query": content4retrieval_domain},
            #         # "grd_keyword_retriever_demo": {"query": content4retrieval_grd_demo, "top_k": 1,
                                                
            #         # "filters": {"field" : "meta.label", "operator": "in", "value": [ground_ans]
            #         # }
            #         # # },
            #         # "con_keyword_retriever_demo": {
            #         #     "query": content4retrieval_con_demo,
            #         #     "top_k": 1,
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [contrast_ans],
            #         #     },
            #         # },
            #         # "con_keyword_retriever_demo": {"query": content4retrieval_domain},
            #         "ranker_domain": {"query": content4retrieval_domain},
            #         # "grd_ranker_demo": {"query": content4retrieval_grd_demo, "top_k": 1},
            #         # "con_ranker_demo": {"query": content4retrieval_con_demo, "top_k": 1},
            #     }
            # )
            # pretty_print_res_of_ranker(retrieved["ranker_domain"])
            # print("___________________________________________________________")
            # pretty_print_res_of_ranker(retrieved["grd_ranker_demo"])
            # print("___________________________________________________________")
            # assert(0)
            # pretty_print_res_of_ranker(retrieved["con_ranker_demo"])
            # assert(0)
            rag_pipeline.add_component("prompt_builder", prompt_builder)
            rag_pipeline.connect("ranker_domain", "prompt_builder.documents_domain")
            # rag_pipeline.connect("grd_ranker_demo", "prompt_builder.grd_demo")
            # rag_pipeline.connect("con_ranker_demo", "prompt_builder.con_demo")
            # 打印看看喂给llm的prompt长什么样子
            # final_prompt = rag_pipeline.run(
            #     {
            #         # "text_embedder_domain": {"text": content4retrieval_domain},
            #         # "keyword_retriever_domain": {"query": content4retrieval_domain},
            #         # "ranker_domain": {"query": content4retrieval_domain},
            #         # "grd_demo_embedder": {"text": content4retrieval_grd_demo},
            #         # "con_demo_embedder": {"text": content4retrieval_con_demo},
            #         # "grd_embedding_retriever_demo": {
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [ground_ans],
            #         #     },
            #         #     "top_k": 1,
            #         # },
            #         # "con_embedding_retriever_demo": {
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [contrast_ans],
            #         #     }
            #         # },
            #         # "grd_keyword_retriever_demo": {
            #         #     "query": content4retrieval_grd_demo,
            #         #     "top_k": 1,
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [ground_ans],
            #         #     },
            #         # },
            #         # "con_keyword_retriever_demo": {
            #         #     "query": content4retrieval_con_demo,
            #         #     "top_k": 1,
            #         #     "filters": {
            #         #         "field": "meta.label",
            #         #         "operator": "in",
            #         #         "value": [contrast_ans],
            #         #     },
            #         # },
            #         # "grd_ranker_demo": {
            #         #     "query": content4retrieval_grd_demo,
            #         #     "top_k": 1,
            #         # },
            #         # "con_ranker_demo": {
            #         #     "query": content4retrieval_con_demo,
            #         #     "top_k": 1,
            #         # },             
            #         "prompt_builder": {"query": query},
            #     }
            # )
            # print(f"final_prompt is:\n{final_prompt['prompt_builder']['prompt']}")
            # print("___________________________________________________________")
            # assert(0)
            rag_pipeline.add_component("llm", generator)
            rag_pipeline.connect("prompt_builder", "llm")
            # rag_pipeline.draw("rag_pipeline.png")
            result = rag_pipeline.run(
                {
                    "text_embedder_domain": {"text": content4retrieval_domain},
                    "keyword_retriever_domain": {"query": content4retrieval_domain},
                    "ranker_domain": {"query": content4retrieval_domain},
                    # "grd_demo_embedder": {"text": content4retrieval_grd_demo},
                    # "con_demo_embedder": {"text": content4retrieval_con_demo},
                    # "grd_embedding_retriever_demo": {
                    #     "filters": {
                    #         "field": "meta.label",
                    #         "operator": "in",
                    #         "value": [ground_ans],
                    #     },
                    #     "top_k": 1,
                    # },
                    # "con_embedding_retriever_demo": {
                    #     "filters": {
                    #         "field": "meta.label",
                    #         "operator": "in",
                    #         "value": [contrast_ans],
                    #     }
                    # },
                    # "grd_keyword_retriever_demo": {
                    #     "query": content4retrieval_grd_demo,
                    #     "top_k": 1,
                    #     "filters": {
                    #         "field": "meta.label",
                    #         "operator": "in",
                    #         "value": [ground_ans],
                    #     },
                    # },
                    # "con_keyword_retriever_demo": {
                    #     "query": content4retrieval_con_demo,
                    #     "top_k": 1,
                    #     "filters": {
                    #         "field": "meta.label",
                    #         "operator": "in",
                    #         "value": [contrast_ans],
                    #     },
                    # },
                    # "grd_ranker_demo": {
                    #     "query": content4retrieval_grd_demo,
                    #     "top_k": 1,
                    # },
                    # "con_ranker_demo": {
                    #     "query": content4retrieval_con_demo,
                    #     "top_k": 1,
                    # },             
                    "prompt_builder": {"query": query},
                }
            )
            an = result["llm"]["replies"][0]
            print(an)
            if i == 1:
                f.write(f'\nconfig={config}\n=================BEGIN A NEW RUNz({grd})====================\n\n')
            f.write(an)
            f.write(f'\n{i} done_____________________________\n')
            ans.append(an)
            # assert(0)
            if i % 5 == 0:
                print(f"第{i}次预测完成")
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"总共耗时{elapsed_time}秒")
    # print(eval_generated_ans(ans, ground_ans, contrast_ans)

    






    
 