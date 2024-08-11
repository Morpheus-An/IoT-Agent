from imports import *

from common.utils import * 
from common.args import args 


# # 获取当前的日期和时间
# now = datetime.datetime.now()

# # 使用f-string格式化并打印当前时间
# print(f"当前时间：{now:%Y-%m-%d %H:%M:%S}")

def eval_an(an):
    answer_pattern = r'\[ANSWER\]:\s*(no_person|have_person)'
    answer_match = re.findall(answer_pattern, an)
    
    if answer_match:
        if answer_match[0] == 'no_person':
            return 1
        else:
            return 0
    else:
        first_no_person = an.rfind('no_person')
        first_yes_person = an.rfind('have_person')
        if first_no_person < first_yes_person:
            return 1
        elif first_no_person > first_yes_person:
            return 0
        else:
            return -1

def extract_coordinates(text):
    matches = re.findall(r"\s*[\[\(]\s*(-?\d+(\.\d*)?),\s*(-?\d+(\.\d*)?)\s*[\]\)]", text)
    if matches:
        last_match = matches[-1]
        x = float(last_match[0]) if '.' in last_match[0] else int(last_match[0])
        y = float(last_match[2]) if '.' in last_match[2] else int(last_match[2])
        return [x, y]
    else:
        return None

def extract_coordinates_llama(text):
    # 匹配带小数点或整数的坐标，中间用空格分隔
    matches = re.findall(r"\s*[\[\(]\s*(-?\d+(\.\d*)?)\s+(-?\d+(\.\d*)?)\s*[\]\)]", text)
    if matches:
        last_match = matches[-1]
        # 使用 float() 直接转换，因为整数也可以被正确转换
        x = float(last_match[0])
        y = float(last_match[2])
        return [x, y]
    return None



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
    # data_dict, label_dict = read_machine_data()
    ground_ans = args.ground_an
    data_dict, label_dict = read_IoT_data(args.task_type, cls_num=args.cls_num) # type: ignore
    start_time = time.perf_counter()
    cur_time = datetime.datetime.now()
    config = f"""
curtime: {cur_time: %Y-%m-%d %H:%M:%S}
task_type: {args.task_type}
cls_num: {args.cls_num}
sample_num: {args.sample_num}
no_domain_knowledge: {args.no_domain_knowledge}
no_demo_knowledge: {args.no_demo_knowledge}
model: {args.model} on {args.device}
generate results are saved in {args.output_file_path}"""
    # with rerieval:
    # 首先，准备好document_store并写入:
    # 得到指定文件夹下所有文件的路径
    if not args.no_domain_knowledge:
        KB_paths, _ = get_knowledge_paths(args)
        document_store_domain = InMemoryDocumentStore()
        splitter_kwags_domain = {"split_by": "sentence", "split_length": 2}
        embedded_document_store_KB = prepare_and_embed_documents(document_store_domain, KB_paths, device=args.device, splitter_kwards=splitter_kwags_domain)
    if not args.no_demo_knowledge: # TODO
        document_store_demo = InMemoryDocumentStore()
        splitter_kwags_demo = {"split_by": "passage", "split_length": 1}
        Demo_paths, _ = get_knowledge_paths(args, False, data_dict=data_dict)
        meta_data = [
        {
            "label": file_path.split('/')[-1][0:-len("_i.txt")]
        }
        for file_path in Demo_paths
        ]
        embedded_document_store_DM = prepare_and_embed_documents(document_store_demo, Demo_paths, device=args.device, splitter_kwards=splitter_kwags_demo, meta_data=meta_data)

    ans = []
    correct = []
    previous_locals = None
    locals_to_release = None
    with open(args.output_file_path, "a") as f:
        for i in range(0, args.sample_num):
            
            # pdb.set_trace()

            if previous_locals is not None and locals_to_release is None:
                locals_to_release = set(locals().keys()).difference(previous_locals) 
            if previous_locals == None:
                previous_locals = set(locals())
            # 释放掉local所占用的内存
            if locals_to_release is not None:
                for lc in locals_to_release:
                    if lc in locals():
                        del locals()[lc]
                        torch.cuda.empty_cache()


            generator = ChatModel(args.model, args.device, args.temperature)
            ###* 在下面的函数中根据不同的task编辑相关的信息
            grd, con, template, data_des, query = task_dependent_info(args, i, data_dict, label_dict, ground_ans)
            ###*

            
            prompt_builder = PromptBuilder(template=template)
            
            # 创建pipeline:
            rag_pipeline = Pipeline()
            rag_pipeline.add_component("prompt_builder", prompt_builder)
            run_kwargs =  {"prompt_builder": {"query": query}}

            if not args.no_domain_knowledge:


                # print("#debug: in no domain knowledge")



                text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(args.device))

                embedding_retriever_domain = InMemoryEmbeddingRetriever(embedded_document_store_KB)

                keyword_retriever_domain = InMemoryBM25Retriever(embedded_document_store_KB)
                
                document_joiner_domain = DocumentJoiner()

                ranker_domain = TransformersSimilarityRanker(model=RANKER_MODEL)

                # 1. for domain-knowledge:
                rag_pipeline.add_component("text_embedder_domain", text_embedder)
                rag_pipeline.add_component("embedding_retriever_domain", embedding_retriever_domain)
                rag_pipeline.add_component("keyword_retriever_domain", keyword_retriever_domain)
                rag_pipeline.add_component("document_joiner_domain", document_joiner_domain)
                rag_pipeline.add_component("ranker_domain", ranker_domain)

                # 将components连接起来
                # 1. for domain-knowledge:
                rag_pipeline.connect("text_embedder_domain", "embedding_retriever_domain")
                rag_pipeline.connect("embedding_retriever_domain", "document_joiner_domain")
                rag_pipeline.connect("keyword_retriever_domain", "document_joiner_domain")
                rag_pipeline.connect("document_joiner_domain", "ranker_domain")
                content4retrieval_domain = gen_content4retrive_domain(args.task_type, data_des)
                rag_pipeline.connect("ranker_domain", "prompt_builder.documents_domain")

                run_kwargs["text_embedder_domain"] = {"text": content4retrieval_domain}


                # print("#debug: content4retrieval_domain is: ", run_kwargs["text_embedder_domain"])


                run_kwargs["keyword_retriever_domain"] = {"query": content4retrieval_domain}
                run_kwargs["ranker_domain"] = {"query": content4retrieval_domain}

            if not args.no_demo_knowledge: # TODO

                # print("#debug: in no demo knowledge")

                if args.task_type == "wifi_localization":

                    # print("#debug: in wifi_localization")


                    demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(args.device))
                    embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)

                    keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)
                    document_joiner_demo = DocumentJoiner()
                    ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)

                    # 2. for demo-knowledge:
                    rag_pipeline.add_component("demo_embedder", demo_embedder)
                    rag_pipeline.add_component("embedding_retriever_demo", embedding_retriever_demo)
                    rag_pipeline.add_component("keyword_retriever_demo", keyword_retriever_demo)
                    rag_pipeline.add_component("document_joiner_demo", document_joiner_demo)
                    rag_pipeline.add_component("ranker_demo", ranker_demo)

                    # 2. for demo-knowledge:
                    rag_pipeline.connect("demo_embedder", "embedding_retriever_demo")
                    rag_pipeline.connect("embedding_retriever_demo", "document_joiner_demo")
                    rag_pipeline.connect("keyword_retriever_demo", "document_joiner_demo")
                    rag_pipeline.connect("document_joiner_demo", "ranker_demo")
                    content4retrieval_demo = data_des
                    
                    rag_pipeline.connect("ranker_demo", "prompt_builder.demo")
                    run_kwargs =  {
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


                    
                else:
                    grd_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(args.device))
                    con_demo_embedder = SentenceTransformersTextEmbedder(model=EMBEDDER_MODEL, device=ComponentDevice.from_str(args.device))

                    grd_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)
                    con_embedding_retriever_demo = InMemoryEmbeddingRetriever(embedded_document_store_DM)

                    grd_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)
                    con_keyword_retriever_demo = InMemoryBM25Retriever(embedded_document_store_DM)

                    grd_document_joiner_demo = DocumentJoiner()
                    con_document_joiner_demo = DocumentJoiner()
                
                    grd_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)
                    con_ranker_demo = TransformersSimilarityRanker(model=RANKER_MODEL)

                    # 2. for demo-knowledge:
                    # 2.1 for ground-truth demo knowledge:
                    rag_pipeline.add_component("grd_demo_embedder", grd_demo_embedder)
                    rag_pipeline.add_component("grd_embedding_retriever_demo", grd_embedding_retriever_demo)
                    rag_pipeline.add_component("grd_keyword_retriever_demo", grd_keyword_retriever_demo)
                    rag_pipeline.add_component("grd_document_joiner_demo", grd_document_joiner_demo)
                    rag_pipeline.add_component("grd_ranker_demo", grd_ranker_demo)
                    # 2.2 for contrast demo knowledge:
                    rag_pipeline.add_component("con_demo_embedder", con_demo_embedder)
                    rag_pipeline.add_component("con_embedding_retriever_demo", con_embedding_retriever_demo)
                    rag_pipeline.add_component("con_keyword_retriever_demo", con_keyword_retriever_demo)
                    rag_pipeline.add_component("con_document_joiner_demo", con_document_joiner_demo)
                    rag_pipeline.add_component("con_ranker_demo", con_ranker_demo)

                    # 2. for demo-knowledge:
                    # 2.1. for ground-truth demo knowledge:
                    rag_pipeline.connect("grd_demo_embedder", "grd_embedding_retriever_demo")
                    rag_pipeline.connect("grd_embedding_retriever_demo", "grd_document_joiner_demo")
                    rag_pipeline.connect("grd_keyword_retriever_demo", "grd_document_joiner_demo")
                    rag_pipeline.connect("grd_document_joiner_demo", "grd_ranker_demo")
                    # 2.2. for contrast demo knowledge:
                    rag_pipeline.connect("con_demo_embedder", "con_embedding_retriever_demo")
                    rag_pipeline.connect("con_embedding_retriever_demo", "con_document_joiner_demo")
                    rag_pipeline.connect("con_keyword_retriever_demo", "con_document_joiner_demo")
                    rag_pipeline.connect("con_document_joiner_demo", "con_ranker_demo")
                    # rag_pipeline.draw("retriver_pipeline2.png")
                    # print("draw1 done")        
                    # content4retrieval_grd_demo = None
                    # content4retrieval_con_demo = None
                    content4retrieval_grd_demo = data_des 
                    content4retrieval_con_demo = f"ANSWER: {con}"
                    rag_pipeline.connect("grd_ranker_demo", "prompt_builder.grd_demo")
                    rag_pipeline.connect("con_ranker_demo", "prompt_builder.con_demo")

                    run_kwargs["grd_demo_embedder"] = {"text": content4retrieval_grd_demo} # type: ignore
                    run_kwargs["con_demo_embedder"] = {"text": content4retrieval_con_demo} # type: ignore
                    run_kwargs["grd_embedding_retriever_demo"] = { # type: ignore
                        "filters": {
                            "field": "meta.label",
                            "operator": "in",
                            "value": [grd]
                        },
                        "top_k": 1,
                    }
                    run_kwargs["con_embedding_retriever_demo"] = { # type: ignore
                        "filters": {
                            "field": "meta.label",
                            "operator": "in",
                            "value": [con]
                        }
                    }
                    run_kwargs["grd_keyword_retriever_demo"] = { # type: ignore
                        "query": content4retrieval_grd_demo,
                        "top_k": 1,
                        "filters": {
                            "field": "meta.label",
                            "operator": "in",
                            "value": [grd],
                        },
                    }
                    run_kwargs["con_keyword_retriever_demo"] = { # type: ignore
                        "query": content4retrieval_con_demo,
                        "top_k": 1,
                        "filters": {
                            "field": "meta.label",
                            "operator": "in",
                            "value": [con],
                        },
                    }
                    run_kwargs["grd_ranker_demo"] = { # type: ignore
                        "query": content4retrieval_grd_demo,
                        "top_k": 1,
                    }
                    run_kwargs["con_ranker_demo"] = { # type: ignore
                        "query": content4retrieval_con_demo,
                        "top_k": 1,
                    }
             
            if args.debug:
                # pdb.set_trace()
                final_prompt = rag_pipeline.run(run_kwargs)
                print(f"final_prompt is:\n{final_prompt['prompt_builder']['prompt']}")
                assert(0)
            
            rag_pipeline.add_component("llm", generator) # type: ignore
            rag_pipeline.connect("prompt_builder", "llm")
            # rag_pipeline.draw("rag_pipeline.png")

            
    
            result = rag_pipeline.run(
                run_kwargs
            )
            
            if args.model == "llama2" and args.task_type == "wifi_occupancy":
                an = result["llm"]["replies"][0][0:50]
            elif args.model == "llama2" and args.task_type == "wifi_localization":
                an = result["llm"]["replies"][0][0:70]
            else:
                an = result["llm"]["replies"][0]
            print(an)
            if i == 1:
                f.write(f'\nconfig={config}\n=================BEGIN A NEW RUN({grd})====================\n\n')
            f.write(an)
            f.write(f'\n{i} done_____________________________\n')
            ans.append(an)
            an_class = eval_an(an)
            correct.append(an_class)
            # assert(0)
            if i % 5 == 0:
                print(f"第{i}次预测完成")

        if args.task_type == "wifi_occupancy":
            if ground_ans == "no_person":
                count_ones = correct.count(1)
                ratio = count_ones / len(ans)
            else:
                count_zeros = correct.count(0)
                ratio = count_zeros / len(ans)
            print(f"correct_rate is {ratio}")
            f.write(f"correct_rate is {ratio}\n")
        elif args.task_type == "wifi_localization":
            error = []
            i = 0
            for an in ans:
                if args.model == "llama2":
                    coordinates = np.array(extract_coordinates_llama(an))
                else:
                    coordinates= np.array(extract_coordinates(an))
                gt = np.array(data_dict['val_position'][i, :])
                print(f"gt is {gt}")
                print(f"coordinates is {coordinates}")
                if gt.shape == coordinates.shape:
                    error.append(mse(coordinates,gt))

                i = i+1
            error_mean = np.mean(error)
            print(f"mse is {error_mean}")
            f.write(f"mse is {error_mean}\n")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"总共耗时{elapsed_time}秒")
    # print(eval_generated_ans(ans, ground_ans, contrast_ans)


    

