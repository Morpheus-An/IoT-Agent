from imports import *

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

def ChatModel(model, device="cuda"):
    assert model in ["gpt3.5", "gpt4", "llama2", "Mistral-7b", "Gemini", "Haiku"]
    if model in ["gpt4", "gpt3.5"]:
        set_openAI_key_and_base(False, set_proxy=PROXY)
        generator = OpenAIGenerator(model=MODEL[model])
        return generator
    elif model == "llamma2":
        pass 
    elif model == "Mistral-7b":
        pass 
    elif model == "Gemini":
        pass 
    elif model == "Haiku":
        pass 
