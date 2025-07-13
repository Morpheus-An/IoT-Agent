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




def ChatModel(model, device="cuda", temperature=0.9):
    # assert model in ["gpt3.5", "gpt4", "llama2", "Mistral", "gemini-pro", "claude"]
    # if model in ["gpt4", "gpt3.5"]:
    if "gpt" in model:
        set_openAI_key_and_base(True, set_proxy=PROXY)
        generator = OpenAIGenerator(model=MODEL[model], generation_kwargs={
            "temperature": temperature,
        } )
        return generator
    elif model == "llama2":

        generator = HuggingFaceLocalGenerator(
            model=MODEL["llama2"],
            task="text-generation",
            generation_kwargs={
                   "temperature": temperature,
            },
            device=ComponentDevice.from_str(device)
        )

        generator.warm_up()
        return generator 
    elif model == "Mistral":
        generator = HuggingFaceLocalGenerator(
            model=MODEL["Mistral"],
            task="text-generation",
            generation_kwargs={
                   "temperature": temperature,
            },
            device=ComponentDevice.from_str(device)
        ) 
        return generator
    elif "gemini" in model:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
        generator = GoogleAIGeminiGenerator(
            model=model,
            generation_config={
                   "temperature": temperature,
            }, # config可以参考https://ai.google.dev/api/python/google/generativeai/types/GenerationConfig
        )
        return generator
    elif "claude" in model:
        os.environ["ANTHROPIC_API_KEY"] = CLAUDE_KEY
        generator = AnthropicGenerator(
            model="claude-3-5-sonnet-20240620",
            generation_kwargs={
                "temperature": temperature,
            }
            
        )
        return generator 

