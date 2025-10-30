# \RAG\models.py
#可用模型列表，以及获得访问模型的客户端
#实际使用时可以根据自己的实际情况调整
ALI_TONGYI_API_KEY_SYSVAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max-latest"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING = "text-embedding-v4"
ALI_TONGYI_RERANK = "gte-rerank-v2"
DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


import os
from langchain_openai import ChatOpenAI
import inspect
from langchain_community.embeddings import BaichuanTextEmbeddings, DashScopeEmbeddings, HunyuanEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank

from enum import Enum
class Constants(Enum):
    API_KEY_SYSVAR_NAME = ALI_TONGYI_API_KEY_SYSVAR_NAME
    BASE_URL = ALI_TONGYI_URL
    LLM_MODEL = ALI_TONGYI_MAX_MODEL
    EMBEDDING_MODEL = ALI_TONGYI_EMBEDDING
    RERANK_MODEL = ALI_TONGYI_RERANK
    REASONER_MODEL = ALI_TONGYI_DEEPSEEK_R1

def get_lc_model_client(api_key=os.getenv(Constants.API_KEY_SYSVAR_NAME.value), base_url=Constants.BASE_URL.value
                        , model=Constants.LLM_MODEL.value, verbose=False,temperature = 0.7, debug=False):
    function_name = inspect.currentframe().f_code.co_name
    if(verbose):
        print(f"{function_name}:{base_url},{model}")
    if(debug):
        print(f"{function_name}:{base_url},{model},{api_key}")
    return ChatOpenAI(api_key=api_key, base_url=base_url,model=model)

def get_ali_model_client(model=ALI_TONGYI_MAX_MODEL,temperature = 0.7,verbose=False, debug=False):
    '''
    过LangChain获得阿里大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为阿里百炼里的qwen-max-latest，温度=0.7
    '''
    return get_lc_model_client(api_key=os.getenv(Constants.API_KEY_SYSVAR_NAME.value), base_url=ALI_TONGYI_URL
                        ,model=model,temperature =temperature,verbose=verbose, debug=debug )

def get_ds_model_client(model=DEEPSEEK_CHAT_MODEL,temperature = 0.7,verbose=False, debug=False):
    '''
    过LangChain获得DeepSeek大模型的客户端
    可以通过传入model，temperature 两个参数来覆盖默认值
    verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
    :return: 指定平台和模型的客户端，默认模型为DeepSeek的deepseek-chat，温度=0.7
    '''
    return get_lc_model_client(api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME), base_url=DEEPSEEK_URL
                        ,model=model,temperature =temperature,verbose=verbose, debug=debug )


def get_ali_embeddings():
    '''
    通过LangChain获得一个阿里通义千问嵌入模型的实例
    :return: 阿里通义千问嵌入模型的实例，目前为text-embedding-v3
    '''
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME)
)


def get_ali_clients():
    '''
    产生阿里大模型客户端和嵌入模型的客户端
    :return: 阿里大模型客户端和嵌入模型的客户端
    '''
    return get_ali_model_client(),get_ali_embeddings()


def get_ali_rerank(top_n=3):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    '''
    return DashScopeRerank(
        model=ALI_TONGYI_EMBEDDING, dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME),
        top_n=top_n
)
