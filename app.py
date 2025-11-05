from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import logging
import asyncio

# ======================================================================
# 环境准备部分
# =====================================================================
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
rag_system = None
is_initialized = False
initialization_lock = asyncio.Lock()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动中...")
    # 不在这里初始化 RAG，先让服务器启动
    yield
    logger.info("应用关闭中...")

app = FastAPI(title="个人信息查询系统", lifespan=lifespan)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# =====================================================================
# 数据模型
# =====================================================================
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# =====================================================================
# 文档读取与检索
# =====================================================================
# 从PDF文档中提取文字
def extract_text_from_pdf(pdf_path: str) -> str:
    """从PDF提取文本 - 使用 pypdf"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF文本提取失败: {e}")
        raise Exception(f"PDF文本提取失败: {e}")


# 加载并分割文档
def load_split_docs(text: str):
    """直接处理文本而不是文件路径"""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 将文本转换为Document对象
    docs = [Document(page_content=text)]

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    # 执行分割
    split_docs = text_splitter.split_documents(docs)
    return split_docs


# TF-IDF检索器
class TFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.doc_texts = [doc.page_content for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def get_relevant_documents(self, query, k=3):
        """获取相关文档"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 获取最相似的k个文档
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_docs = [self.documents[i] for i in top_k_indices]

        return top_k_docs


# 向量检索器类
class VectorRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def get_relevant_documents(self, query, k=4):
        return self.vectorstore.similarity_search(query, k=k)


# 混合检索器
class HybridRetriever:
    def __init__(self, vector_retriever, tfidf_retriever, vector_weight=0.7, tfidf_weight=0.3):
        self.vector_retriever = vector_retriever
        self.tfidf_retriever = tfidf_retriever
        self.vector_weight = vector_weight
        self.tfidf_weight = tfidf_weight

    def get_relevant_documents(self, query, k=4):
        """混合检索获取相关文档"""
        # 从两种检索器获取文档
        vector_docs = self.vector_retriever.get_relevant_documents(query, k=k)
        tfidf_docs = self.tfidf_retriever.get_relevant_documents(query, k=k)

        # 合并文档并去重
        all_docs = []
        seen_contents = set()

        # 添加向量检索结果（带权重）
        for doc in vector_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                doc.metadata["score"] = self.vector_weight
                all_docs.append(doc)

        # 添加TF-IDF检索结果（带权重）
        for doc in tfidf_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                doc.metadata["score"] = self.tfidf_weight
                all_docs.append(doc)
            else:
                # 如果文档已存在，增加其分数
                for existing_doc in all_docs:
                    if existing_doc.page_content == doc.page_content:
                        existing_doc.metadata["score"] += self.tfidf_weight

        # 按分数排序并返回前k个
        all_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
        return all_docs[:k]


# 自定义检索器 - 继承BaseRetriever
class CustomRetriever:
    def __init__(self, hybrid_retriever):
        self.hybrid_retriever = hybrid_retriever

    def get_relevant_documents(self, query: str):
        return self.hybrid_retriever.get_relevant_documents(query)


# 初始化检索系统
def initialize_retrieval_system():
    """初始化混合检索系统"""
    try:
        pdf_filename = "TanYu_PM.pdf"
        if not os.path.exists(pdf_filename):
            raise FileNotFoundError(f"PDF文件 {pdf_filename} 不存在")

        logger.info("开始提取PDF文本...")
        # 提取和分割文本
        text = extract_text_from_pdf(pdf_filename)
        split_docs = load_split_docs(text)
        logger.info(f"文本分割完成，共 {len(split_docs)} 个文档块")

        logger.info("初始化向量检索...")
        # 初始化向量检索
        from langchain_community.vectorstores import Chroma
        from models import get_ali_clients
        
        llm, embeddings_model = get_ali_clients()
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory="vector_store"
        )

        # 创建向量检索器
        vector_retriever = VectorRetriever(vectorstore)

        # 创建TF-IDF检索器
        tfidf_retriever = TFIDFRetriever(split_docs)

        # 创建混合检索器
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            tfidf_retriever=tfidf_retriever,
            vector_weight=0.7,
            tfidf_weight=0.3
        )

        # 包装成LangChain兼容的检索器
        custom_retriever = CustomRetriever(hybrid_retriever=hybrid_retriever)

        return custom_retriever, split_docs, llm

    except Exception as e:
        logger.error(f"初始化检索系统失败: {e}")
        raise


# 初始化RAG系统
def initialize_rag_system():
    """初始化RAG问答系统"""
    try:
        logger.info("开始初始化RAG系统...")
        retriever, split_docs, llm = initialize_retrieval_system()

        # 创建提示模板
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import RetrievalQA
        
        prompt_template = """你是一个专业的个人信息助手，请根据以下上下文信息准确回答用户的问题。如果上下文中没有相关信息，请如实告知你不知道，不要编造信息。

上下文信息:
{context}

用户问题: {question}

请根据上下文提供准确、详细的回答，保持专业和友好的语气:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # 创建检索QA链 - 不再返回源文档
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,  # 设置为False，不返回源文档
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info("RAG系统初始化完成")
        return qa_chain

    except Exception as e:
        logger.error(f"RAG系统初始化失败: {e}")
        raise


# 延迟初始化函数
async def initialize_rag():
    global rag_system, is_initialized
    async with initialization_lock:
        if is_initialized:
            return True
            
        try:
            logger.info("开始初始化 RAG 系统...")
            rag_system = initialize_rag_system()
            is_initialized = True
            logger.info("RAG 系统初始化完成")
            return True
        except Exception as e:
            logger.error(f"RAG 初始化失败: {e}")
            is_initialized = False
            return False


# 带重试的初始化函数
async def initialize_rag_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            success = await initialize_rag()
            if success:
                return True
        except Exception as e:
            logger.warning(f"RAG 初始化第 {attempt + 1} 次失败: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)  # 等待5秒后重试
    return False


# =====================================================================
# API路由
# =====================================================================
@app.get("/")
async def read_index():
    """返回前端页面"""
    return FileResponse("static/index.html")


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """回答问题"""
    global rag_system, is_initialized
    
    # 如果还没初始化，先初始化
    if not is_initialized:
        logger.info("RAG系统未初始化，开始初始化...")
        success = await initialize_rag_with_retry()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="系统正在初始化，请稍后重试。如果长时间无法初始化，请检查PDF文件是否存在。"
            )
    
    try:
        logger.info(f"收到问题: {request.question}")

        # 使用RAG系统回答问题
        result = rag_system.invoke({"query": request.question})

        # 只返回答案，不返回源文档
        response = AnswerResponse(
            answer=result["result"]
        )

        return response

    except Exception as e:
        logger.error(f"回答问题失败: {e}")
        raise HTTPException(status_code=500, detail=f"回答问题失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    status = "initialized" if is_initialized else "initializing"
    pdf_exists = os.path.exists("TanYu_PM.pdf")
    
    return {
        "status": status,
        "port": os.getenv("PORT", "8000"),
        "pdf_exists": pdf_exists,
        "message": "个人信息查询系统运行正常"
    }


@app.get("/init-status")
async def init_status():
    """初始化状态检查端点"""
    pdf_exists = os.path.exists("TanYu_PM.pdf")
    
    return {
        "rag_initialized": is_initialized,
        "pdf_file_exists": pdf_exists,
        "vector_store_exists": os.path.exists("vector_store"),
        "message": "检查系统初始化状态"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting uvicorn on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
