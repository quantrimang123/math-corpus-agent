import os
import urllib.request
import xml.etree.ElementTree as ET
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from config import TOPIC_LIST, MAX_RESULTS_PER_TOPIC, OUTPUT_FILENAME

# Lấy Gemini API Key từ môi trường GitHub Secrets
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

# 1. Cấu hình LLM bằng Gemini 2.5 Flash mới nhất (Cực nhanh, thông minh và miễn phí)
Settings.llm = Gemini(model="gemini-2.5-flash")

# 2. Cấu hình Embedding chạy local bằng HuggingFace (Miễn phí 100%)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_arxiv_source(topic, max_results):
    documents = []
    try:
        encoded_topic = topic.replace(" ", "+")
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_topic}&max_results={max_results}"
        
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            
            doc_text = f"Title: {title}\nAbstract: {summary}"
            documents.append(Document(text=doc_text))
    except Exception as e:
        print(f"   [!] Lỗi nguồn arXiv cho chủ đề '{topic}': {e}")
    return documents

def aggregate_all_sources():
    all_documents = []
    print("🚀 [Ingestion] Đang tổng hợp dữ liệu từ arXiv trên GitHub Actions...")
    
    for topic in TOPIC_LIST:
        print(f"   [-] Đang xử lý chủ đề: {topic}")
        arxiv_docs = fetch_arxiv_source(topic, MAX_RESULTS_PER_TOPIC)
        all_documents.extend(arxiv_docs)
        
    print(f"✅ Tổng số tài liệu thu thập được: {len(all_documents)}")
    return all_documents

def run_pipeline():
    all_documents = aggregate_all_sources()
    if not all_documents:
        print("[!] Không thu thập được tài liệu nào.")
        return

    print("🧠 [Core] Đang xây dựng Vector Index bằng Local Embedding...")
    index = VectorStoreIndex.from_documents(all_documents)
    base_query_engine = index.as_query_engine(similarity_top_k=2)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="math_corpus_tool",
                description="Kho tri thức chứa các bài toán và định lý toán học."
            ),
        )
    ]

    print("🤖 [APISAI] Khởi tạo Sub-Question Agent với Gemini 2.5 Flash...")
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True
    )

    prompt = "Hãy tổng hợp các phương pháp giải quyết, định lý và công thức cốt lõi từ các tài liệu toán học vừa thu thập."
    
    print(f"\n💡 Đang thực hiện truy vấn tổng hợp...")
    response = query_engine.query(prompt)

    print("=" * 50)
    print("✨ KẾT QUẢ TỪ GEMINI 2.5 AGENT:")
    print("=" * 50)
    print(response)

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(str(response))
    print(f"\n💾 Đã lưu file '{OUTPUT_FILENAME}' thành công!")

if __name__ == "__main__":
    run_pipeline()
