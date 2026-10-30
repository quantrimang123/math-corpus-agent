import os
import urllib.request
import xml.etree.ElementTree as ET
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import TOPIC_LIST, MAX_RESULTS_PER_TOPIC, OUTPUT_FILENAME

# Cấu hình mô hình nhúng local miễn phí cho GitHub Actions
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def fetch_arxiv_documents():
    documents = []
    print("🚀 [Phân tích & Ingestion] Đang quét dữ liệu từ arXiv API trên GitHub Actions...")
    
    for topic in TOPIC_LIST:
        try:
            print(f"   [-] Đang tải tài liệu cho chủ đề: {topic}")
            encoded_topic = topic.replace(" ", "+")
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_topic}&max_results={MAX_RESULTS_PER_TOPIC}"
            
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
            print(f"   [!] Lỗi khi tải chủ đề '{topic}': {e}")
            
    return documents

def run_pipeline():
    all_documents = fetch_arxiv_documents()

    if not all_documents:
        print("[!] Không thu thập được tài liệu nào. Dừng chương trình.")
        return

    print("🧠 [Core] Đang xây dựng Vector Index từ dữ liệu thô...")
    index = VectorStoreIndex.from_documents(all_documents)
    base_query_engine = index.as_query_engine(similarity_top_k=2)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="math_corpus_tool",
                description="Kho tri thức chứa các bài toán và định lý toán học đã được thu thập."
            ),
        )
    ]

    print("🤖 [APISAI] Khởi tạo Sub-Question Agent để tổng hợp...")
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True
    )

    prompt = "Hãy tổng hợp các phương pháp giải quyết, định lý và công thức cốt lõi từ các tài liệu toán học vừa thu thập."
    
    print(f"\n💡 Đang thực hiện truy vấn tổng hợp...")
    response = query_engine.query(prompt)

    print("=" * 50)
    print("✨ KẾT QUẢ TỪ AGENT:")
    print("=" * 50)
    print(response)

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(str(response))
    print(f"\n💾 [Kết quả] Đã lưu báo cáo thành công vào '{OUTPUT_FILENAME}'!")

if __name__ == "__main__":
    run_pipeline()
