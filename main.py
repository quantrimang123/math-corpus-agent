import os
from llama_index.core import VectorStoreIndex
from llama_index.readers.arxiv import ArxivReader
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from config import TOPIC_LIST, MAX_RESULTS_PER_TOPIC, OUTPUT_FILENAME

def run_pipeline():
    print("🚀 [Phân tích & Ingestion] Bắt đầu quét dữ liệu từ arXiv...")
    
    all_documents = []
    for topic in TOPIC_LIST:
        try:
            print(f"   [-] Đang tải tài liệu cho chủ đề: {topic}")
            reader = ArxivReader(search_query=topic, max_results=MAX_RESULTS_PER_TOPIC)
            docs = reader.load_data()
            all_documents.extend(docs)
        except Exception as e:
            print(f"   [!] Lỗi khi tải chủ đề '{topic}': {e}")

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

    # Lưu kết quả
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write(str(response))
    print(f"\n💾 [Kết quả] Đã lưu báo cáo thành công vào '{OUTPUT_FILENAME}'!")

if __name__ == "__main__":
    run_pipeline()
