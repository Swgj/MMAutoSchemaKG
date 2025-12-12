import streamlit as st
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# 引入项目组件
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.vectorstore.embedding_model import EmbeddingAPI
from atlas_rag.multimodal.hipporag_adapter import Neo4jToHippoAdapter
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.retriever.inference_config import InferenceConfig
from atlas_rag.multimodal.multimodal_react import MultimodalReAct

# 加载环境变量
load_dotenv()

# --- 页面配置 ---
st.set_page_config(page_title="Multimodal KG RAG Demo", layout="wide")
st.title("🧩 Multimodal Knowledge Graph RAG")
st.markdown("Ask questions about your data, and see how the system retrieves both **Text** and **Images** from the Graph.")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("Configuration")
    neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("Neo4j User", value="neo4j")
    neo4j_password = st.text_input("Neo4j Password", value="password", type="password")
    
    st.divider()
    
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=3)
    hipporag_mode = st.selectbox("HippoRAG Mode", ["query2node", "query2edge", "ner"])

# --- 核心资源加载 (带缓存) ---
@st.cache_resource
def load_system_resources(uri, user, pwd):
    """
    初始化模型和加载图谱数据。
    使用 cache_resource 装饰器，确保只加载一次，不用每次刷新页面都重跑。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found! Please check your .env file.")
        return None, None, None

    with st.spinner("🚀 Loading Knowledge Graph & Models... (This may take a moment)"):
        # 1. 初始化模型
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 注意：这里的模型名要和你 vector_store.py 里算向量时用的一致
        embedding_model = EmbeddingAPI(client, model_name="gemini-embedding-001")
        llm_generator = LLMGenerator(client, model_name="gemini-2.5-flash")
        
        # 2. 从 Neo4j 加载数据到内存
        adapter = Neo4jToHippoAdapter(uri, user, pwd, embedding_model, database_name="locomo-hard-0")
        data = adapter.load_data()
        adapter.close()
        
        return llm_generator, embedding_model, data

# 加载资源
llm_gen, emb_model, kg_data = load_system_resources(neo4j_uri, neo4j_user, neo4j_password)

if not kg_data:
    st.stop() # 如果加载失败，停止运行

# 初始化 Retriever (配置可能随侧边栏变化，所以不缓存这个对象，只缓存 data)
config = InferenceConfig()
config.hipporag_mode = hipporag_mode
config.topk_nodes = 20
config.ppr_alpha = 0.85

retriever = HippoRAGRetriever(
    llm_generator=llm_gen,
    sentence_encoder=emb_model,
    data=kg_data,
    inference_config=config
)

image_map = kg_data.get("image_map", {})

# --- 主交互区 ---

query = st.chat_input("Ask a question about Caroline, Melanie, or anything in the graph...")

if query:
    # 1. 显示用户问题
    with st.chat_message("user"):
        st.write(query)

    # 2. 执行检索
    with st.chat_message("assistant"):
        st.write("🔍 **Retrieving context from Knowledge Graph...**")

        # 调用 HippoRAG 检索（用于可视化上下文）
        retrieved_contents, retrieved_ids = retriever.retrieve(query, topN=top_k)

        if not retrieved_ids:
            st.warning("No relevant information found.")
        else:
            # --- 3. 展示检索结果 (可视化核心) ---
            tabs = st.tabs([f"Chunk {i+1}" for i in range(len(retrieved_ids))])

            for i, tab in enumerate(tabs):
                content = retrieved_contents[i]
                chunk_id = retrieved_ids[i]

                with tab:
                    st.caption(f"Source ID: `{chunk_id}`")

                    # 解析文本中的图片标签并渲染
                    parts = re.split(r'(\(Image: IMG_[^\)]+\))', content)

                    for part in parts:
                        img_match = re.match(r'\(Image: (IMG_[^\)]+)\)', part)
                        if img_match:
                            img_id = img_match.group(1)
                            img_url = image_map.get(img_id)

                            if img_url:
                                st.image(img_url, caption=f"{img_id}", width=400)
                            else:
                                st.warning(f"⚠️ Image found in text [{img_id}] but URL missing in map.")
                        else:
                            if part.strip():
                                st.markdown(part)

            # --- 4. 使用多模态 ReAct 生成回答 ---
            st.divider()
            st.markdown("### 🤖 AI Answer (Multimodal ReAct)")

            with st.spinner("Running multimodal ReAct reasoning..."):
                mm_react = MultimodalReAct(llm_gen)
                answer, history = mm_react.generate_with_rag_react(
                    question=query,
                    retriever=retriever,
                    image_map=image_map,
                    max_iterations=5,
                    max_new_tokens=1024,
                    logger=None,
                )

                st.write(answer)

                # 展示 ReAct 推理过程（可选）
                if history:
                    with st.expander("Show ReAct search history"):
                        for i, (thought, action, observation) in enumerate(history):
                            st.markdown(f"**Step {i+1}**")
                            st.markdown(f"- **Thought**: {thought}")
                            st.markdown(f"- **Action**: {action}")
                            st.markdown(f"- **Observation**: {observation}")
                            st.markdown("---")