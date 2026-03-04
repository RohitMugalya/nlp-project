import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from rag_core import RAGSystem
from evaluator import Evaluator

st.set_page_config(layout="wide", page_title="Live RAG System")

@st.cache_resource
def get_rag_system():
    return RAGSystem()

@st.cache_resource
def get_evaluator():
    return Evaluator()

st.title("Live RAG with Retrieval Depth Sensitivity Evaluation")

if not os.getenv("GOOGLE_API_KEY"):
    api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.sidebar.success("API Key set!")
        # Re-initialize to pickup API key
        st.cache_resource.clear()
        rag_system = get_rag_system()
        evaluator = get_evaluator()
    else:
        st.sidebar.warning("Please enter GOOGLE_API_KEY to proceed.")
        st.stop()
else:
    rag_system = get_rag_system()
    evaluator = get_evaluator()

tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Documents", "💬 Adaptive RAG Chat", "🧪 Evaluation Setup", "📊 Dashboard Visualization"])

with tab1:
    st.header("Upload & Index Documents")
    uploaded_files = st.file_uploader("Upload PDFs or DOCXs", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Process & Index Documents"):
        if uploaded_files:
            with st.spinner("Indexing documents..."):
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                for file in uploaded_files:
                    path = os.path.join(temp_dir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(path)
                
                rag_system.ingest_documents(file_paths)
                st.success(f"Successfully indexed {len(file_paths)} documents!")
                
                if "indexed_files" not in st.session_state:
                    st.session_state["indexed_files"] = []
                
                for f in uploaded_files:
                    if f.name not in st.session_state["indexed_files"]:
                        st.session_state["indexed_files"].append(f.name)
        else:
            st.error("Please upload documents first.")
            
    if "indexed_files" in st.session_state and st.session_state["indexed_files"]:
        st.write("### Currently Indexed Documents:")
        for name in st.session_state["indexed_files"]:
            st.write(f"- {name}")

with tab2:
    st.header("Live Chat (Adaptive Retrieval)")
    query = st.text_input("Ask a question about your documents:")
    if st.button("Ask"):
        if query:
            with st.spinner("Generating answer..."):
                retrieved_nodes = rag_system.adaptive_retrieve(query)
                if not retrieved_nodes:
                    st.warning("No relevant context found.")
                else:
                    k_used = len(retrieved_nodes)
                    st.info(f"💡 Adaptive RAG chose K={k_used} based on similarity scores.")
                    
                    response = rag_system.generate_response(query, retrieved_nodes)
                    st.write("### Response:")
                    st.write(response)
                    
                    st.write("---")
                    st.write("### Document Contributions:")
                    scores = [getattr(node, 'score', 0) for node in retrieved_nodes]
                    docs = [node.metadata.get("file_name", "unknown") for node in retrieved_nodes]
                    
                    if sum(scores) > 0:
                        df_contrib = pd.DataFrame({"Document": docs, "Contribution": [s/sum(scores) * 100 for s in scores]})
                        df_contrib = df_contrib.groupby("Document").sum().reset_index()
                        fig = px.pie(df_contrib, names="Document", values="Contribution", title=f"Contribution for K={k_used}")
                        st.plotly_chart(fig)

with tab3:
    st.header("Experimental Evaluation")
    
    eval_query = st.text_input("Evaluation Query:")
    if "indexed_files" in st.session_state and st.session_state["indexed_files"]:
        relevant_docs = st.multiselect("Select Ground Truth Documents for Query:", options=st.session_state["indexed_files"])
    else:
        st.warning("Please upload and index documents first.")
        relevant_docs = []
        
    if st.button("Run Full K-Sensitivity Evaluation"):
        if eval_query and relevant_docs:
            with st.spinner("Running evaluation for K=1 to 5... This might take a minute."):
                evaluator.evaluate_query(eval_query, relevant_docs, rag_system, k_values=[1, 2, 3, 4, 5])
                st.success("Evaluation complete! Check Dashboard for results.")
        else:
            st.error("Please provide both a query and select at least one relevant document.")

with tab4:
    st.header("Dashboard Visualization")
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r") as f:
            try:
                eval_data = json.load(f)
            except json.JSONDecodeError:
                eval_data = []
            
        if eval_data:
            df = pd.DataFrame(eval_data)
            
            queries = df["query"].unique()
            selected_query = st.selectbox("Select Query to Visualize:", queries)
            
            df_q = df[df["query"] == selected_query].copy()
            
            max_lat = df_q["latency"].max() if df_q["latency"].max() > 0 else 1.0
            df_q["norm_latency"] = df_q["latency"] / max_lat
            df_q["Quality_Score"] = (0.4 * df_q["recall_at_k"]) + (0.3 * df_q["grounding_score"]) - (0.2 * df_q["hallucination_rate"]) - (0.1 * df_q["norm_latency"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Retrieval Metrics vs K")
                fig1 = go.Figure()
                for metric in ["hit_at_k", "recall_at_k", "precision_at_k", "mrr"]:
                    fig1.add_trace(go.Scatter(x=df_q["K"], y=df_q[metric], mode='lines+markers', name=metric))
                fig1.update_layout(xaxis_title="K", yaxis_title="Score")
                st.plotly_chart(fig1)
                
            with col2:
                st.subheader("Response Quality vs K")
                fig2 = go.Figure()
                for metric in ["grounding_score", "hallucination_rate"]:
                    fig2.add_trace(go.Scatter(x=df_q["K"], y=df_q[metric], mode='lines+markers', name=metric))
                fig2.update_layout(xaxis_title="K", yaxis_title="Score")
                st.plotly_chart(fig2)
                
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Latency vs K")
                fig3 = px.line(df_q, x="K", y="latency", markers=True, title="Response Latency vs K (seconds)")
                st.plotly_chart(fig3)
                
            with col4:
                st.subheader("Analysis Graphs")
                fig4 = px.scatter(df_q, x="hallucination_rate", y="recall_at_k", text="K", 
                                  title="Recall vs Hallucination Rate",
                                  labels={"hallucination_rate": "Hallucination Rate", "recall_at_k": "Recall@K"})
                fig4.update_traces(textposition='top center')
                st.plotly_chart(fig4)
                
            st.subheader("Overall Quality Score vs K")
            fig_qs = px.line(df_q, x="K", y="Quality_Score", markers=True, title="Quality Score vs K")
            st.plotly_chart(fig_qs)
            
            st.subheader("Document Contribution per K")
            selected_k = st.selectbox("Select K for Document Contribution:", df_q["K"].unique())
            
            k_data = df_q[df_q["K"] == selected_k].iloc[0]
            contribs = k_data.get("doc_contributions", {})
            
            if contribs:
                df_contrib = pd.DataFrame(list(contribs.items()), columns=["Document", "Contribution"])
                df_contrib = df_contrib.groupby("Document").sum().reset_index()
                fig_contrib = px.pie(df_contrib, names="Document", values="Contribution", title=f"Document Contribution for K={selected_k}")
                st.plotly_chart(fig_contrib)
            else:
                st.write("No document contributions available for this K.")
    else:
        st.info("Run an evaluation in the 'Evaluation Setup' tab to see metrics.")
