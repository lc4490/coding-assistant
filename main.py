import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone
import shutil

st.set_page_config(
    page_title="Coding Assistant",
    # page_icon=,
    # layout="wide",  # Layout options: "centered" or "wide"
    # initial_sidebar_state="expanded",  # Sidebar options: "auto", "expanded", or "collapsed"
)
st.title("AI Coding Assistant")

# OPENAI
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.1-8b-instant"
    

# Clone repository
def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.
    Args:
        repo_url: The URL of the GitHub repository.
    Returns:
        The path to the cloned repository if successful, otherwise None.
    """
    try:
        repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
        repo_path = f"./{repo_name}"  # Use a relative path

        # Check if the repository already exists
        if os.path.exists(repo_path):
            # Remove the existing repository
            shutil.rmtree(repo_path)

        # Clone the repository
        Repo.clone_from(repo_url, repo_path)
        return repo_path
    except Exception as e:
        # Handle exceptions and return None if cloning fails
        print(f"Failed to clone repository: {e}")
        return None

def get_file_content(file_path, repo_path):
    """
    Get content of a single file.

    Args:
        file_path (str): Path to the file

    Returns:
        Optional[Dict[str, str]]: Dictionary with file name and content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.

    Args:
        repo_path: Path to the local repository

    Returns:
        List of dictionaries containing file names and contents
    """
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java', '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
    IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git', '__pycache__', '.next', '.vscode', 'vendor', 'lib'}
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        # # Calculate the size of the file in bytes
                        # size_in_bytes = os.path.getsize(file_path)

                        # # Format the size in a human-readable way (e.g., KB, MB)
                        # size_kb = size_in_bytes / 1024
                        # formatted_size = f"{size_kb:.2f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"

                        # # Display the file name and size
                        # st.markdown(f"**{file_content['name']}** - *{formatted_size}*")
                        
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content
    
# Embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)
    
# RAG  
def perform_rag(query, namespace, pinecone_index, message_history):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as needed to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """

    # Include the existing message base in the conversation
    messages = [{"role": "system", "content": system_prompt}]
    for msg in message_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": augmented_query})

    # Call the model with streaming
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True
    )

    return stream


# global variables
if "project_path" not in st.session_state:
    st.session_state.project_path = None
    
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None

# clone repo
project = st.text_input("Enter the URL of your GitHub project")
if project:
    if(project != st.session_state.project_path):
        # Add a markdown status
        status = st.markdown("🔄 **Cloning repository...**")

        path = clone_repository(project)
        if path:
            status.markdown("✅ **Repository successfully cloned to:** " + path)
            # st.success(f"Repository successfully cloned to: {path}")
        else:
            status.markdown("❌ **Invalid GitHub repository URL.**")
            # st.warning("Please provide a valid GitHub repository URL.")
            st.stop()  # Stop execution if cloning fails

        # Update status
        status.markdown("🔄 **Reading main files in the repository...**")
        file_content = get_main_files_content(path)
        status.markdown("✅ **Files successfully read. Preparing to index...**")

        # Pinecone initialization
        status.markdown("🔄 **Initializing Pinecone vector store...**")
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        pinecone_index = pc.Index("codebase-rag")
        vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings())
        status.markdown("✅ **Pinecone vector store initialized.**")

        # Prepare documents
        status.markdown("🔄 **Processing documents for vectorization...**")
        documents = []
        for file in file_content:
            doc = Document(
                page_content=f"{file['name']}\n{file['content']}",
                metadata={"source": file['name']}
            )
            documents.append(doc)
        status.markdown("✅ **Documents prepared. Deleting old namespace...**")

        # delete previous namespace
        existing_namespaces = pinecone_index.describe_index_stats().get('namespaces', {})
        if project in existing_namespaces:
            pinecone_index.delete(namespace=project, delete_all=True)
            status.markdown("✅ **Old namespace deleted. Indexing new documents...**")
        else:
            status.markdown("🔄 **No existing namespace to delete. Proceeding with indexing...**")

        # Add documents to vector store
        try:
            status.markdown("🔄 **Indexing documents to Pinecone vector store...**")
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=HuggingFaceEmbeddings(),
                index_name="codebase-rag",
                namespace=project
            )
            status.markdown("✅ **Documents successfully indexed. Ready to chat!**")
        except Exception as e:
            # Log the error and display a message to the user
            # print(f"Error while indexing documents: {e}")
            status.markdown("❌ **Failed to index documents. The project might be too large.**")
            st.error("An error occurred while indexing documents. Please ensure the project is not too large and try again.")

        st.session_state.project_path = project
        st.session_state.pinecone_index = pinecone_index
        
        # remove repo
        repo_name = project.split("/")[-1]  # Extract repository name from URL
        repo_path = f"./{repo_name}"  # Use a relative path
        shutil.rmtree(repo_path)
        
    # chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("AI Coding Assistant"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # stream = client.chat.completions.create(
            #     model=st.session_state["openai_model"],
            #     messages=[
            #         {"role": m["role"], "content": m["content"]}
            #         for m in st.session_state.messages
            #     ],
            #     stream=True,
            # )
            response = st.write_stream(perform_rag(prompt, project, st.session_state.pinecone_index, st.session_state.messages))
        st.session_state.messages.append({"role": "assistant", "content": response})