from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
model = 'gpt-5-mini'
#db_name = 'enc_records'
collection_name='medical_notes'

PERSIST_DIR = r"YOUR LOCAL PATH TO VECTOR DATABSE"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2"
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=collection_name,  
    embedding_function=embeddings   
)

llm = ChatOpenAI(model_name=model)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=retriever)

# Question and Answer - 'message' is the query(question)
def rag_chat(message,history):
    response = conversation_chain.invoke({'question':message})
    return response['answer']


## DARK MODE

# Define this variable and then pass js=force_dark_mode when creating the Interface

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
# User interface with Gradio

with gr.Blocks(js=force_dark_mode) as UI:

    gr.ChatInterface(
        rag_chat, 
        type='messages',
        title="Medical RAG Pipeline",
        description= "Get patient records, diagnoses, summaries, and medical notes."
    )

UI.launch(inbrowser=True, share=False)

