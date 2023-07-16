import os, tempfile
from glob import glob
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler, StreamingStdOutCallbackHandler




from utilities import (
    ChromaDB,
    Summarizer,
    get_doc_type
)


# Setting page title and header
st.set_page_config(page_title="Anselm", page_icon=":woozy_face:")
st.markdown("<h1 style='text-align: center;color: orange;'>Knowledge based Question/Answering</h1>", unsafe_allow_html=True)



# Helper functions



def clear_messages():
    st.session_state['messages'] = []
    
def clear_everything():
    clear_messages()
    st.session_state['todo'] = None
    st.session_state['root_url'] = None
    st.session_state['grobid'] = False
    st.session_state['new_project_input'] = ""
    
def get_directories(path="./"):
    directories = [d for d in os.listdir(path) 
                   if os.path.isdir(d) and d.startswith("+")
                   ]
    directories.insert(0, "None")
    return directories

def get_files(path="../uploaded"):
    file_paths = [os.path.join(path, f) for f in os.listdir(path) if  os.path.isfile(os.path.join(path,f))]

    return file_paths
    
def post_ingestion(message):
    file_paths = get_files()
    for f in file_paths:
        os.remove(f)
    st.error(message)
    
    
def create_new_project():
    if len(st.session_state['new_project_input']) > 0:
        new_directory = '+' + st.session_state['new_project_input']
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
            st.session_state['directories'] = get_directories()
            
            st.session_state['current_project'] = new_directory
            new_project_container.warning(f"New project created: {st.session_state['current_project']}")
        else:
            st.error("Directory already exists")
        

        
def project_changed():
    if st.session_state['current_project'] != "None":
        st.session_state['db'] = ChromaDB(st.session_state['current_project'], 
                                          llm_model=st.session_state['llm_model'])
    else:
        st.session_state['db'] = None
        

def add_documents(files):
    """ 
    Add uploaded files from st.file_upload() to ChromaDB
    """
    if len(files) > 0:
        file_paths = []
        for file in files:
            file_path = os.path.join("../uploaded/", file.name)
            with open(file_path,"wb") as f: 
                f.write(file.getbuffer())
                file_paths.append(file_path)

        st.session_state['db'].ingest_pdfs(file_paths,
                                       grobid=st.session_state['grobid'], 
                                       callback=post_ingestion)
        
def add_root_url(root_url):
    return st.session_state['db'].ingest_by_crawl(root_url)
        
def get_unique_source_documents(source_documents):
    if len(source_documents) == 0:
        return None
    else:
        unique_source_documents = [source_documents[0]]
        for source in source_documents[1:]:
            if unique_source_documents[-1].page_content != source.page_content:
                unique_source_documents.append(source)
        return unique_source_documents

def display_sources(response, expanded=False):
    unique_sources = get_unique_source_documents(response['source_documents'])
    if unique_sources:
        unique_labels = [f"Source {i+1}" for i in range(len(unique_sources))]
                
        with st.expander("Answer based on:", expanded=expanded):
                tabs = st.tabs(unique_labels)
                for j, source in enumerate(unique_sources):
                    tabs[j].markdown(f"Document: **:red[{source.metadata['source'].replace('uploaded/', '')}]**")
                    tabs[j].markdown(source.page_content)

# Initialise session state variables
st_callback = StreamlitCallbackHandler(st.container())

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'directories' not in st.session_state:
    st.session_state['directories'] = []
if 'current_project' not in st.session_state:
    st.session_state['current_project'] = "None"
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'todo' not in st.session_state:
    st.session_state['todo'] = None
if 'root_url' not in st.session_state:
    st.session_state['root_url'] = None
if 'new_project_input' not in st.session_state:
    st.session_state['new_project_input'] = False
if 'grobid' not in st.session_state:
    st.session_state['grobid'] = False
if 'retrieval_type' not in st.session_state:
    st.session_state['retrieval_type'] = 'similarity'
if 'k' not in st.session_state:
    st.session_state['k'] = 4
if 'llm_model' not in st.session_state:
    st.session_state['llm_model'] = "gpt-3.5-turbo"
    
    
if 'error' not in st.session_state:
    st.session_state['error'] = None

st.session_state['directories'] = get_directories()

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation

with st.sidebar:
    
    
    directory_container = st.container()
    new_project_container = st.container()
    task_container = st.container()
    option_container = st.container()
    status_container = st.container()
    

    # create selectbox with the foldernames
    
    directory_container.selectbox(
        label="Choose a directory", 
        options=st.session_state['directories'],
        # index=st.session_state['directories'].index(st.session_state['current_project']),
        key="current_project",
        on_change=project_changed
        )
    
    with new_project_container: # new project
        if st.button("Create New Project?", type='primary'):
            st.text_input("Name of the new project", 
                          key="new_project_input",
                          on_change=create_new_project)
            
            
    with task_container:
        st.session_state['todo'] = st.selectbox(
            label='Choose a task type',
            options=['Ingest pdf documents', 'Ingest website by crawling']
        )
        
        if st.session_state['todo'] == 'Ingest pdf documents':
            # st.session_state['grobid'] = st.radio("Grobid segmentation?",
            #                                 [False, True],
            #                                 horizontal=True)
            uploaded_files = st.file_uploader("pdfs to ingest..", type="pdf", 
                            accept_multiple_files=True, label_visibility='hidden')
            
            add_documents(uploaded_files)
            
        elif st.session_state['todo'] == "Ingest website by crawling":
            st.session_state['root_url'] = st.text_input("Enter the root URL to crawl", placeholder="http(s)://")
            if len(st.session_state['root_url']):
                st.session_state['error']=add_root_url(st.session_state['root_url'])
    
    with option_container:
        st.radio("LLM model",
                 ["gpt-3.5-turbo", "gpt-4"],
                 key="llm_model",
                 on_change=project_changed,
                 horizontal=True)
        st.session_state['retrieval_type'] = st.radio("Retrival type",
                                                      ['similarity', 'mmr', 'compress', 'multi'])
        st.session_state['k'] = st.slider("k value", 2, 8, 4, step=1)    
            
    with status_container:
        if st.session_state['error']:
            st.error(st.session_state['error'])
        if st.session_state['db']:
            st.success(f"Database loaded: {st.session_state['current_project']}")
            st.write("**Contents of DB:**")
            for i in st.session_state['db'].get_metadata():
                st.markdown("- " + i)
        
#=================================================================================
        
query = st.chat_input("Enter the query:")
tab1, tab2 = st.tabs(['New Q/A', 'Previous Q/As'])

with tab2:
    if len(st.session_state.messages):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message['role'] == "user":
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"]['answer'])
                    display_sources(message["content"])
        if st.button("Clear conversations?", type='primary'):
            clear_messages()
    else:
        st.info("No conversations recorded.")


with tab1:        
    if query:
    # Display user message in chat message container
        st.session_state['db'].get_QA_chain(type=st.session_state['retrieval_type'],
                                            k=st.session_state['k'])

        st.chat_message("user").markdown(query)
        with st.container():
            st_callback = StreamlitCallbackHandler(parent_container=st.container())
            if response := st.session_state['db'].run(query, callbacks=[st_callback]):
                st.session_state['messages'].append({"role": "user", "content": query})
                st.session_state['messages'].append({"role": "assistant", "content": response})
                if len(response['source_documents']) > 0:
                    with st.chat_message("assistant"):
                        display_sources(response, expanded=True)



css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)