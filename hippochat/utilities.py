import langchain
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.document_loaders.parsers import GrobidParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.utilities import ApifyWrapper
from langchain.schema import Document



import os, requests
from glob import glob


prefix="""
    You are a highly esteemed academic researcher.
    Answer the question based on the chat history(delimited by <hs></hs>) and context(delimited by <ctx> </ctx>) below. 
    You can consult the general knowledge outside the provided context, but do not make up untrue answers when you do not know. Answer as detailed as possible.
    
    <ctx>
    {context}
    </ctx>
    
    <hs>
    {chat_history}
    </hs>"""





# nest_asyncio.apply()

##===== Helper Functions ====================================
class Blob():
    def __init__(self, filepath):
        self.source = filepath

def get_doc_type(file_path):
    if os.path.isfile(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in ['.pdf', '.docx', 'md']:
            return ext[1:]
    return False




def is_openai_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True

def set_openai_key(key):
    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key

##====== Basic Setup =======================================


embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, 
                                               chunk_overlap=48)

from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

##===== Class Definition ===================================

class Summarizer():
    def __init__(self, llm_model="gpt-3.5-turbo") -> None:
        self.summarize_chain = load_summarize_chain(llm=ChatOpenAI(model=llm_model,
                                                                   temperature=0,
                                                                   streaming=True),
                                                    chain_type="map_reduce")
    
    def __load_pdf__(self, file_path):
        if get_doc_type(file_path)=="pdf":
            self.docs = UnstructuredPDFLoader(file_path).load_and_split()
        else:
            print("Documents other than pdf are not ready.")
    
    def run(self, file_path):
        self.__load_pdf__(file_path)
        return self.summarize_chain.run(self.docs)
    
    
class ChromaDB():
    
    def __init__(self, directory_name, 
                 prefix=prefix,
                 llm_model="gpt-3.5-turbo",
                 embeddings=embeddings,
                 text_splitter=text_splitter):
        
        self.prefix = prefix
        self.llm = ChatOpenAI(model=llm_model, temperature=0, streaming=True)
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        self.directory_name = directory_name
        self.documents = []
        if not os.path.exists(self.directory_name):
            os.mkdir(directory_name)
        self.db = Chroma(persist_directory=directory_name,
                         embedding_function=embeddings)
        self.get_QA_chain()
            
    def ingest(self, grobid=False, callback=None):
        message = ""
        if len(self.documents) > 0:
            try:
                if grobid:
                    self.db.add_documents(self.documents)
                else:
                    texts = self.text_splitter.split_documents(self.documents)
                    self.db.add_documents(texts)
                
                self.db.persist()
                self.documents = []
                message = "Sucessfully ingested"
            except Exception as e:
                message = "Ingest failed"
                print(e)
            finally:
                if callback:
                    callback(message)
                    
    def ingest_directory(self, directory):
        path = os.path.join(directory, '**/*.pdf')
        files = list(glob(path, recursive=True))
        if len(files):
            self.ingest_pdfs(files)
        else:
            print("No pdf files could be found")
        
        
    def ingest_pdfs(self, files, grobid=False, callback=None):

        blobs = [Blob(filepath=file) for file in files]
            
        if grobid:
            self.documents = []
            parser = GrobidParser(segment_sentences=False)
            for blob in blobs:
                texts = parser.lazy_parse(blob)
                # for text in texts:
                #     text.metadata = {"source": os.path.basename(blob.source)}
                self.documents.extend(texts)
        else:
            loaders = [UnstructuredPDFLoader(file) for file in files]
            self.documents = []
            for loader in loaders:
                self.documents.extend(loader.load())
        self.ingest(grobid=grobid, callback=callback)
                    
    def ingest_by_crawl(self, root_url, callback=None):
        response = requests.get(root_url)
        if response.status_code == 200:
            
            apify = ApifyWrapper()
            loader = apify.call_actor(
                actor_id="apify/website-content-crawler",
                run_input={"startUrls": [{"url": root_url}]},
                dataset_mapping_function=lambda item: Document(
                    page_content=item["text"] or "", metadata={"source": item["url"]})
                )
            self.documents = loader.load()
            self.ingest(callback=callback)
        # response = requests.get(root_url)
        # if response.status_code == 200:
        #     self.documents = RecursiveUrlLoader(root_url).load()
        #     self.ingest(callback=callback)
        #     pass
        # else:
        #     return f"Error due to {response.reason} error"
            
    def get_metadata(self, key="source"):
        dd = self.db.get()
        if len(dd['metadatas']) and key in dd['metadatas'][0].keys():
            return list(set(os.path.basename(item[key]) for item in dd['metadatas']))
        else:
            return []
    
    def get_QA_chain(self, type=None, k=4):
        """type can be ['similarity', 'mmr', 'compress', 'multi']"""
        
        if self.db is not None:
            if type=="compress":
            
                compressor = LLMChainExtractor.from_llm(self.llm)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.db.as_retriever(search_kwargs={'k':2*k})
                )
            elif type=="multi":
                retriever = MultiQueryRetriever.from_llm(
                    retriever=self.db.as_retriever(search_kwargs={'k':k}),
                    llm=self.llm)
            elif type=="mmr":
                retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k':k})        
            else: # similarity
                retriever = self.db.as_retriever(search_type='similarity', search_kwargs={'k':k})

            self.QA_chain = ConversationalRetrievalChain.from_llm(llm=self.llm, 
                                        retriever=retriever,
                                        memory=memory,
                                        return_source_documents=True)
        else:
            print("ChromaDB not yet loaded")
            
            
    def run(self, query, callbacks=None):
        if self.QA_chain is not None:
            query = f"{self.prefix}\n\nQuestion: {query}"
            result = self.QA_chain({'question': query}, callbacks=callbacks)
            return result
        else:
            print("QA chain not yet ready")
            return None