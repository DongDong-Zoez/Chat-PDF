# file backend/server/apps/llm/llama_7b_chat_hf/model.py
from PyPDF2 import PdfReader
from apps.llm.llama_7b_chat_hf.utils import load_model, build_pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss, Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ConversationalRAG:
    def __init__(self, pretrained_model_name_or_path: str = "your path", quantize = False, **kwargs):
        # self.model, self.tokenizer = load_model(pretrained_model_name_or_path, quantize)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.quantize = quantize

    def init_model(self):
        self.model, self.tokenizer = load_model(self.pretrained_model_name_or_path, self.quantize)

    def build_chain(self, docs, **kwargs):
        llm = self.build_llm_chain(**kwargs)

        text = self._get_pdf_context(docs)
        chunks = self._get_context_chunks(text)
        vectorstore = self._get_vectorstore(chunks)
        retriever = vectorstore.as_retriever()

        chain, memory = self.build_conversational_chain(llm, retriever)

        self.chain = chain
        self.memory = memory
        self.retriever = retriever

    def build_llm_chain(self, **kwargs):
        llm = build_pipeline(
            task = "text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            **kwargs
        )
        llm = HuggingFacePipeline(pipeline=llm)
        return llm

    def build_conversational_chain(self, llm, retriever):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True, 
            output_key="answer", 
            input_key="question",
        )
        condense_prompt, standalone_prompt = self._bulid_prompt()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False,
            chain_type="stuff",
            return_generated_question=True,
            condense_question_llm=llm,
            condense_question_prompt=condense_prompt,
            combine_docs_chain_kwargs={"prompt": standalone_prompt}
        )

        return chain, memory
    
    def clear_memory(self):
        if getattr(self, "memory", None):
            self.memory.clear()
        else:
            print("Memory is not initialized, you should build chain first.")
    
    @staticmethod
    def _get_pdf_context(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    @staticmethod
    def _get_context_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    @staticmethod   
    def _get_vectorstore(chunks):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2", 
            model_kwargs={
                "device": "cuda"
            }
        )
        # vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_db")
        return vectorstore
    
    @staticmethod
    def _bulid_prompt():
        condense_prompt = """Given the following conversation and a follow-up message, \
        rephrase the follow-up message to a stand-alone question or instruction that \
        represents the user's intent, add all context needed if necessary to generate a complete and \
        unambiguous question or instruction, only based on the history, don't make up messages. \
        Maintain the same language as the follow up input message.

        Chat History:
        {chat_history}

        Follow Up Input: {question}
        Standalone question or instruction:"""

        condense_prompt = PromptTemplate(
            template=condense_prompt, input_variables=["chat_history", "question"]
        )

        standalone_prompt = """
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know.
            Don't try to make up an answer.
            {context}

            Respond in the persona of {persona}

            Question: {question}
            Answer: 
            """

        persona = "teaching assistant"

        standalone_prompt = PromptTemplate(
                input_variables=["question", "context", "persona"],
                template=standalone_prompt 
            )
        standalone_prompt = standalone_prompt.partial(persona=persona)
        return condense_prompt, standalone_prompt

    def preprocessing(self, input_data):
        if isinstance(input_data, str):
            input_data = {
                "question": input_data
            }
        return input_data

    def predict(self, input_data):
        return self.chain(input_data)

    def postprocessing(self, input_data):
        return {
            **input_data,
            "status": "OK"
        }

    def generate(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction