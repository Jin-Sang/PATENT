import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders.csv_loader import CSVLoader
import csv

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model_id = "NCSOFT/Llama-VARCO-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")




text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=False,
    max_new_tokens=500,
)

prompt_template = """
### [INST]
Instruction: You're a patent registration examiner. Based on existing patent registration records and knowledge, provide the outcome of the patent application for the given trademark name.
Here is context to help:

{context}

### EXISTING PATENT
Here are the results of previous patent applications. Please refer to them in your response.
Here is existing patent record table:

{patent}

### QUESTION:
Is it possible to register {question} as a patent name?
한글로 답변해줘 
[/INST]
 """

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "patent", "question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = prompt | llm


loader = PyPDFLoader("/home/jskim/lecture/RAG_project/example.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# Patent
# Increase the field size limit
from langchain.schema import Document
import pandas as pd

# Load only specific columns using pandas
df = pd.read_csv('data.csv', usecols=['등록일자', '상표한글명', '상표영문명', '최종처분코드명'], encoding='utf-8')

# Convert each row to a Document object
documents = [
    Document(
        page_content=f"등록일자: {row['등록일자']}, 상표한글명: {row['상표한글명']}, 상표영문명: {row['상표영문명']}, 최종처분코드명: {row['최종처분코드명']}",
        metadata={"등록일자": row['등록일자'], "상표한글명": row['상표한글명'], "상표영문명": row['상표영문명'], "최종처분코드명": row['최종처분코드명']}
    )
    for _, row in df.iterrows()
]


from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)


db = FAISS.from_documents(texts, hf)

patent = FAISS.from_documents(documents, hf)

retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k': 3}
                        )

patent_retriever = patent.as_retriever(
                                search_type="similarity",
                                search_kwargs={'k':3}
                        
                        )

rag_chain = (
 {"context": retriever, "patent": patent_retriever,"question": RunnablePassthrough()}
    | llm_chain
)

import warnings
warnings.filterwarnings('ignore')

import torch

result = rag_chain.invoke("현대")
print(result)

