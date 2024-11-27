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

# GPU 0번 사용 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

model.to(device)




text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=False,
    max_new_tokens=5,
)

prompt_template = """
### [INST]
Instruction: Review the provided trademark registration records and trademark law case precedents to determine if '{question}' is eligible for trademark registration.

1. **Analyze**: Check for potential conflicts with similar terms in the trademark registration records and examine any precedent cases in the trademark law context provided.
2. **Apply**: Use the specific legal principles and rulings from the trademark law precedents, considering any relevant factors like distinctiveness, descriptiveness, likelihood of confusion, and prior usage.
3. **Answer**: Respond with Only "가능" or "불가능" in Korean, and briefly specify any key issue if it affects eligibility.

### Trademark Law Case Precedents
{context}

### Trademark Registration Records
The following records document previous trademark applications. Compare these with the requested term to assess any overlap or conflict.

{patent}

### QUESTION:
Can I register trademark name? Identify any issues or conflicts based on the provided legal precedents.
{question}
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
df = pd.read_csv('data.csv', usecols=['등록일자', '상표한글명', '최종처분코드명'], encoding='utf-8')

# Convert each row to a Document object
documents = [
    Document(
        page_content=f"등록일자: {row['등록일자']}, 상표한글명: {row['상표한글명']}, 최종처분코드명: {row['최종처분코드명']}",
        metadata={"등록일자": row['등록일자'], "상표한글명": row['상표한글명'], "최종처분코드명": row['최종처분코드명']}
    )
    for _, row in df.iterrows()
]


from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf1 = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

hf2 = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(texts, hf1)

patent = FAISS.from_documents(documents, hf2)

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

import csv

# CSV 파일 열기
with open("test.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)  # CSV 읽기 객체 생성
    i = 0
    with open("result.txt", mode="w", encoding="utf-8") as txt_file:
        for row in reader:
            if i == 3:
                break
            question = f"등록하고 싶은 상표한글명은 '{row['상표한글명']}'이야. 출원일자는 {row['상표한글명']}에 하는거야"
            result = rag_chain.invoke(question)
            print(result)
            i += 1

'''
import torch

result = rag_chain.invoke("배달 예스맨")
print(result)
'''