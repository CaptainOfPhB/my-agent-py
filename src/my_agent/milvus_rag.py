import os
from glob import glob

from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

load_dotenv()

EMBEDDING_DIM = 1024
COLLECTION_NAME = "milvus_faq"


# 初始化 OpenAI
openai_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

milvus_client = MilvusClient(uri="./milvus_demo.db")


# 将文本嵌入到向量空间中
def embed_text(text):
    return (
        openai_client.embeddings.create(
            input=text, model="qwen3.7-text-embedding", dimensions=EMBEDDING_DIM
        )
        .data[0]
        .embedding
    )


# 创建向量数据，插入 milvus 数据库
def insert_vectors():
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.drop_collection(COLLECTION_NAME)

    schema = milvus_client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM
    )
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)

    milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    # 将文档内容加载到内存中
    text_lines = []

    for file_path in glob("/Users/captainofphb/Downloads/en/faq/*.md", recursive=True):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")

    data = []

    for i, text in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        data.append({"id": i, "vector": embed_text(text), "text": text})

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)


insert_vectors()

# 搜索向量数据库
question = embed_text("How is data stored in milvus?")

search_res = milvus_client.search(
    limit=3,  # Return top 3 results
    data=[question],
    collection_name=COLLECTION_NAME,
    search_params={"metric_type": "COSINE", "params": {}},  # Cosine similarity
    output_fields=["text"],  # Return the text field
)


retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]

# 拼接成上下文
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)

SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""

USER_PROMPT = f"""
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

response = openai_client.chat.completions.create(
    model="qwen3.8-omni-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)

print(response.choices[0].message.content)
