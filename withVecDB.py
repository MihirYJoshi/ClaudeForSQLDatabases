import os
from langchain import LLMChain
from langchain.llms import Anthropic
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
import pymysql
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
import os
import pandas as pd

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = ""
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "gamefrc2024"

def createIndex(index_name):
    if index_name in pc.list_indexes():
        pc.delete_index(index_name)

    if index_name not in pc.list_indexes():
        #CREATE INDEX
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )

def loadDocuments(index_name, document): # "data/2024GameManual.pdf"
    # Use PyPDFLoader instead of PyPDFDirectoryLoader
    loader = PyPDFLoader(document)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    docs = text_splitter.split_documents(documents)


    PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)



vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Define prompt
template = '''You are an FRC 2024 game expert that interprets results from a database query result, refers to context when needed, and provides answers to questions

Context: {context};
Conversation history: {conversation_history};

Query: {query};
Query Result: {result};

Table Schema:
[`matchKey` varchar(60) NOT NULL PRIMARY KEY (teamNumber-matchNumber), `scout` varchar(60) NOT NULL (scouter name), 
`matchNumber` varchar(10) NOT NULL (Match Number), `teamNumber` varchar(10) NOT NULL (Team Number), 
`autoMobility` tinyint(1) NOT NULL (0: Did not Exit Zone During Autonomous Mode, 1:Exited Zone During Autonomous Mode), `autoAmpNote` smallint(6) NOT NULL (Scored in AMP During Autonomous Mode), 
`autoSpeakerNote` smallint(6) NOT NULL (Scored in Speaker During Autonomous Mode), `autoPath` longtext NOT NULL (Empty Column), 
`teleopAmpNote` smallint(6) NOT NULL (Scored in Amp During Teleoperated Mode), `teleopSpeaker` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode), 
`teleopSpeakerAmplified` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode While the Speaker was Amplified), 
`teleopTrap` smallint(6) NOT NULL (Scored in Trap During Teleoperated Mode), `climb` varchar(100) DEFAULT NULL (NONE: No climb, PARKED: No climb but in zone, ONSTAGE: Climbed on Chain), 
`climbSpotlighted` tinyint(1) NOT NULL (0: Not Spotlighted, 1: Climbed on Chain and Spotlighted), `climbHarmony` tinyint(1) NOT NULL (0: Not Climbed on Chain With Others, 1: Climbed on Chain With Others), 
`cannedComments` text DEFAULT NULL (Comments in Multiple Choice Format), `textComments` text DEFAULT NULL (Comments in Free Response Format)];

Question: {question};

Based the above information, provide an answer with an in-depth analysis using the data and context of the FRC 2024 game. 
Use your knowledge of the game and point values to place importance on the right aspects of the data you are given
'''

prompt = ChatPromptTemplate.from_template(template)

os.environ["ANTHROPIC_API_KEY"] = ""
model = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=1000
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain with additional inputs
rag_chain = (
    {
        "context": lambda x: format_docs(vectorstore.as_retriever().get_relevant_documents(x["question"])),
        "question": lambda x: x["question"],
        "conversation_history": lambda x: x["conversation_history"],
        "query": lambda x: x["query"],
        "result": lambda x: x["result"]
    }
    | prompt
    | model
    | StrOutputParser()
)

# Initialize Anthropic client
anthropic = Anthropic(api_key="")

def generate_sql_query(prompt, conversation_history):
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        system = '''You are an SQL query generator. Use the provided schema and prompt to generate accurate and relevant SQL queries that will get all the necessary information or to answer a prompt. 
            If a prompt is not relevant to the data, provide Nothing "''',
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type":"text",
                        "text": f"""Given the following schema for `24_3476_caoc_data` table: 
                            [`matchKey` varchar(60) NOT NULL PRIMARY KEY (teamNumber-matchNumber), `scout` varchar(60) NOT NULL (scouter name), 
                            `matchNumber` varchar(10) NOT NULL (Match Number), `teamNumber` varchar(10) NOT NULL (Team Number), 
                            `autoMobility` tinyint(1) NOT NULL (0: Did not Exit Zone During Autonomous Mode, 1:Exited Zone During Autonomous Mode), `autoAmpNote` smallint(6) NOT NULL (Scored in AMP During Autonomous Mode), 
                            `autoSpeakerNote` smallint(6) NOT NULL (Scored in Speaker During Autonomous Mode), `autoPath` longtext NOT NULL (Empty Column), 
                            `teleopAmpNote` smallint(6) NOT NULL (Scored in Amp During Teleoperated Mode), `teleopSpeaker` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode), 
                            `teleopSpeakerAmplified` smallint(6) NOT NULL (Scored in Speaker During Teleoperated Mode While the Speaker was Amplified), 
                            `teleopTrap` smallint(6) NOT NULL (Scored in Trap During Teleoperated Mode), `climb` varchar(100) DEFAULT NULL (NONE: No climb, PARKED: No climb but in zone, ONSTAGE: Climbed on Chain), 
                            `climbSpotlighted` tinyint(1) NOT NULL (0: Not Spotlighted, 1: Climbed on Chain and Spotlighted), `climbHarmony` tinyint(1) NOT NULL (0: Not Climbed on Chain With Others, 1: Climbed on Chain With Others), 
                            `cannedComments` text DEFAULT NULL (Comments in Multiple Choice Format), `textComments` text DEFAULT NULL (Comments in Free Response Format)];

                            Conversation history:
                            {conversation_history}

                            The query should output between 8 and 15 unless explicitly told otherwise.
                            You can Average within a column but DO NOT add or average multiple columns with eachother. Keep each column separate.
                            If you transform, name the column [transformation _ original column name]
                            Do not assume what Score means. Only output either Average, Median, Mode, or Sum

                            Generate an SQL query that will get all data to answer the following prompt: {prompt}

                            
                            Provide only the SQL query without any additional explanation. If a question is not relevant to the data, provide an empty response"""
                    }
                ]
            }
        ]
    )
    return response.content[0].text.strip()
        

def execute_query(sql):
    # Database connection parameters
    connection_params = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'Scouting24',
        'port': 8889
    }

    try:
        connection = pymysql.connect(**connection_params)
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
        return pd.DataFrame(results, columns=columns)
    except Exception as e:
        return None
    

def main():
    conversation_history = []
    while True:
        question = input("Enter your question about the 24_3476_caoc_data table (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        conversation_history.append(f"Human: {question}")

        query = generate_sql_query(prompt, "\n".join(conversation_history))
        print(query)

        result = execute_query(query)
        if result is not None:
            print("\nQuery Result:")
            print(result)


            final_answer = rag_chain.invoke({
                "question": question, 
                "conversation_history": conversation_history,
                "query": query,
                "result": result
                })
            conversation_history.append(f"Claude: {final_answer}")
            
            print("\nClaude: " + final_answer)
            conversation_history.append(f"Claude: {final_answer}")
        else:
            final_answer = rag_chain.invoke({
                "question": question, 
                "conversation_history": conversation_history,
                "query": "No Query Executed",
                "result": "No Query Executed"
                })
            conversation_history.append(f"Claude: {final_answer}")
            print("No Query Executed")
            print("\nClaude: " + final_answer)
            conversation_history.append(f"Claude: {final_answer}")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()