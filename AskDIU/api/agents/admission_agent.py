from dotenv import load_dotenv
import os
from copy import deepcopy
from .utils import get_chatbot_response, get_embedding, get_groq_chatbot_response
from openai import OpenAI
from pinecone import Pinecone
from groq import Groq
load_dotenv()

class AdmissonAgent:
    def __init__(self):
        # Load environment variables
        required_env_vars = [
            "RUNPOD_TOKEN", "RUNPOD_CHATBOT_URL", "RUNPOD_EMBEDDING_URL",
            "MODEL_NAME", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GROQ_API_KEY", "GROQ_MODEL","EMBEDDING_MODEL"
        ]
        for var in required_env_vars:
            if not os.getenv(var):
                raise EnvironmentError(f"Missing required environment variable: {var}")

        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.embedding_client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_EMBEDDING_URL"),
        )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = os.getenv("GROQ_MODEL")
        self.model_name = os.getenv("MODEL_NAME")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL")
        self.system_prompt = """ 
            You are a helpful AI assistant for Daffodil International University(DIU) Admisson Agent.  
            Your task is to provide accurate and relevant information to prospective students about admission-related queries Shortly in astructure way.  
            You will use the provided contexts to answer the user's query effectively.  
            Your response should be informative and concise, addressing the user's concerns regarding admission procedures, requirements, and deadlines.  
            If you are unsure about the answer, you can ask the user for more details or suggest contacting the university's admission office for assistance.  
            """

    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        index = self.pc.Index(index_name)
        return index.query(
            namespace="ns1",
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )

    def get_response(self, messages):
        messages = deepcopy(messages)
        user_message = messages[-1]['content']
        try:
            embedding = get_embedding(self.embedding_client, self.embedding_model_name, user_message)[0]
            result = self.get_closest_results(self.index_name, embedding)

            # Check if matches exist and extract source knowledge safely
            if result.get('matches'):
                source_knowledge = "\n".join([
                    x['metadata'].get('text', '').strip() + '\n'
                    for x in result['matches']
                    if 'metadata' in x and 'text' in x['metadata']
                ])
            else:
                source_knowledge = "No relevant matches found in the database."
        except Exception as e:
            source_knowledge = f"Error during embedding or query: {str(e)}"

        search_prompt = [
            {"role": "system", "content": "You are a helpful AI assistant with web scraping abilities. Gather **Admisson** related informatiion based on user message  from Daffodil International University.REMEMBER You are Web scraping Agent In a Admisson info Agent Of Daffodil International University Bangladesh"},
            {"role": "user", "content": user_message}
        ]
        try:
            response = get_groq_chatbot_response(self.groq_client, self.groq_model, search_prompt)
        except Exception as e:
            response = f"Error fetching Groq response Do it With what have information You Have: {str(e)}"

        # print(f"Groq Response: {response}")
        # print(f"Source Response: {source_knowledge}")
        
        prompt = f"""
        Using the contexts below merge knowlegde from all texts, answer the question shortly give little bit of details if user asks.

        Contexts:
        {source_knowledge + response}

        Query: {user_message}
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": self.system_prompt}] + messages[-1:]
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        if chatbot_output is None:
             return {"role": "assistant", "content": "Sorry, I could not process your request at this time."}
        # print(f"User message: {user_message}")
        # print(f"Embedding: {embedding}")
        # print(f"Result: {result}")
        # print(f"Source Knowledge: {source_knowledge}")
        
        return self.postprocess(chatbot_output)


    def postprocess(self, output):
        return {
            "role": "assistant",
            "content": output,
            "memory": {"agent": "admission_agent"}
        }
