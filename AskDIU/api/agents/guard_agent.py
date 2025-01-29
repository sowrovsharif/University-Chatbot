from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response
from openai import OpenAI
load_dotenv()

class GuardAgent:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

    def get_response(self,messages):
        messages = deepcopy(messages)

        system_prompt = """  
             You are a helpful AI assistant for Daffodil International University. Your task is to determine if a user's query is relevant to the university based on the following categories:
                1.Admissions – Requirements, deadlines, programs, and application process.
                2.Tuition and Fees – Costs, scholarships, financial aid, and payment procedures.
                3.Subject Recommendation – Program or subject suggestions based on student preferences.
                4.Course Information – Information about specific courses and programs.
                5.Campus Information – Campus location and related details.
                6.Faculty and Departments – Information about faculty members, research areas, and department contacts.
                7.Student Services – Library access, IT support, career counseling, health services, and student clubs.
                8.Extracurricular Activities – Sports, cultural events, clubs, and student organizations.
                9.Internships and Job Placement – Career opportunities, industry collaborations, and internship programs.
                10.Research and Innovation – Research centers, publications, funding, and collaboration opportunities.
                11.Alumni Network – Alumni associations, networking opportunities, and success stories.
             You should NOT answer queries unrelated to Daffodil International University, personal advice, or confidential internal operations.
            
            Your response should be structured in the following JSON format:  
            {  
                "chain of thought": "Analyze the user’s query based on the allowed topic categories and justify whether it falls within the university-related scope.",  
                "decision": "allowed" or "not allowed",  
                "message": "Leave this empty if the query is allowed, otherwise respond with 'Sorry, I can only assist with queries related to Daffodil International University. How can I assist you today?'"  
            }  
            """  

        input_messages = [{"role": "system", "content": system_prompt}] + messages[-1:]

        chatbot_output = get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = json.loads(output)
        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {"agent":"guard_agent",
                       "guard_decision": output['decision']
                      }
        }
        return dict_output



    