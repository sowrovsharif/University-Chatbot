from dotenv import load_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response
from openai import OpenAI
load_dotenv()

class ClassificationAgent():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
    
    def get_response(self,messages):
        messages = deepcopy(messages)

        system_prompt = system_prompt = """  
            You are a helpful AI assistant for Daffodil International University.  
            Your task is to determine which agent should handle the user's input.  
            There are 3 specialized agents available to assist with different types of queries:  

            1. **admission_agent** – This agent is responsible for handling all admission-related inquiries, including:  
            - Admission requirements, eligibility criteria, and application procedures.  
            - Deadlines for application submissions and document requirements.  
            - Available programs, tuition fees, scholarships, and financial aid options.  
            - Admission test details and result announcements.  
            - Enrollment and registration processes for new students.  

            2. **academic_info_agent** – This agent manages general academic and university-related queries, such as:  
            - Course details, faculty information, class schedules, and exam timetables,course Cost.  
            - Academic policies, grading systems, and credit transfer procedures.  
            - Student services, campus facilities, and extracurricular activities.  
            - University events, workshops, and career counseling services.  
            - Policies related to attendance, internships, and graduation requirements.

            3. **Recomendation Agent** -"You are a helpful Subject Recommendation Agent for Daffodil International University.Your role is to assist students in choosing the most suitable subjects or programs based on their interests, academic goals, and career aspirations and gpa.Please provide clear, personalized, and insightful responses to help the student make an informed decision. Your response should address the following:
            -Student Interests & Goals – 
            -Inquire about the student’s academic interests, strengths, agpa and  suitable subjects or programs. 

            ### Response Format:  
            Your response should strictly follow the JSON format below:  

            {  
                "chain of thought": "Analyze the user's query against the categories of the available agents and explain why it belongs to a specific category.",  
                "decision": "admission_agent" or "recommendation_agent" or "academic_info_agent". Choose only one option.  
                "message": ""  
            }  

            """  
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_output =get_chatbot_response(self.client,self.model_name,input_messages)
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self,output):
        output = json.loads(output)

        dict_output = {
            "role": "assistant",
            "content": output['message'],
            "memory": {"agent":"classification_agent",
                       "classification_decision": output['decision']
                      }
        }
        return dict_output

    