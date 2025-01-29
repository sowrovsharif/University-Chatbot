from agents import (GuardAgent,AdmissonAgent,AgentProtocol,AcademicAgent,ClassificationAgent,RecommendationAgent)

class AgentController():
    def __init__(self):
        self.guard_agent = GuardAgent()
        self.classification_agent = ClassificationAgent()
        
        self.agent_dict: dict[str, AgentProtocol] = {
            "admission_agent": AdmissonAgent(),
            "academic_info_agent": AcademicAgent(),
            "recommendation_agent":RecommendationAgent()
        }
    
    def get_response(self,input):
        # Extract User Input
        job_input = input["input"]
        messages = job_input["messages"]

        # Get GuardAgent's response
        guard_agent_response = self.guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            return guard_agent_response
        
        # Get ClassificationAgent's response
        classification_agent_response = self.classification_agent.get_response(messages)
        chosen_agent = classification_agent_response["memory"]["classification_decision"]

        # Get the chosen agent's response
        agent = self.agent_dict[chosen_agent]
        response = agent.get_response(messages)

        return response