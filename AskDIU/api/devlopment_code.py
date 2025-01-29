import streamlit as st
from agents import (GuardAgent, AgentProtocol,AcademicAgent, ClassificationAgent, RecommendationAgent, AdmissonAgent)

def main():
    st.markdown('<h1 style="color: #4CAF50;">Welcome to Ask DIU🤖</h1>', unsafe_allow_html=True)    
    st.write("Connect with our AI assistant for all your university-related queries. Whether it's about admissions, programs, or campus life, we're here to help you.")

    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()

    agent_dict: dict[str, AgentProtocol] = {
        "admission_agent": AdmissonAgent(),
        "academic_info_agent": AcademicAgent(),
        "recommendation_agent": RecommendationAgent(),
    }

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()
        
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Typing..."):
                # Get GuardAgent's response
                guard_agent_response = guard_agent.get_response(st.session_state.messages)
                if guard_agent_response["memory"]["guard_decision"] == "not allowed":
                    response_text = "Sorry😞, I'm here to assist you with queries related to Daffodil International University, including admissions, programs, fees, and campus information. If you have any questions about the university, feel free to ask! For topics outside the university's scope, I may not have the information you're looking for."
                else:
                    # Get ClassificationAgent's response
                    classification_agent_response = classification_agent.get_response(st.session_state.messages)
                    chosen_agent = classification_agent_response["memory"]["classification_decision"]
                    
                    # Get the chosen agent's response
                    agent = agent_dict.get(chosen_agent, None)
                    if agent:
                        response = agent.get_response(st.session_state.messages)
                        response_text = response["content"]
                    else:
                        response_text = "I'm not sure how to respond to that."
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.write(response_text)
                st.rerun()

if __name__ == "__main__":
    main()