def get_chatbot_response(client,model_name,messages,temperature=0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response


def get_embedding(embedding_client,model_name,text_input):
    output = embedding_client.embeddings.create(input = text_input,model=model_name)
    
    embedings = []
    for embedding_object in output.data:
        embedings.append(embedding_object.embedding)

    return embedings

def get_groq_chatbot_response(client, model_name,messages,temperature=0):

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # a list of messages from the user and chatbot, in chronological order
        temperature=temperature
    )
    response_message = response.choices[0].message
    return response_message.content

