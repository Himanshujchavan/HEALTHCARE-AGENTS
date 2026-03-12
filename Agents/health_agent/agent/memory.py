from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory


def create_memory():
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        chat_memory=message_history,
        return_messages=True,
        output_key='output'
    )
    return memory