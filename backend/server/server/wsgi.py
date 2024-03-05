import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.llm.registry import MLRegistry
from apps.llm.llama_7b_chat_hf.model import ConversationalRAG

try:
    registry = MLRegistry() # create ML registry
    # Conversational RAG
    rag_chain = ConversationalRAG("meta-llama/Llama-2-7b-chat-hf")
    rag_chain.init_model()
    # add to ML registry
    registry.add_algorithm(
        endpoint_name="conversational_rag",
        algorithm_object=rag_chain,
        algorithm_name="ConversationalRAG",
        algorithm_status="production",
        algorithm_version="0.0.1",
        owner="DongDong",
        algorithm_description="Build a web application with conversational RAG",
        algorithm_code=inspect.getsource(ConversationalRAG)
    )

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))