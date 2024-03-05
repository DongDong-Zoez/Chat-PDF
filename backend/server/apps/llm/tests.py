import inspect

from django.test import TestCase

from apps.llm.llama_7b_chat_hf.model import ConversationalRAG
from apps.llm.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "question": "hello"
        }
        rag_chain = ConversationalRAG("/home/s311657007/llm_chat_model")
        rag_chain.init_model()
        rag_chain.build_chain(["/home/s311657007/doc_reader/hwk04_new.pdf"])
        response = rag_chain.generate(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('answer' in response)
        # self.assertEqual('<=50K', response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "conversational_rag"
        algorithm_object = ConversationalRAG("/home/s311657007/llm_chat_model")
        algorithm_name = "Conversational RAG"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "DongDong"
        algorithm_description = "Conversational RAG"
        algorithm_code = inspect.getsource(ConversationalRAG)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)