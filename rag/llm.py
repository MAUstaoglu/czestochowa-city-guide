"""
LLM integration using Ollama with Gemma:2b model.
"""

import json
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_MODEL, OLLAMA_BASE_URL


class LLM:
    """Wrapper for Ollama LLM (Gemma:2b)."""
    
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        """Initialize the LLM client."""
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check for exact match or match without tag
                for name in model_names:
                    if name == self.model or name.startswith(self.model.split(":")[0]):
                        return True
                print(f"Model {self.model} not found. Available models: {model_names}")
                return False
            return False
        except requests.RequestException:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except requests.RequestException:
            return []
    
    def get_current_model(self) -> str:
        """Get the current model name."""
        return self.model
    
    def set_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        available = self.get_available_models()
        # Check if model is available (exact match or prefix match)
        for name in available:
            if name == model_name or name.startswith(model_name.split(":")[0]):
                self.model = model_name
                return True
        return False
    
    def generate(self, prompt: str, context: str = "", temperature: float = 0.7, 
                 max_tokens: int = 500) -> str:
        """Generate a response from the LLM."""
        
        # Build the full prompt with context
        if context:
            full_prompt = self._build_rag_prompt(prompt, context)
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build a RAG prompt with context and question."""
        prompt = f"""You are a friendly and helpful tourist guide for Częstochowa, Poland.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the Context below.
2. If the context contains relevant information, provide a helpful answer with specific details (names, addresses, ratings).
3. If the context contains RELATED information (e.g., user asks about "Jasna Góra" and context mentions "Wieża Jasnogórska" or related sites), use that information and explain the connection.
4. Be conversational and helpful. Recommend the best options based on ratings.
5. If you truly cannot find relevant information, politely say so and suggest what IS available.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_stream(self, prompt: str, context: str = "", temperature: float = 0.7,
                       max_tokens: int = 500):
        """Generate a streaming response (yields chunks)."""
        if context:
            full_prompt = self._build_rag_prompt(prompt, context)
        else:
            full_prompt = prompt
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                        
        except requests.RequestException as e:
            yield f"Error: {str(e)}"


# Global instance
llm = LLM()


if __name__ == "__main__":
    print("Testing LLM connection...")
    
    if not llm.is_available():
        print("\n⚠️  Ollama is not running or the model is not available.")
        print("Please start Ollama and pull the model:")
        print(f"  1. Start Ollama: ollama serve")
        print(f"  2. Pull model: ollama pull {OLLAMA_MODEL}")
    else:
        print("✅ LLM is available!")
        
        # Test generation
        print("\nTesting generation...")
        response = llm.generate(
            "What is Częstochowa famous for?",
            context="Częstochowa is a city in Poland, famous for Jasna Góra Monastery which houses the Black Madonna painting."
        )
        print(f"Response: {response}")
