# code base on Github https://github.com/Infini-AI-Lab/gsm_infinite/

import os
import re
import math
from typing import List, Tuple, Optional, Any, Dict, Union
import torch
import json


class ModelHandler:
    SUPPORTED_BACKENDS = ["aws","AI71"]

    def __init__(self, model_name: str, backend_type: str = "AI71" , device_map:str = "auto"):
        if backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        self.model_type = backend_type
        self.model_name = model_name
        self.client = None
        self.device_map = device_map
        self._initialize_client()

    def _initialize_client(self):
        
        if self.model_type == "aws":
            import boto3
            from sagemaker import Session, Predictor
            from sagemaker.serializers import JSONSerializer
            from transformers import AutoTokenizer
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Ollama backend requires GPU support")
            
            # Create boto3 session using a specific profile
            boto_sess = boto3.Session(profile_name="personal")

            # Create a SageMaker session from the boto3 session
            sagemaker_session = Session(boto_session=boto_sess)
            self.predictor = Predictor(
                endpoint_name=self.model_name,  # endpoint_name
                sagemaker_session=sagemaker_session,
                serializer=JSONSerializer(),
            )
            model_name_hf = "tiiuae/Falcon3-10B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_hf , cache_dir= "/data_hdd/damian/llm")
        

        elif self.model_type == "AI71":
            import openai
            AI71_BASE_URL = "https://api.ai71.ai/v1/"
            AI71_API_KEY = "###Use your API########"
            self.client = openai.OpenAI(
                api_key=AI71_API_KEY,
                base_url=AI71_BASE_URL,
    )



    # @retry_with_exponential_backoff
    def generate_answer(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, Dict[str, Any]]:

        if self.model_type == "aws":
            return self._get_aws_response(prompt, **kwargs)
        elif self.model_type == "AI71":
            return self._get_AI71_response(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported backend type: {self.model_type}")

  
    def _get_aws_response(
        self, prompt: Union[str, List[str]], max_tokens: int = 4096, temperature=None, debug: bool = False, **kwargs
    ) -> str:
        messages = (
            [
                {"role": "user", "content": prompt}
            ]
            if type(prompt) == str
            else prompt
        )
        message_formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        #message_formatted = self.formatted_prompt(messages)
        payload = {"inputs": message_formatted, "parameters": kwargs  }
        # Make the prediction
        response = self.predictor.predict(payload)
        parsed_response = json.loads(response.decode("utf-8"))
        return parsed_response['generated_text']

    def _get_AI71_response(
        self, messages: Union[str, List[str]], max_tokens: int = 4096, debug: bool = False, **kwargs
    ) -> str:
    # Rename user-friendly keys to API-expected keys
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
        if "do_sample" in kwargs:
            do_sample = kwargs.pop("do_sample")
            if do_sample is False:
                kwargs["temperature"] = 0.0      
        
        response = self.client.chat.completions.create(
        model = self.model_name,
        messages = messages,
        **kwargs
        #top_p = 0.8
        )
        return response.choices[0].message.content
            

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "client") and self.client:
            if hasattr(self.client, "close"):
                self.client.close()
            self.client = None


def main():
    prompt = "What is the capital of France? Please describe it."

    # try:
    handler = ModelHandler(model_name="tiiuae/falcon3-10b-instruct", backend_type="AI71")
    ret = handler.generate_answer(
        prompt,
        max_tokens=4000,
        temperature=0.7,
    )
    print(f" Response: {ret}")

if __name__ == "__main__":
    main()
