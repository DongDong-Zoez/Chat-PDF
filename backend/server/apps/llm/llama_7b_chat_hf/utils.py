import torch
import transformers
from typing import Optional, Union
from transformers.pipelines import Pipeline
from transformers import TextStreamer

def load_model(base_model: str, quantize: Optional[bool] = False) -> tuple:
    """
    Load a pretrained model, optionally with quantization.

    Args:
        base_model (str): The path or identifier of the base model.
        quantize (bool, optional): Whether to apply quantization. Defaults to False.

    Returns:
        tuple: A tuple containing the loaded model, tokenizer, and configuration.
    """
    if quantize:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    model_config = transformers.AutoConfig.from_pretrained(base_model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    return model, tokenizer

def build_pipeline(
    task: str,
    model: str,
    tokenizer: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: Union[str, dict] = "auto",
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 4,
    repetition_penalty: float = 1.1,
    return_full_text: bool = True,
    max_new_tokens: int = 1000,
) -> Pipeline:
    """
    Builds and returns a high-level pipeline for various tasks using Hugging Face's Transformers library.

    Parameters:
    - task (str): The task for which the pipeline will be built (e.g., "text-generation", "question-answering").
    - model (str): The name or path of the pre-trained model to be used.
    - tokenizer (str): The name or path of the pre-trained tokenizer corresponding to the model.
    - torch_dtype (torch.dtype, optional): The torch data type to be used (default is torch.float16).
    - device_map (str or dict, optional): Specification for device placement (default is "auto").
    - temperature (float, optional): The temperature parameter for sampling methods (default is 0.1).
    - top_p (float, optional): The top-p parameter for filtering tokens during sampling (default is 0.75).
    - top_k (int, optional): The top-k parameter for filtering tokens during sampling (default is 40).
    - num_beams (int, optional): The number of beams for beam search (default is 4).
    - repetition_penalty (float, optional): The repetition penalty parameter (default is 1.1).
    - return_full_text (bool, optional): Whether to return full text or a structured output (default is True).
    - max_new_tokens (int, optional): The maximum number of new tokens to be generated (default is 1000).

    Returns:
    - Pipeline: A high-level pipeline object for the specified task using Hugging Face's Transformers library.
    """
    hf_pipeline = transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map=device_map,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        return_full_text=return_full_text,
        max_new_tokens=max_new_tokens,
    )

    return hf_pipeline