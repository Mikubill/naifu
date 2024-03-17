class PromptStyleBase:
    """
    Base class for different styles of prompts.

    Attributes:
        input_field (str): The field to use for input.
        output_field (str): The field to use for output.
    """
    def __init__(self, prompt_field: str, output_field: str, **kwargs) -> None:
        "Initialize the PromptStyleBase."
        self.input_field = prompt_field
        self.output_field = output_field
    
    def apply(self, data: str):
        "Apply the style to the data. This method should be overridden by subclasses."
        raise NotImplementedError


class Phi2QAStyle(PromptStyleBase):
    """
    Class for the Phi2QA style of prompts.
    See also: https://huggingface.co/microsoft/phi-2

    QA Format:
    You can provide the prompt as a standalone question as follows:

        Instruct: Write a detailed analogy between mathematics and a lighthouse.
        Output: Mathematics is like a lighthouse. Just as a lighthouse guides ships safely to shore, mathematics provides a guiding light in the world of numbers and logic.

    where the model generates the text after "Output:".
    """
    def apply(self, data: str) -> tuple:
        inputs = f"""Instruct: {data[self.input_field]}\nOutput: """
        outputs = data[self.output_field]
        return inputs, outputs
    
    def build_instruct(self, prompt: str) -> str:
        "Build the instructions for the prompt."
        return f"Instruct: {prompt}\nOutput: "
    
class Phi2QAStyle2(PromptStyleBase):
    def apply(self, data: str) -> tuple:
        inputs = f"""Question: {data[self.input_field]}\nAnswer: """
        outputs = data[self.output_field]
        return inputs, outputs
    
    def build_instruct(self, prompt: str) -> str:
        "Build the instructions for the prompt."
        return f"Question: {prompt}\nAnswer: "