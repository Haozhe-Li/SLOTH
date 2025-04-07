import os
import json
import time
from openai import OpenAI
import tiktoken
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


class LLMEvaluator:
    def __init__(
        self, model_id, api_key=None, api_base=None, temperature=0.8, max_tokens=512, timesleep=5
    ):
        """
        Initialize the LLM evaluator with model settings and API configuration.

        Args:
            model_id (str): The model identifier to use for evaluation
            api_key (str, optional): OpenAI API key. Defaults to environment variable.
            api_base (str, optional): OpenAI API base URL. Defaults to environment variable.
            temperature (float, optional): Sampling temperature. Defaults to 0.8.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 512.
        """
        print(model_id, api_key, api_base, temperature, max_tokens)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timesleep = timesleep

        if "groq" in api_base:
            self.provider = "groq"
        elif "fireworks" in api_base:
            self.provider = "fireworks"
        if "openai" in api_base:
            self.provider = "openai"
        else:
            self.provider = "custom"

        # Initialize OpenAI client
        self.client = OpenAI(base_url=api_base, api_key=api_key)

        # Create a model-specific results directory
        self.model_dir_name = self.model_id.replace("-", "_")
        self.results_dir = os.path.join(
            "./results", f"{self.provider}_{self.model_dir_name}"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def count_tokens(self, text):
        """Count the number of tokens in a text string."""
        try:
            # Use a default tokenizer (cl100k_base) for models not directly supported by tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: {e}")
            # Fallback to approximate token count (cast to int)
            return int(len(text.split()) * 1.3)

    def generate_completion(self, problem):
        """
        Generate code completion for a single problem using OpenAI API.

        Args:
            problem (dict): Problem data containing the prompt

        Returns:
            tuple: (completion_text, metrics_dict)
        """
        prompt = problem["prompt"]
        system_msg = "You are an expert Python programmer. Write code to solve the given problem."

        # Count input tokens
        total_input_tokens = self.count_tokens(system_msg + prompt)

        metrics = {
            "input_tokens": total_input_tokens,
            "output_tokens": 0,  # Initialize output tokens
            "response_time": 0,
            "time_to_first_token": 0,
        }

        try:
            start_time = time.time()

            # Set stream=True to measure time to first token
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=1,
                stream=True,
                stop=None,
            )

            # Initialize variables for streaming
            completion_content = ""
            first_token_received = False

            # Process the streaming response
            for chunk in response:
                if not first_token_received and chunk.choices[0].delta.content:
                    first_token_received = True
                    metrics["time_to_first_token"] = time.time() - start_time

                if chunk.choices[0].delta.content:
                    completion_content += chunk.choices[0].delta.content

            # Calculate total response time
            metrics["response_time"] = time.time() - start_time

            # Count output tokens
            metrics["output_tokens"] = self.count_tokens(completion_content)

            # Extract only the code part if it contains markdown code blocks
            if "```python" in completion_content and "```" in completion_content:
                code_block = (
                    completion_content.split("```python")[1].split("```")[0].strip()
                )
                return code_block, metrics

            return completion_content, metrics

        except Exception as e:
            print(f"Error generating completion for problem: {e}")
            return "", metrics

    def evaluate(self, num_samples=None):
        """
        Evaluate the model on HumanEval dataset.

        Args:
            num_samples (int, optional): Number of problems to evaluate. Defaults to None (all problems).

        Returns:
            tuple: (evaluation_data, results_directory)
        """
        problems = read_problems()

        # For testing, you can limit to fewer problems
        if num_samples:
            selected_problems = list(problems.items())[:num_samples]
        else:
            selected_problems = list(problems.items())

        # Generate completions
        completions = []
        all_metrics = []

        # For collecting pairs
        token_response_time_pairs = []
        token_ttft_pairs = []

        for problem_id, problem in selected_problems:
            print(f"Generating completion for {problem_id}...")
            completion, metrics = self.generate_completion(problem)

            # Format as expected by the evaluation script
            completions.append({"task_id": problem_id, "completion": completion})

            # Store metrics along with problem ID
            metrics["problem_id"] = problem_id
            all_metrics.append(metrics)

            # Record the pairs
            token_response_time_pairs.append(
                {
                    "input_tokens": metrics["input_tokens"],
                    "response_time": metrics["response_time"],
                    "problem_id": problem_id,
                }
            )

            token_ttft_pairs.append(
                {
                    "input_tokens": metrics["input_tokens"],
                    "time_to_first_token": metrics["time_to_first_token"],
                    "problem_id": problem_id,
                }
            )

            # Add a small delay to avoid rate limiting
            time.sleep(self.timesleep)

        # Save completions to a file in the model-specific directory
        output_file = os.path.join(self.results_dir, "completions.jsonl")
        write_jsonl(output_file, completions)

        # Evaluate the completions
        try:
            results = evaluate_functional_correctness(output_file)
            print(f"\nResults for {self.model_id}:")
            print(f"Pass@1: {results['pass@1']:.4f}")
        except AssertionError as e:
            print(f"Evaluation error: {e}")
            print("Note: This error may occur if not all problems were completed.")
            # Create a basic results structure if evaluation fails
            results = {"pass@1": 0.0, "evaluation_error": str(e)}

        # Combine results and metrics
        evaluation_data = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "evaluation_results": results,
            "problem_metrics": all_metrics,
            "average_metrics": {
                "avg_input_tokens": sum(m["input_tokens"] for m in all_metrics)
                / len(all_metrics),
                "avg_output_tokens": sum(m["output_tokens"] for m in all_metrics)
                / len(all_metrics),
                "avg_response_time": sum(m["response_time"] for m in all_metrics)
                / len(all_metrics),
                "avg_time_to_first_token": sum(
                    m["time_to_first_token"] for m in all_metrics
                )
                / len(all_metrics),
            },
            "token_response_time_pairs": token_response_time_pairs,
            "token_ttft_pairs": token_ttft_pairs,
        }

        # Save detailed evaluation data
        self.save_evaluation_data(evaluation_data)

        return evaluation_data, self.results_dir

    def save_evaluation_data(self, evaluation_data):
        """Save evaluation data to a JSON file."""
        evaluation_file = os.path.join(self.results_dir, "evaluation.json")
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_data, f, indent=2)
        print(f"Detailed evaluation data saved to {evaluation_file}")


if __name__ == "__main__":
    # Example usage
    # Make sure you have set the OPENAI_API_KEY environment variable
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Please set the OPENAI_API_KEY environment variable")
    #     exit(1)

    # Initialize the evaluator with model settings
    evaluator = LLMEvaluator(
        model_id="qwen-2.5-32b",  # Model to evaluate
        temperature=1,
        max_tokens=1024,
        api_key="your-api-key",
        api_base="base",
    )

    # Run evaluation
    evaluation_data, results_dir = evaluator.evaluate()
