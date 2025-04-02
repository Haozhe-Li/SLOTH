import os
import json
import time
from openai import OpenAI
import tiktoken
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


client = OpenAI(
    base_url=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a text string."""
    try:
        # Use a default tokenizer (cl100k_base) for models not directly supported by tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Warning: {e}")
        # Fallback to approximate token count (cast to int)
        return int(len(text.split()) * 1.3)


def generate_completions(
    problem, model="llama-3.1-8b-instant", temperature=0.8, max_tokens=512
):
    """Generate code completion using OpenAI API."""
    prompt = problem["prompt"]
    system_msg = (
        "You are an expert Python programmer. Write code to solve the given problem."
    )

    # Count input tokens
    total_input_tokens = count_tokens(system_msg + prompt, model)

    metrics = {
        "input_tokens": total_input_tokens,
        "output_tokens": 0,  # Initialize output tokens
        "response_time": 0,
        "time_to_first_token": 0,
    }

    try:
        start_time = time.time()

        # Set stream=True to measure time to first token
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
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
        metrics["output_tokens"] = count_tokens(completion_content, model)

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


def evaluate_model(
    model_name="llama-3.1-8b-instant", temperature=0.8, max_tokens=512, num_samples=None
):
    """Evaluate a model on HumanEval dataset."""
    problems = read_problems()

    # For testing, you can limit to fewer problems
    if num_samples:
        selected_problems = list(problems.items())[:num_samples]
    else:
        selected_problems = list(
            problems.items()
        )  # Use all problems if num_samples not specified

    # Generate completions
    completions = []
    all_metrics = []

    # For collecting pairs
    token_response_time_pairs = []
    token_ttft_pairs = []

    for problem_id, problem in selected_problems:
        print(f"Generating completion for {problem_id}...")
        completion, metrics = generate_completions(
            problem, model=model_name, temperature=temperature, max_tokens=max_tokens
        )

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
        time.sleep(1)

    # Create model-specific results directory
    model_dir_name = model_name.replace("-", "_")
    results_dir = os.path.join("./results", model_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save completions to a file in the model-specific directory
    output_file = os.path.join(results_dir, f"completions.jsonl")
    write_jsonl(output_file, completions)

    # Evaluate the completions
    try:
        results = evaluate_functional_correctness(output_file)
        print(f"\nResults for {model_name}:")
        print(f"Pass@1: {results['pass@1']:.4f}")
    except AssertionError as e:
        print(f"Evaluation error: {e}")
        print("Note: This error may occur if not all problems were completed.")
        # Create a basic results structure if evaluation fails
        results = {"pass@1": 0.0, "evaluation_error": str(e)}

    # Combine results and metrics
    evaluation_data = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
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
        # Add the requested pairs
        "token_response_time_pairs": token_response_time_pairs,
        "token_ttft_pairs": token_ttft_pairs,
    }

    return evaluation_data, results_dir


if __name__ == "__main__":
    # Make sure you have set the OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)

    # Evaluate models
    model_name = "llama-3.3-70b-versatile"  # Default model
    evaluation_data, results_dir = evaluate_model(
        model_name=model_name,
        temperature=1,
        max_tokens=1024,
        # num_samples=164,  # Adjust this for more or fewer problems
    )

    # Save detailed evaluation data including metrics to the model-specific directory
    evaluation_file = os.path.join(results_dir, "evaluation.json")
    with open(evaluation_file, "w") as f:
        json.dump(evaluation_data, f, indent=2)

    print(f"Detailed evaluation data saved to {evaluation_file}")
