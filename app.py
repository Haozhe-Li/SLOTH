import gradio as gr
from eval import LLMEvaluator

def process_api_info(api_key, api_base_url, model_id):
    evaluator = LLMEvaluator(
        model_id=model_id,
        temperature=1,
        max_tokens=1024,
        api_key=api_key,
        api_base=api_base_url,
    )
    evaluation_data, results_dir = evaluator.evaluate()
    return f"Evaluation completed. Results saved. Please use visualizer to view the results."

def update_api_base(provider):
    if provider == "OpenAI":
        return "https://api.openai.com/v1"
    elif provider == "Groq":
        return "https://api.groq.com/openai/v1"
    elif provider == "Fireworks AI":
        return "https://api.fireworks.ai/inference/v1"
    else:  # "Custom"
        return ""

with gr.Blocks() as demo:
    gr.Markdown("# API Configuration Interface")
    
    # Provider selection
    provider = gr.Radio(
        ["OpenAI", "Groq", "Fireworks AI", "Custom"], 
        label="Provider", 
        value="OpenAI"
    )
    
    # Input components
    api_key = gr.Textbox(label="API Key", placeholder="Enter your API key", type="password")
    api_base_url = gr.Textbox(label="API Base URL", placeholder="Enter the base URL for the API. Leave blank for OpenAI")
    model_id = gr.Textbox(label="Model ID", placeholder="Enter the model identifier")
    
    # Update API base URL when provider changes
    provider.change(
        fn=update_api_base,
        inputs=provider,
        outputs=api_base_url
    )
    
    # Submit button and output
    submit_btn = gr.Button("Submit")
    output = gr.Textbox(label="Status")
    
    # Define the action when submit is clicked
    submit_btn.click(
        fn=process_api_info,
        inputs=[api_key, api_base_url, model_id],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()