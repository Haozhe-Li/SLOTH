import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd
import gradio as gr
from glob import glob
import base64
from io import BytesIO
import tempfile
import shutil

# Set the style for the plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def get_available_models():
    """Get all model names from results directory"""
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        return []
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    return model_dirs

def load_evaluation_data(selected_models=None):
    """Load evaluation.json files from results directory for selected models"""
    results_dir = './results'
    model_results = {}
    
    # If no models are selected, get all available models
    if selected_models is None or len(selected_models) == 0:
        model_dirs = get_available_models()
    else:
        model_dirs = selected_models
    
    for model_dir in model_dirs:
        eval_path = os.path.join(results_dir, model_dir, 'evaluation.json')
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                data = json.load(f)
                model_results[model_dir] = data
    
    return model_results

def plot_model_metrics(model_results):
    """Plot average time, pass rate, and time to first token for each model"""
    models = list(model_results.keys())
    avg_times = [data['average_metrics']['avg_response_time'] for data in model_results.values()]
    pass_rates = [data['evaluation_results']['pass@1'] for data in model_results.values()]
    ttft = [data['average_metrics']['avg_time_to_first_token'] for data in model_results.values()]
    
    # Create a static directory for storing images
    static_dir = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Clear previous images
    for file in os.listdir(static_dir):
        if file.endswith('.png'):
            os.remove(os.path.join(static_dir, file))
    
    image_paths = {}
    
    # Plot 1: Average Response Time
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, avg_times, color='skyblue')
    plt.title('Average Response Time by Model', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact values above each bar
    for bar, value in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02, 
                f'{value:.2f}s', 
                ha='center', va='bottom', fontsize=12)
    
    # Add pass rates as annotations
    for i, (model, pass_rate) in enumerate(zip(models, pass_rates)):
        plt.text(i, avg_times[i]/2, 
                f'Pass@1: {pass_rate:.2f}', 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='navy', alpha=0.6))
    
    plt.tight_layout()
    avg_time_path = os.path.join(static_dir, 'avg_response_time_comparison.png')
    plt.savefig(avg_time_path)
    image_paths['avg_response_time'] = avg_time_path
    plt.close()
    
    # Plot 2: Pass@1 Rate
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, pass_rates, color='lightgreen')
    plt.title('Pass@1 Rate by Model', fontsize=16)
    plt.ylabel('Pass Rate', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact values above each bar
    for bar, value in zip(bars, pass_rates):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02, 
                f'{value:.3f}', 
                ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    pass_rate_path = os.path.join(static_dir, 'pass_rate_comparison.png')
    plt.savefig(pass_rate_path)
    image_paths['pass_rate'] = pass_rate_path
    plt.close()
    
    # Plot 3: Time to First Token
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, ttft, color='salmon')
    plt.title('Average Time to First Token by Model', fontsize=16)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add exact values above each bar
    for bar, value in zip(bars, ttft):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.005, 
                f'{value:.3f}s', 
                ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    ttft_path = os.path.join(static_dir, 'ttft_comparison.png')
    plt.savefig(ttft_path)
    image_paths['ttft'] = ttft_path
    plt.close()
    
    return image_paths

def build_regression_models(model_results):
    """Build regression models for token length vs time metrics"""
    # Prepare data for combined regression plots
    all_models_data = {}
    image_paths = {}
    
    # Create a static directory for storing images
    static_dir = os.path.join(os.getcwd(), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    for model_name, data in model_results.items():
        if 'problem_metrics' not in data:
            print(f"Skipping {model_name} due to missing 'problem_metrics' in the data")
            continue
            
        problem_metrics = data['problem_metrics']
        
        # Skip if no problem metrics or if necessary metrics aren't available
        if not problem_metrics or len(problem_metrics) == 0:
            print(f"Skipping {model_name} due to empty problem_metrics")
            continue
            
        # Check if the required keys exist in the first problem metric
        if not all(key in problem_metrics[0] for key in 
                  ['input_tokens', 'output_tokens', 
                   'time_to_first_token', 'response_time']):
            print(f"Skipping {model_name} due to missing metrics in the data")
            continue
        
        # Extract data for regression
        input_tokens = np.array([metric['input_tokens'] for metric in problem_metrics])
        output_tokens = np.array([metric['output_tokens'] for metric in problem_metrics])
        total_tokens = input_tokens + output_tokens
        ttft = np.array([metric['time_to_first_token'] for metric in problem_metrics])
        response_time = np.array([metric['response_time'] for metric in problem_metrics])
        
        # Store data for combined plots
        all_models_data[model_name] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'ttft': ttft,
            'response_time': response_time
        }
        
        # Create dataframe for easier plotting
        df = pd.DataFrame({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'ttft': ttft,
            'response_time': response_time
        })
        
        # Regression 1: Input tokens vs time to first token
        X1 = input_tokens.reshape(-1, 1)
        y1 = ttft
        reg1 = LinearRegression().fit(X1, y1)
        score1 = reg1.score(X1, y1)
        
        # Regression 2: Total tokens vs response time
        X2 = total_tokens.reshape(-1, 1)
        y2 = response_time
        reg2 = LinearRegression().fit(X2, y2)
        score2 = reg2.score(X2, y2)
        
        # Create individual model plots
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Input tokens vs TTFT
        axs[0].scatter(input_tokens, ttft, alpha=0.5, color='blue')
        axs[0].plot(input_tokens, reg1.predict(X1), color='red', linewidth=2)
        axs[0].set_title(f'Input Tokens vs Time to First Token\nR² = {score1:.3f}')
        axs[0].set_xlabel('Input Tokens')
        axs[0].set_ylabel('Time to First Token (s)')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Total tokens vs Response time
        axs[1].scatter(total_tokens, response_time, alpha=0.5, color='green')
        axs[1].plot(total_tokens, reg2.predict(X2), color='red', linewidth=2)
        axs[1].set_title(f'Total Tokens vs Response Time\nR² = {score2:.3f}')
        axs[1].set_xlabel('Total Tokens (Input + Output)')
        axs[1].set_ylabel('Response Time (s)')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f'Regression Models for {model_name}', fontsize=16)
        plt.tight_layout()
        
        model_regression_path = os.path.join(static_dir, f'{model_name}_regression.png')
        plt.savefig(model_regression_path)
        image_paths[f'{model_name}_regression'] = model_regression_path
        plt.close()
    
    # Create combined plots for comparison if we have multiple models
    if len(all_models_data) > 1:
        # Combined plot 1: Input tokens vs TTFT
        plt.figure(figsize=(12, 8))
        
        # Use a distinct color palette for different models
        colors = plt.cm.tab10.colors
        
        for i, (model_name, data) in enumerate(all_models_data.items()):
            color = colors[i % len(colors)]
            
            # Get data
            input_tokens = data['input_tokens']
            ttft = data['ttft']
            
            # Regression
            X = input_tokens.reshape(-1, 1)
            y = ttft
            reg = LinearRegression().fit(X, y)
            score = reg.score(X, y)
            
            # Scatter and regression line
            plt.scatter(input_tokens, ttft, alpha=0.3, color=color, label=f"{model_name} data")
            
            # Create smooth line for prediction
            x_sorted = np.sort(input_tokens)
            plt.plot(x_sorted, reg.predict(x_sorted.reshape(-1, 1)), color=color, 
                     linewidth=2, label=f"{model_name} (R² = {score:.3f})")
        
        plt.title('Input Tokens vs Time to First Token (All Models)', fontsize=16)
        plt.xlabel('Input Tokens', fontsize=14)
        plt.ylabel('Time to First Token (s)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        
        combined_ttft_path = os.path.join(static_dir, 'combined_ttft_regression.png')
        plt.savefig(combined_ttft_path)
        image_paths['combined_ttft_regression'] = combined_ttft_path
        plt.close()
        
        # Combined plot 2: Total tokens vs Response time
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, data) in enumerate(all_models_data.items()):
            color = colors[i % len(colors)]
            
            # Get data
            total_tokens = data['total_tokens']
            response_time = data['response_time']
            
            # Regression
            X = total_tokens.reshape(-1, 1)
            y = response_time
            reg = LinearRegression().fit(X, y)
            score = reg.score(X, y)
            
            # Scatter and regression line
            plt.scatter(total_tokens, response_time, alpha=0.3, color=color, label=f"{model_name} data")
            
            # Create smooth line for prediction
            x_sorted = np.sort(total_tokens)
            plt.plot(x_sorted, reg.predict(x_sorted.reshape(-1, 1)), color=color, 
                     linewidth=2, label=f"{model_name} (R² = {score:.3f})")
        
        plt.title('Total Tokens vs Response Time (All Models)', fontsize=16)
        plt.xlabel('Total Tokens (Input + Output)', fontsize=14)
        plt.ylabel('Response Time (s)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        
        combined_response_path = os.path.join(static_dir, 'combined_response_time_regression.png')
        plt.savefig(combined_response_path)
        image_paths['combined_response_time_regression'] = combined_response_path
        plt.close()
    
    return image_paths

def visualize_models(selected_models):
    """Generate visualizations for selected models and return markdown content with images"""
    if not selected_models:
        return "Please select at least one model to visualize."
    
    # Load data for selected models
    model_results = load_evaluation_data(selected_models)
    
    if not model_results:
        return "No evaluation data found for the selected models."
    
    try:
        # Generate plots
        metric_images = plot_model_metrics(model_results)
        regression_images = build_regression_models(model_results)
        
        # Get server URL for image paths
        static_dir = os.path.join(os.getcwd(), 'static')
        
        # Create image gallery for Gradio display
        image_gallery = []
        if 'avg_response_time' in metric_images:
            image_gallery.append(metric_images['avg_response_time'])
        if 'pass_rate' in metric_images:
            image_gallery.append(metric_images['pass_rate'])
        if 'ttft' in metric_images:
            image_gallery.append(metric_images['ttft'])
        
        # Add regression images to gallery
        if 'combined_ttft_regression' in regression_images:
            image_gallery.append(regression_images['combined_ttft_regression'])
        if 'combined_response_time_regression' in regression_images:
            image_gallery.append(regression_images['combined_response_time_regression'])
        
        # Add individual model regression plots
        for model_name in selected_models:
            regression_key = f'{model_name}_regression'
            if regression_key in regression_images:
                image_gallery.append(regression_images[regression_key])
        
        # Create markdown content with descriptions
        markdown_content = f"""
## Comparing Results for {', '.join(selected_models)}

### Performance Metrics

- **Average response time**: How long it takes each model to generate complete responses
- **Pass@1 rate**: Percentage of problems the model can solve correctly on the first attempt
- **Time to first token (TTFT)**: How quickly the model starts generating output
- **Regression analysis**: How response time and TTFT correlate with tokens

Images are displayed in the gallery below. You can click on them to view in full size.
"""
        
        return markdown_content, image_gallery
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return f"Error generating visualizations: {str(e)}", []

def launch_gradio_interface():
    """Launch Gradio interface for model visualization"""
    # Get available models
    available_models = get_available_models()
    
    # Create static directory for images
    os.makedirs('static', exist_ok=True)
    
    # Define interface
    with gr.Blocks(title="LLM Evaluation Visualizer", css="#gallery { min-height: 400px; }") as demo:
        gr.Markdown("# LLM Evaluation Visualizer")
        gr.Markdown("Select models to compare and visualize results")
        
        with gr.Row():
            if available_models:
                model_selector = gr.CheckboxGroup(
                    choices=available_models,
                    label="Select Models to Compare",
                    info="Choose at least one model",
                    value=[available_models[0]] if available_models else None
                )
            else:
                model_selector = gr.CheckboxGroup(
                    choices=[],
                    label="Select Models to Compare",
                    info="No models found in results directory"
                )
            visualize_button = gr.Button("Generate Visualizations")
        
        output_markdown = gr.Markdown("Select models and click 'Generate Visualizations' to start")
        
        # Image gallery for displaying the visualizations
        gallery = gr.Gallery(
            label="Visualization Results",
            show_label=True,
            elem_id="gallery",
            columns=2,
            rows=3,
            height="auto",
            object_fit="contain"
        )
        
        visualize_button.click(
            fn=visualize_models,
            inputs=[model_selector],
            outputs=[output_markdown, gallery]
        )
    
    demo.launch(share=False)

def main():
    # Check if results directory exists
    if not os.path.exists('./results'):
        print("Results directory not found. Creating empty directory.")
        os.makedirs('./results', exist_ok=True)
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("No model results found in the results directory!")
        print("Please add evaluation data to the results directory.")
    else:
        print(f"Found {len(available_models)} models: {', '.join(available_models)}")
    
    # Launch Gradio interface
    launch_gradio_interface()

if __name__ == "__main__":
    main()