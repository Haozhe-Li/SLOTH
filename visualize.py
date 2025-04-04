import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pandas as pd
from glob import glob

# Set the style for the plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def load_evaluation_data():
    """Load all evaluation.json files from results directory"""
    results_dir = './results'
    model_results = {}
    
    # Get all subdirectories in results
    model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
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
    plt.savefig('./results/avg_response_time_comparison.png')
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
    plt.savefig('./results/pass_rate_comparison.png')
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
    plt.savefig('./results/ttft_comparison.png')
    plt.close()

def build_regression_models(model_results):
    """Build regression models for token length vs time metrics"""
    # Prepare data for combined regression plots
    all_models_data = {}
    
    for model_name, data in model_results.items():
        problem_metrics = data['problem_metrics']
        
        # Skip if necessary metrics aren't available
        if not problem_metrics or not all(key in problem_metrics[0] for key in 
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
        plt.savefig(f'./results/{model_name}_regression.png')
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
        plt.savefig('./results/combined_ttft_regression.png')
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
        plt.savefig('./results/combined_response_time_regression.png')
        plt.close()

def main():
    # Load all evaluation data
    model_results = load_evaluation_data()
    
    if not model_results:
        print("No evaluation.json files found in results directory!")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    # Plot model metrics comparison
    plot_model_metrics(model_results)
    
    # Build regression models for each model
    build_regression_models(model_results)
    
    print(f"Visualization complete! Generated plots for {len(model_results)} models.")
 
if __name__ == "__main__":
    main()