# LLM Evaluation Index

A framework for evaluating and benchmarking Large Language Models (LLMs) across various dimensions and tasks.

## 📋 Overview

LLM Evaluation Index provides tools and metrics to assess the performance of different language models on standardized tasks. This project helps researchers and developers compare models systematically and make informed decisions about which models best suit their needs.

## ✨ Features

- Comprehensive evaluation across multiple dimensions (reasoning, knowledge, safety, etc.)
- Support for popular LLM frameworks and models
- Standardized benchmarking datasets
- Performance visualization and comparison tools
- Customizable evaluation metrics

## 🚀 Installation

Clone the repo and

```bash
# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

### Basic Evaluation

```python
from llm_eval_index import Evaluator
from llm_eval_index.models import GPT4, LLaMA

# Initialize models
gpt4 = GPT4(api_key="your-api-key")
llama = LLaMA(model_path="path/to/model")

# Create evaluator
evaluator = Evaluator()

# Run benchmark
results = evaluator.evaluate([gpt4, llama], 
                            tasks=["reasoning", "knowledge", "coding"],
                            verbose=True)

# View results
results.summary()
results.plot_comparison()
```

### Custom Evaluation Tasks

```python
from llm_eval_index import Task, Evaluator

# Define custom task
my_task = Task(
    name="my_custom_task",
    prompt_template="Solve the following problem: {problem}",
    evaluation_metric="accuracy",
    dataset_path="path/to/dataset"
)

# Run evaluation with custom task
evaluator = Evaluator()
results = evaluator.evaluate([model1, model2], tasks=[my_task])
```

## 📊 Benchmark Results

| Model | Reasoning | Knowledge | Coding | Safety | Average |
|-------|-----------|-----------|--------|--------|---------|
| GPT-4 | 92.3      | 89.7      | 95.1   | 88.2   | 91.3    |
| LLaMA | 85.6      | 82.3      | 79.8   | 84.5   | 83.0    |
| ...   | ...       | ...       | ...    | ...    | ...     |

## 📚 Documentation

For full documentation, visit [our docs page](https://github.com/yourusername/llm_eval_index/docs).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Your Name - [your-email@example.com](mailto:your-email@example.com)
- Project Link: [https://github.com/yourusername/llm_eval_index](https://github.com/yourusername/llm_eval_index)
