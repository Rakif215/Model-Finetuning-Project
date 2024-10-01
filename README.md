# LLama Fine-tuning Project for Social Media Analytics

This project focuses on fine-tuning LLama models for social media analytics tasks using the Unsloth library. It provides a flexible framework for initializing, training, and deploying fine-tuned language models with a specific emphasis on social media content analysis.

## Features

- Fine-tune LLama models (including LLama 3) using Parameter-Efficient Fine-Tuning (PEFT) techniques
- Support for multiple model variants and quantization methods
- Custom dataset preparation and loading
- Configurable training process with JSON-based configurations
- Inference capabilities with support for few-shot learning
- Easy model saving and deployment options, including Hugging Face Hub integration

## Installation

1. Clone this repository:
   ```
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the `install_packages.py` script to install additional dependencies:
   ```
   python install_packages.py
   ```

## Usage

### Fine-tuning

To fine-tune a model:

1. Prepare your dataset and place it in the `data` directory.
2. Adjust the configurations in `config/lora_config.json` and `config/training_config.json` as needed.
3. Run the fine-tuning process:
   ```
   python finetuning.py --dataset_path [path_to_your_dataset]
   ```

To resume training from a checkpoint, use the `--resume` flag:
```
python finetuning.py --dataset_path [path_to_your_dataset] --resume
```

### Inference

You can use the trained model for inference by initializing the `LLamafine` class and using its `inference` or `inference_n_shots` methods.

Example:
```python
llamafine = LLamafine()
llamafine.initialize_model(model_name="your_fine_tuned_model_name")
output, _ = llamafine.inference(sys_prompt, instruction, input_text)
```

## Project Structure

- `finetuning.py`: Main script for model fine-tuning and management
- `install_packages.py`: Script to install necessary packages and dependencies
- `requirements.txt`: List of Python package dependencies
- `config/`: Directory containing JSON configuration files for LoRA and training
- `data/`: Directory to store datasets (not included in the repository)
- `checkpoints/`: Directory to store model checkpoints during training

## Configuration

- `lora_config.json`: Configure LoRA (Low-Rank Adaptation) settings
- `training_config.json`: Set training arguments such as learning rate, batch size, etc.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- This project uses the Unsloth library for efficient fine-tuning of LLama models.
- Special thanks to the Hugging Face team for their transformers library and model hosting capabilities.
