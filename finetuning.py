import os
from datasets import load_from_disk
import click
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

import json
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import os
import json
from datasets import Dataset
import re
import pandas as pd

class LLamafine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True
        self.model_name= None
        self.model_full_name = {"shortname": "long_name"}

        self.get_prompt = {"unsloth/llama-3-8b-Instruct-bnb-4bit": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}: {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""}

    def initialize_model(self, model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
        """
        Initialize the model and tokenizer with the given parameters.

        Args:
            model_name (str): The name of the model to be loaded.
            max_seq_length (int, optional): The maximum sequence length. Default is 4096.
            dtype (torch.dtype, optional): The data type to be used. Default is None.
            load_in_4bit (bool, optional): Whether to use 4bit quantization. Default is True.

        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,

        )



    def load_datasets(self,file_path):
        # Define the data directory
        data_dir = file_path

        # List all items in the data directory
        items = os.listdir(data_dir)

        # Filter out only files from the listed items
        datasets = [item for item in items if os.path.isfile(os.path.join(data_dir, item))]

        # Initialize an empty list to store the DataFrames
        dataframes = []

        # Loop through each dataset filename
        for filename in datasets:
            data_list = []
            try:
                # Open the JSONL file and read its content
                with open(os.path.join(data_dir, filename), 'r') as file:
                    for line in file:
                        # Parse each line as a JSON object and append to data_list
                        data = json.loads(line)
                        data_list.append(data)
                # Convert the list of dictionaries to a pandas DataFrame
                df = pd.DataFrame(data_list)
                # Append the DataFrame to the list of DataFrames
                dataframes.append(df)
            except FileNotFoundError:
                print(f"File {filename} not found.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {filename}: {e}")

        # Print the first few rows of each DataFrame

        # Set your random seed
        random_seed = 42

        # Shuffle the dataframes
        shuffled_dfs = [df.sample(frac=1, random_state=random_seed).reset_index(drop=True) for df in dataframes]

        if shuffled_dfs:
            concatenated_dataframe = pd.concat(shuffled_dfs, ignore_index=True, sort=False)
            sys_prompt = """أنت خبير في تحليل وسائل التواصل الاجتماعي. يمكنك فهم وتحليل المقالات الإخبارية والمنشورات على وسائل التواصل الاجتماعي. مهمتك هي إجراء فحوصات شاملة للمحتوى. تشمل مهامك على اكتشاف محتوى البالغين، تقييم الحقائق، تحديد الرسائل غير المرغوب فيها، تقييم الأهمية، اكتشاف الادعاءات، تقييم الأضرار، تحديد خطاب الكراهية، اكتشاف اللغة المسيئة، التعرف على الدعاية، وتقييم الموضوعية.
            """
            concatenated_dataframe["prompt"] = sys_prompt

            concatenated_dataframe = concatenated_dataframe[["prompt","instruction","input","output"]]

            self.concatenated_dataframe = concatenated_dataframe
            return self.concatenated_dataframe

    def prepare_dataset_llama3(self,dataset = None):
          if dataset is None:
              dataset = Dataset.from_pandas(self.concatenated_dataframe)

          prompt = self.get_prompt[self.model_name]
          EOS_TOKEN = self.tokenizer.eos_token

          def formatting_prompts_func(examples):
              sys_prompt = examples["prompt"]
              instructions = examples["instruction"]
              inputs = examples["input"]
              outputs = examples["output"]
              texts = []
              for sys_prompt, instruction, input, output in zip(sys_prompt, instructions, inputs, outputs):
                  text = prompt.format(sys_prompt, instruction, input, output) + EOS_TOKEN
                  texts.append(text)
              return {"text": texts}


          dataset = dataset.map(formatting_prompts_func, batched=True)

          return dataset


    def load_peft_config(self, config_file):
        """
        Load PEFT configuration from a JSON file.

        Args:
            config_file (str): Path to the JSON configuration file.

        Returns:
            config (dict): Loaded configuration.
        """
        with open(config_file, 'r') as file:
            config = json.load(file)
        return config

    def apply_peft_model(self, config_file):
        """
        Apply PEFT (Parameter-Efficient Fine-Tuning) to the model using a configuration file.

        Args:
            config_file (str): Path to the JSON configuration file.

        Returns:
            model: The model with PEFT applied.
        """
        config = self.load_peft_config(config_file)
        lora_config = config["lora_config"]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
            random_state=lora_config["random_state"],
            use_rslora=lora_config["use_rslora"],
            loftq_config=lora_config["loftq_config"],
        )
        return self.model



    def train_model(self, dataset, config_file,output_dir='checkpoints',resume_from_checkpoint=False):
        """
        Train the model using the provided dataset and training arguments from a configuration file.

        Args:
            dataset: The preprocessed dataset for training.
            config_file (str): Path to the JSON configuration file with training arguments.

        Returns:
            trainer: The trainer instance used for training.
        """
        with open(config_file, 'r') as file:
            config = json.load(file)
        training_args = TrainingArguments(**config["training_args"],output_dir=output_dir)

        trainer = SFTTrainer(

            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.

            args=training_args,
        )
        if resume_from_checkpoint and os.path.exists(output_dir):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        return trainer

    def save_model(self, save_path, hub_name, token):
        self.model.push_to_hub(hub_name, token=token) # Online saving
        self.model.save_pretrained(save_path)
        self.model.save_pretrained("lora_adapter", save_function="lora_adapter") # Saving the LoRA adapter


    def save_model_to_hub(self, save_method = "merged_16bit", user_name="", token=""):
        self.model.push_to_hub_merged(user_name, self.tokenizer, save_method =save_method, token = token)


    def generate_output(self, instruction, input_text):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

        inputs = self.tokenizer(
            [
                alpaca_prompt.format(
                    instruction,  # instruction
                    input_text,  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        outputs = self.model.generate(**inputs, max_new_tokens=128, use_cache=True)
        return f"answer: {self.tokenizer.batch_decode(outputs, skip_special_tokens=True)}"


#     def generate_and_add_predictions(self, dataset):
#         """
#         Generate outputs for each row in the dataset and add them as a new column.

#         Args:
#             dataset: The dataset containing the instructions and inputs.

#         Returns:
#             dataset: The dataset with an additional column for predicted outputs.
#         """
#         alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

#         predictions = []
#         for index, row in dataset.iterrows():
#             inputs = self.tokenizer(
#                 [
#                     alpaca_prompt.format(
#                         row['instruction'],  # instruction
#                         row['input'],  # input
#                         "",  # output - leave this blank for generation!
#                     )
#                 ], return_tensors="pt").to("cuda")

#             outputs = self.model.generate(**inputs, max_new_tokens=128, use_cache=True)
#             print(outputs)
#             prediction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#             predictions.append(prediction)

#         dataset['predicted_output'] = predictions
#         return dataset

    def inference(self, sys_prompt, instruction, input_text, max_new_tokens = 128):
      prompt_template = self.get_prompt[self.model_name]
      EOS_TOKEN = self.tokenizer.eos_token

      def formatting_prompt_func(sys_prompt, instruction, input_text):
        return prompt_template.format(sys_prompt, instruction, input_text, "") + EOS_TOKEN

      formatted_prompt = formatting_prompt_func(sys_prompt, instruction, input_text)
      print(formatted_prompt)

      inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

      outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
      text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

      regex_pattern = r"assistant(.*)"
      # Extract the matched group
      match = re.search(regex_pattern, text, re.DOTALL)  # re.DOTALL to match across newlines

      if match:
          matched_text = match.group(1).strip()
          return matched_text, outputs
      else:
          return text, outputs




      return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0], outputs

    def inference_n_shots(self, sys_prompt, instruction, input_text, n_shots,max_new_tokens = 128):
          #n_shots is a list of tuples of  input_texts, response
          prompt_template = self.get_prompt[self.model_name]
          EOS_TOKEN = self.tokenizer.eos_token

          def formatting_prompt_func(sys_prompt, instruction, input_text, response = ""):
            return prompt_template.format(sys_prompt, instruction, input_text, response) + EOS_TOKEN
          #write code to include n_shots in the formated prompt
          formatted_prompt = ""
          for txt, res in n_shots:
            formatted_prompt += formatting_prompt_func(sys_prompt, instruction, txt,res)


          formatted_prompt += formatting_prompt_func(sys_prompt, instruction, input_text)
          print(formatted_prompt)

          inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

          outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
          text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

          regex_pattern = r"assistantassistant(.*)"
          # Extract the matched group
          matches = re.findall(regex_pattern, text, re.DOTALL)    # re.DOTALL to match across newlines

          if matches:
              matched_text = matches[0].strip()
              return matched_text, outputs
          else:
              return text, outputs

    def prepare_for_cloud(self, save_mehod, save_pretrained_metge = False, user_name="", token=""):
        # Merge to 16bit
        if save_mehod == "merged_16bit":
          if save_pretrained_metge: self.model.save_pretrained_merged("model", self.tokenizer, save_method="merged_16bit",)
          if False: self.model.push_to_hub_merged(user_name, self.tokenizer, save_method="merged_16bit", token=token)

        # Merge to 4bit
        elif save_mehod == "merged_4bit":
          if False: self.model.save_pretrained_merged("model", self.tokenizer, save_method="merged_4bit",)
          if False: self.model.push_to_hub_merged("hf/model", self.tokenizer, save_method="merged_4bit", token="")

        # Just LoRA adapters
        elif save_mehod == "lora":
          if True: self.model.save_pretrained_merged("model", self.tokenizer, save_method="lora",)
          if True: self.model.push_to_hub_merged("hf/model", self.tokenizer, save_method="lora", token="")

        # Save to 8bit Q8_0
        elif save_mehod == "q8_0":
          if False: self.model.save_pretrained_gguf("model", self.tokenizer,)
          if False: self.model.push_to_hub_gguf("hf/model", self.tokenizer, token="")

        # Save to 16bit GGUF
        elif save_mehod == "f16":
          if False: self.model.save_pretrained_gguf("model", self.tokenizer, quantization_method="f16")
          if False: self.model.push_to_hub_gguf("hf/model", self.tokenizer, quantization_method="f16", token="")

        # Save to q4_k_m GGUF
        elif save_mehod == "q4_k_m":
          if False: self.model.save_pretrained_gguf("model", self.tokenizer, quantization_method="q4_k_m")
          if False: self.model.push_to_hub_gguf("hf/model", self.tokenizer, quantization_method="q4_k_m", token="")
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
] #
@click.command()
@click.option('--resume', default=False, help='Resume training from a checkpoint.')
@click.option('--dataset_path', required=True, help='Path to the dataset to load.')

def main(resume,dataset_path="data"):
    llamafine = LLamafine()
    llamafine.initialize_model(model_name="unsloth/llama-3-8b-Instruct-bnb-4bit")
    llamafine.apply_peft_model(config_file="config/lora_config.json")
    loaded_dataset = load_from_disk(dataset_path)
    dataset = llamafine.prepare_dataset_llama3(loaded_dataset)
    #make it passable
    trainer = llamafine.train_model(dataset, config_file="config/training_config.json",output_dir='checkpoints',resume_from_checkpoint = resume)
    llamafine.save_model_to_hub( user_name="mbayan/fine-tuned-social-media-analytics_1", token="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")
