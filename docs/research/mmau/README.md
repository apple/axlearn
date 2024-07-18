# MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains

![MMAU Logo](./figures/MMAU-herofig.png)


[![Dataset](https://img.shields.io/badge/Dataset-orange)](./README.md#dataset-preparation)
[![arXiv](https://img.shields.io/badge/arXiv-gray)](https://arxiv.org/abs/2407.18961)
[![GitHub](https://img.shields.io/badge/GitHub-black)](https://github.com/apple/axlearn/docs/research/mmau)

---

## Introducing MMAU 🎉

The Massive Multitask Agent Understanding (MMAU) benchmark is designed to evaluate the performance of large language models (LLMs) as agents across a wide variety of tasks. It provides comprehensive insights into the capabilities and limitations of these models by featuring extensive offline tasks that eliminate the need for complex environment setups.

MMAU evaluates models across five key domains:
- **Tool-use**
- **Directed Acyclic Graph (DAG) QA**
- **Data Science and Machine Learning Coding**
- **Contest-level Programming**
- **Mathematics**

These domains cover five essential capabilities:
- **Understanding**
- **Reasoning**
- **Planning**
- **Problem-solving**
- **Self-correction**

With a total of 20 meticulously designed tasks encompassing over 3,000 distinct prompts, MMAU offers a robust framework for assessing the strengths and weaknesses of LLM agents.


## Table of Contents

- [Overview](#overview)
- [Dataset Summary](#dataset-summary)
- [Quick Start](#quick-start)
- [Data prepartion](#dataset-preparation)
- [Benchmark](#benchmark)
- [Citation](#citation)

---

## Overview

### Key Evaluation Results on MMAU

![MMAU Results](./figures/results_radar_bar_combined.png)

### MMAU key features

- **Comprehensive Evaluation**: MMAU provides evaluations from both application scenarios and fundamental capabilities. This dual approach offers a holistic framework for understanding the strengths and limitations of LLM agents.
- **Simplified Evaluation Process**: The evaluation process on MMAU is straightforward and unified on a static dataset. This method avoids the instability issues that can arise from interactive evaluations, ensuring more reliable and consistent results.
- **Open Access**: We release our evaluation dataset and scripts to the public. This transparency aims to set a new standard for performance assessment in the AI landscape, encouraging further research and development.

### Benchmark Comparison Table
Table 1: Comparison of benchmarks in evaluating core capabilities of LLM agents. “En.” and “Dis.” represent entangled and disentangled, specifically. Understand.: understanding, Reason.: reasoning, Plan.: planning, Prob.-solv.: problem-solving, Self-corr.: self-correction, MM: multimodal grounding.

| Benchmarks    | Understand. En. | Understand. Dis. | Reason. En. | Reason. Dis. | Plan. En. | Plan. Dis. | Prob.-solv. En. | Prob.-solv. Dis. | Self-corr. | MM |
|---------------|-----------------|------------------|-------------|--------------|-----------|------------|-----------------|------------------|------------|----|
| [AgentBench](https://arxiv.org/abs/2308.03688) | ✅               | ❌                | ✅           | ❌            | ✅         | ❌          | ✅               | ❌                | ✅          | ✅  |
| [AgentBoard](https://arxiv.org/abs/2401.13178) | ✅               | ❌                | ✅           | ❌            | ✅         | ❌          | ✅               | ❌                | ✅          | ❌  |
| [PlanBench](https://arxiv.org/abs/2206.10498)  | ✅               | ❌                | ✅           | ❌            | ✅         | ❌          | ✅               | ❌                | ❌          | ❌  |
| [MMLU](https://arxiv.org/abs/2009.03300)        | ✅               | ❌                | ✅           | ❌            | ✅         | ❌          | ✅               | ❌                | ❌          | ❌  |
| [MMMU](https://arxiv.org/abs/2311.16502)       | ✅               | ❌                | ✅           | ❌            | ✅         | ❌          | ✅               | ❌                | ✅          | ❌  |
| MMAU           | ✅               | ✅                | ✅           | ✅            | ✅         | ✅          | ✅               | ✅                | ✅          | ✅  |


MMAU's extensive and meticulously curated dataset, along with its robust evaluation framework, provides a valuable resource for assessing and advancing the capabilities of large language models.


## Dataset Summary

The construction of MMAU encompasses both breadth and depth of data, drawn from a diverse range of sources to ensure comprehensive evaluation across various domains.

Our dataset is constructed from heterogeneous sources, including:

1. **In-house Tool-use Data**: Utilized for tasks under tool-use and Directed Acyclic Graph (DAG) QA, providing a robust foundation for evaluating these specific capabilities.
2. **[Kaggle](https://www.kaggle.com/datasets) Datasets**: Rewritten to design tasks for Data Science and Machine Learning (DS & ML) coding, ensuring practical and relevant challenges that reflect real-world scenarios.
3. **[CodeContest](https://www.codecontest.com)**: Sourced for tasks under contest-level programming, offering high-level coding challenges that test the advanced problem-solving skills of LLMs.
4. **[DeepMind Math](https://github.com/deepmind/mathematics_dataset) Dataset**: Used for math tasks, providing a rigorous set of problems to evaluate mathematical reasoning and computational abilities.

MMAU involves meticulous curation and rewriting of these data sources to create 20 diverse tasks encompassing over 3,000 distinct prompts. This extensive and varied dataset allows for a thorough assessment of LLM agents, covering five essential capabilities: understanding, reasoning, planning, problem-solving, and self-correction.


## Quick Start

To get started quickly, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/apple/axlearn.git
    ```
2. Navigate to the repository directory:
    ```sh
    cd axlearn
    ```
3. Install the necessary dependencies:
    ```sh
    # Install for all mmau metrics.
    pip install ".[mmau]"
    # Install for generator only.
    pip install ".[open_api]"
    ```

## Dataset Preparation

Download from GCP (Coming soon).
```sh
mkdir -p ./data/
gsutil -m cp -r "gs://axlearn-public/datasets/mmau/20240712/*" ./data/
```

Download from Huggingface (Coming soon).
```sh
huggingface-cli download apple/mmau --local-dir ./data --repo-type dataset
```

## Benchmark

<details>
  <summary>Click to expand the Benchmark section</summary>


### Generator

The first step is to generate responses from the target client. The generator is designed with following features:

- **Reusable**: same input data (OpenAI request style json line file) across different model clients.
- **Fast**: asynchronously concurrent requests.
- **Extendible**: easy to add new clients and open source models (via [vLLM OpenAI endpoint](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or other similar inference frameworks).

<details>
  <summary>Input Data Format</summary>


**Chat endpoint (Recommended):**

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help you?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "id": 1
}
```


**Completion endpoint:**
```json
{
  "prompt": "Q: What color is mars? A:",
  "id": 1
}
```
</details>


**OpenAI**

```sh
export OPENAI_API_KEY=<your_openai_key>
MODEL=gpt-3.5-turbo-0125
EVAL_SET=tool_use_single_step_20240712.jsonl
CLIENT_NAME=openai
python3 -m axlearn.open_api.generator \
--model $MODEL \
--client_name $CLIENT_NAME \
--input_file ./data/$EVAL_SET \
--output_file ./generated_data/$MODEL/$EVAL_SET \
--decode_parameters '{"temperature": 0.0, "max_tokens": 1024}'
```

**Gemini**

Add following modifications from above OpenAI command examples.
```sh
# Set up environments.
export VERTEX_AI_PROJECT=<your_project_name>
export VERTEX_AI_LOCATION=<your_project_location>
MODEL=gemini-1.0-pro
# Change CLIENT_NAME from openai to gemini.
CLIENT_NAME=gemini
```

**Anthropic**

Add following modifications from above OpenAI command examples.
```sh
# Set up environments.
export ANTHROPIC_API_KEY=<sk-xxxx>
MODEL=claude-3-haiku-20240307
```
Change to `--client_name anthropic` in the above OpenAI example command.


**Open Source Models**

Use [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or other inference framework to start an OpenAI compatible server. In this case, `--client_name=openai` can be re-used. Note it may need implementations of new clients for function calling style.

```sh
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1/8000/v1
MODEL=Mistral-7B-Instruct-v0.3
CLIENT_NAME=openai
```

### Evaluator

**[Tool Use Execution](../../axlearn/open_api/metrics/tool_use_execution.py)**

```sh
EVAL_SET=tool_use_single_step_20240712.jsonl
MODEL=gpt-3.5-turbo-0125
python3 -m axlearn.open_api.evaluator \
--input_file ./generated_data/$MODEL/$EVAL_SET \
--output_file ./metrics/$MODEL/$EVAL_SET \
--metric_name tool_use_execution
```

The same metric can be used for tool use execution retry tasks.

**[Tool Use Plan](../../axlearn/open_api/metrics/tool_use_plan.py)**

Change to use `--metric_name tool_use_plan` from the above Tool Use Execution example command.

**[Math](../../axlearn/open_api/metrics/math.py)**

This benchmark needs OpenAI client as LLM grader.
```sh
export OPENAI_API_KEY=<your_openai_key>
MODEL=gpt-3.5-turbo-0125
EVAL_SET=mmau_math_standard_20240712.jsonl
python3 -m axlearn.open_api.evaluator \
--input_file ./generated_data/$MODEL/$EVAL_SET \
--output_file ./metrics/$MODEL/$EVAL_SET \
--metric_name math \
--grader_model gpt-4o-2024-05-13
```

**[Code Contests](../../axlearn/open_api/metrics/code_contests.py)**

This benchmark needs OpenAI client as LLM grader.

Change to use `--metric_name code_contests` from the above Math example command.

There are also `code_contests_retry`, `code_contests_plan` and `code_contests_understand` for detailed benchmark and study.

**[Code Kaggle](../../axlearn/open_api/metrics/code_kaggle.py)**

This benchmark needs OpenAI client as LLM grader.

Change to use `--metric_name code_kaggle` from the above Math example command.

There are also `code_kaggle_oracle` and `code_kaggle_retry` for detailed benchmark and study.

### All in One

We will provide a simple script to run all benchmarks soon. Stay tuned.

## TODO

- [ ] Add all in one script to launch all benchmarks.
- [ ] Add more open api clients.
- [ ] Load from Huggingface datasets directly.

</details>


## Citation

If you use MMAU in your research, please cite our paper:

```bibtex
@article{patil2024mmau,
  title={MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains},
  author={Guoli Yin, Felix Bai, Shuang Ma, Zirui Wang, et al.},
  year={2024},
  url={https://github.com/apple/axlearn/docs/research/mmau},
  journal={https://arxiv.org/abs/2407.18961},
}
