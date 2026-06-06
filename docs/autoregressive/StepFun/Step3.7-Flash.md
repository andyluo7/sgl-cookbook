# Step-3.7-Flash

## 1. Model Introduction

[Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8) is StepFun's 198B-parameter sparse Mixture-of-Experts (MoE) vision-language model. It pairs a 196B-parameter language backbone with a 1.8B-parameter vision encoder for native image understanding, activates ~11B parameters per token, and is engineered for high-frequency production agentic workloads with up to 400 tokens/s throughput.

The model targets agentic workflows that combine perception, search, and reasoning — parsing long financial reports in a single pass, running multi-step search loops with cross-source verification, or operating concurrent coding agents in high-throughput pipelines.

**Key Features:**

- **Sparse Mixture-of-Experts**: 198B total parameters, ~11B activated per token
- **Vision-Language**: 1.8B-parameter vision encoder for native image understanding
- **256K Context Window**: Supports long-document and long-trajectory agent workloads
- **Three Reasoning Levels**: Low, medium, and high selectable reasoning depth to trade off speed, cost, and cognitive depth
- **Tool Calling**: Built-in `step3p5` tool-call parser support
- **Hybrid Reasoning**: Built-in `step3p5` reasoning parser separates `<think>` content from the final answer

**Available Models:**

| Model | Weights |
|-------|---------|
| Step-3.7-Flash-FP8 | [stepfun-ai/Step-3.7-Flash-FP8](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8) |

**Benchmarks** (from the official model card):

| Benchmark | Score |
| --- | --- |
| SimpleVQA (Search) | 79.2 |
| V* (Python) | 95.3 |
| ClawEval-1.1 | 67.1 |
| Toolathlon | 49.5 |
| HLE w. Tool | 48.1 |
| SWE-Bench PRO | 56.3 |
| Terminal-Bench 2.1 | 59.5 |
| GDPVal-AA | 45.8 |

**License:** Refer to the [model card](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8) for the latest license terms.

**Links:**

- HuggingFace: [stepfun-ai/Step-3.7-Flash-FP8](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8)
- Blog: [StepFun — Step 3.7 Flash](https://static.stepfun.com/blog/step-3.7-flash/)

## 2. SGLang Installation

For installation methods (PyPI, source, Docker), refer to the [official SGLang installation guide](https://docs.sglang.ai/get_started/install.html).

### Docker Images by Hardware Platform

| Hardware Platform                | Docker Image                                                                    |
| ---                              | ---                                                                             |
| AMD MI300X / MI325X              | `lmsysorg/sglang-rocm:v0.5.10.post1-rocm700-mi30x-20260428`                     |
| AMD MI350X / MI355X              | `lmsysorg/sglang-rocm:v0.5.10.post1-rocm700-mi35x-20260428`                     |

Launch the container with GPU access:

```bash
# AMD MI300X / MI325X
docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size 32G --network=host \
  -v <path-to-models>:/models \
  lmsysorg/sglang-rocm:v0.5.10.post1-rocm700-mi30x-20260428 bash

# AMD MI350X / MI355X
docker run -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  --ipc=host --shm-size 32G --network=host \
  -v <path-to-models>:/models \
  lmsysorg/sglang-rocm:v0.5.10.post1-rocm700-mi35x-20260428 bash
```

## 3. Model Deployment

This section provides deployment configurations optimized for different hardware platforms and use cases.

### 3.1 Basic Configuration

**Interactive Command Generator**: Use the configuration selector below to automatically generate the appropriate deployment command for your hardware platform.

import Step37FlashConfigGenerator from '@site/src/components/autoregressive/Step37FlashConfigGenerator';

<Step37FlashConfigGenerator />

### 3.2 Configuration Tips

- **Hardware**: Step-3.7-Flash-FP8 is ~198 GB on disk. Single-node 8×GPU AMD configurations (MI300X 192GB, MI325X 256GB, MI350X 288GB, MI355X 288GB) all fit comfortably with TP=8.
- **Tensor + Expert Parallelism**: AMD GPUs use `--tp 8 --ep 8` for the 198B-parameter MoE. Expert parallelism distributes the 256 experts across the 8 ranks so each rank holds a slice of the expert weights, which is more memory-efficient than full replication.
- **Trust Remote Code**: `--trust-remote-code` is required for the `Step3p7ForConditionalGeneration` architecture defined in the model repository.
- **AITER Kernels**: AMD's AITER kernels are bundled in the `lmsysorg/sglang-rocm:*-rocm700-mi30x|mi35x-*` images and are used automatically. The first request triggers JIT compilation of the relevant kernels; subsequent requests reuse cached `.so` files.
- **Long Context**: The model's `max_model_len` defaults to 262144 (256K). If you encounter OOM during KV cache allocation, reduce with `--context-length <N>` or lower `--mem-fraction-static`.

## 4. Model Invocation

Deploy Step-3.7-Flash-FP8 with the following command (AMD MI300X/MI325X/MI350X/MI355X, all features enabled):

```shell
sglang serve \
  --model-path stepfun-ai/Step-3.7-Flash-FP8 \
  --tp 8 \
  --ep 8 \
  --trust-remote-code \
  --reasoning-parser step3p5 \
  --tool-call-parser step3p5 \
  --host 0.0.0.0 \
  --port 30000
```

### 4.1 Basic Usage

For basic API usage and request examples, please refer to:

- [SGLang Basic Usage Guide](https://docs.sglang.ai/basic_usage/send_request.html)

### 4.2 Reasoning Parser

Step-3.7-Flash uses the `step3p5` reasoning parser to separate the model's thinking process from the final answer. Enable it at deploy time and read the `reasoning_content` field on the response.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="stepfun-ai/Step-3.7-Flash-FP8",
    messages=[
        {"role": "user", "content": "Solve this problem step by step: What is 15% of 240?"}
    ],
    max_tokens=2048,
    stream=True
)

has_thinking = False
has_answer = False
thinking_started = False

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta

        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            if not thinking_started:
                print("=============== Thinking =================", flush=True)
                thinking_started = True
            has_thinking = True
            print(delta.reasoning_content, end="", flush=True)

        if delta.content:
            if has_thinking and not has_answer:
                print("\n=============== Content =================", flush=True)
                has_answer = True
            print(delta.content, end="", flush=True)

print()
```

**Output Example:**

```text
Pending update...
```

### 4.3 Tool Calling

Step-3.7-Flash uses the `step3p5` tool-call parser. Enable both the reasoning parser and the tool-call parser at deploy time so the model can plan a tool call inside `<think>` and then emit a structured function call.

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["location"]
            }
        }
    }
]

def get_weather(location, unit="celsius"):
    return f"The weather in {location} is 22°{unit[0].upper()} and sunny."

print("--- Sending first request ---")
response = client.chat.completions.create(
    model="stepfun-ai/Step-3.7-Flash-FP8",
    messages=[
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    tools=tools,
    stream=False
)

message = response.choices[0].message

reasoning = getattr(message, 'reasoning_content', None)
if reasoning:
    print("=============== Thinking =================")
    print(reasoning)
    print("==========================================")

if message.tool_calls:
    print("\n🔧 Tool Calls detected:")
    history_messages = [
        {"role": "user", "content": "What's the weather in Beijing?"},
        message
    ]

    for tool_call in message.tool_calls:
        print(f"   Tool: {tool_call.function.name}")
        print(f"   Args: {tool_call.function.arguments}")

        args = json.loads(tool_call.function.arguments)
        tool_result = get_weather(args.get("location"), args.get("unit", "celsius"))

        history_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

    print("\n--- Sending tool results ---")
    final_response = client.chat.completions.create(
        model="stepfun-ai/Step-3.7-Flash-FP8",
        messages=history_messages,
        stream=False
    )

    final_message = final_response.choices[0].message
    print("=============== Final Reasoning =================")
    print(getattr(final_message, 'reasoning_content', None))
    print("=============== Final Content =================")
    print(final_message.content)
else:
    if message.content:
        print("=============== Content =================")
        print(message.content)
```

**Output Example:**

```text
Pending update...
```

## 5. Benchmark

### 5.1 Speed Benchmark

Pending update...

### 5.2 Accuracy Benchmark

Pending update...
