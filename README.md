
# A Comparison Evaluation Between Qwen2.5-1.5B and Qwen2.5-7B

## Introduction & Methodology

This study presents a performance comparison between Alibaba's open-weight models Qwen2.5-1.5B and Qwen2.5-7B using Google Colab. To prevent Out-of-Memory (OOM) issues, all evaluations were conducted using FP16 precision. Generation parameters were set to temperature 0.1 and top\_p 0.95 to ensure stable outputs.

The evaluation utilized Eleuther AI's LM Harness library, which integrates with Hugging Face and DeepSpeed, significantly streamlining the assessment process. Models were loaded through Hugging Face and inference was accelerated using DeepSpeed optimization.

## Benchmark Evaluations

### MMLU Evaluation

The MMLU (Massive Multitask Language Understanding) dataset evaluates model comprehension across multiple academic disciplines. Results demonstrate that Qwen2.5-7B consistently outperforms its smaller counterpart across all domains, with an overall accuracy improvement of approximately 12 percentage points.

| TASK              | Qwen2.5-1.5B Accuracy | Qwen2.5-7B Accuracy | Improvement |
| :---------------- | :-------------------- | :------------------ | :---------- |
| **Overall**       | 59.74% ± 0.39%        | 71.90% ± 0.35%      | +12.16%     |
| abstract_algebra  | 35.00% ± 4.79%        | 54.00% ± 5.01%      | +19.00%     |
| anatomy           | 71.05% ± 3.69%        | 83.55% ± 3.02%      | +12.50%     |
| astronomy         | 61.00% ± 4.90%        | 76.00% ± 4.29%      | +15.00%     |
| *... (more tasks)* | *...*                 | *...*               | *...*       |

### HellaSwag Evaluation

HellaSwag evaluates models' natural language inference capabilities. Standard results are based on raw log probabilities, while normalized results adjust for question difficulty, providing a more equitable comparison. Qwen2.5-7B shows approximately 10-11 percentage point improvements over the smaller model.

| Indicator   | Qwen2.5-1.5B Accuracy | Qwen2.5-7B Accuracy | Improvement |
| :---------- | :-------------------- | :------------------ | :---------- |
| Standard    | 50.24% ± 0.50%        | 60.01% ± 0.49%      | +9.77%      |
| Normalized  | 67.75% ± 0.47%        | 78.93% ± 0.41%      | +11.18%     |

### MBPP Evaluation

The MBPP (Mostly Basic Python Problems) dataset evaluates coding proficiency through 500 Python programming challenges. Initial zero-shot evaluations yielded 0% accuracy due to format issues. After switching to two-shot evaluation, results showed that Qwen2.5-7B's programming accuracy surpasses the smaller model by 16.2 percentage points.

| Model         | Qwen2.5-1.5B Accuracy   | Qwen2.5-7B Accuracy | Improvement |
| :---------- | :-------------------- | :------------------ | :---------- |
| Qwen2.5-1.5B  | 46.00% ± 2.23%          | 62.20% ± 2.17%      |+16.20%      |


## Key Findings and Conclusion

Takeaways from the evaluation:

*   **Evaluation toolkit selection is crucial:** Using established evaluation libraries like LM Harness significantly streamlines the assessment process compared to developing custom deployment and metrics.
*   **Different tasks require tailored evaluation approaches:** Particularly evident in the MBPP programming assessment, where transitioning from zero-shot to two-shot evaluation with appropriate prompt engineering substantially improved results.
*   **Non-linear relationship between parameter counts and performance gains:** A fivefold increase in parameters yields only 10-20% performance improvement, consistent with established scaling laws.

Based on the comparative results between these Qwen2.5 models, the author contends that achieving artificial general intelligence solely through expanding model parameters and training data may be insufficient. While large language models can serve as excellent AI agents, reaching true AGI likely requires breakthrough methodological innovations beyond simple scaling.

