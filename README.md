# DriveSOTIF: Advancing Perception SOTIF Through Multi-Modal Large Language Models
 **[Paper (IEEE TVT)](https://ieeexplore.ieee.org/document/11162558)**  
 **[arXiv](https://arxiv.org/abs/2505.07084)** (Note: copyright transferred to IEEE)  
 **[Supplementary material](supplemental.pdf)**  
* Authors: Shucheng Huang, Freda Shi, Chen Sun, Jiaming Zhong, Minghao Ning, Yufeng Yang, Yukun Lu, Hong Wang, Amir Khajepour

DriveSOTIF is the first work to focus on addressing perception-related SOTIF challenges in autonomous driving using multimodal large language models (MLLMs). We introduce the DriveSOTIF dataset and benchmark several state-of-the-art MLLMs (Qwen 2.5 VL and InternVL3) on it. We also provide sample codes for dataset generation and ensemble yolo models to enhance perception performance and estimate uncertainty.

## Abstract
Human drivers possess spatial and causal intelligence, enabling them to perceive driving scenarios, anticipate hazards, and react to dynamic environments. In contrast, autonomous vehicles lack these abilities, making it challenging to manage perception-related Safety of the Intended Functionality (SOTIF) risks, especially under complex or unpredictable driving conditions. To address this gap, we propose fine-tuning multimodal large language models (MLLMs) on a customized dataset specifically designed to capture perception-related SOTIF scenarios. Benchmarking results show that fine-tuned MLLMs achieve an 11.8% improvement in close-ended VQA accuracy and a 12.0% increase in open-ended VQA scores compared to baseline models, while maintaining real-time performance with a 0.59-second average inference time per image. We validate our approach through real-world case studies in Canada and China, where fine-tuned models correctly identify safety risks that challenge even experienced human drivers. This work represents the first application of domain-specific MLLM fine-tuning for the SOTIF domain in autonomous driving.


## Dataset and code release:
Since DriveSOTIF is built upon the PeSOTIF dataset, we wont be able to release the dataset directly due to licensing issues. However, we release sample codes that you can refer to build your own dataset. You can find the released code in the code folder. The code folder contains the following parts:

### Dataset generation scripts (code/gen_dataset.py):
This is demo code to generate a dataset with caption, question and answer pairs for fine-tuning MLLMs.
It contains the following features:
1. Image caption generation, evaluation and metadata extraction
2. Question generation, evaluation and metadata extraction
3. Answer generation, evaluation and metadata extraction
4. Overall evaluation and post-processing

For the evaluation part, we currently check every single caption, question and answer and regenerate failed entries with GPT-4o/GPT-5 to ensure the quality of the generated dataset. You can sample (eg. 2-3 out of 5) generated captions, questions and answers and evaluate them with GPT-4o/GPT-5 to save API credits.

Please be aware that this released demo code is built using GPT and Claude code for code streamlining purposes, and thus the code may look less human like (eg. ~1500 lines of code in a single file). However, the core logic and structure of the code is still human written. The code is modularized and well commented for easy understanding. Also, please note that the code is only a demo version for dataset generation and don't include the switching LLM backends. But it is quite straightforward to integrate with any LLM backend of your choice. Note that using proprietary LLMs such as GPT-4o/GPT-5 etc might consume a significant amount of API credits, especially when processing a large number of images. We recommend testing the code with a small batch of images first to estimate the cost before scaling up.

### Ensemble Yolo models for perception (code/sotif_emsemble.py):
We provide sample codes to ensemble multiple (5) yolo models to enhance perception performance and estimate uncertainty based on the SOTIF entropy method proposed in "SOTIF entropy: Online SOTIF risk quantification and mitigation for autonomous driving".  This is not an **official** implementation of the SOTIF entropy method, but our own implementation based on the method described in the paper.

Please be aware that in the original SOTIF entropy paper, the authors used YOLOv5 models, which officially support ensembling.
However, in this released code, we used YOLOv8 models, which do not provide complete category-level confidence outputs. To obtain the category-level confidence outputs, the hard way is to modify the YOLOv8 source code and expose the required outputs. We took a shortcut by using the bounding box-level confidence outputs and aggregating them to get category-level confidence outputs. This is essentially an conservative approximation of the original SOTIF entropy method, which may lead to slightly different uncertainty estimates. However, we found that this approximation still works well in practice and provides a good balance between performance and complexity.


For more details, please refer to the code provided and our supplementary material. Below is a brief summary of the method and our implementation:
1. **Aggregate ensemble confidence.**  
   For each object cluster (detections across models that overlap and share class), compute the mean confidence  
   `s_bar = mean([s_m for each model m])`.

2. **Base (binary) entropy.**  
   Treat the problem as “detected class” vs “all others” with probability `p = s_bar`.  
   Binary entropy:  
   `H_b(p) = -( p * log p + (1 - p) * log(1 - p) )  `.

3. **Approximate other classes (sampling heuristic).**  
   Since YOLOv8 hides the rest of the class scores, we:
   - Sample several plausible class-probability vectors that distribute `1 - s_bar` across the remaining `K-1` classes (using a helper like `random_prob(s_bar, K)`).
   - Compute the binary entropy for each sampled top-class probability and take the **maximum** (worst case).
   - Combine with the base entropy to get a **conservative SOTIF entropy**:  
     `H_SOTIF = max_n H_b(p^(n)) + H_b(s_bar)`.

4. **Weak consensus adjustment.**  
   If fewer than 5 models support a cluster, inflate uncertainty:  
   `H_final = H_SOTIF * (1 + 0.1 * (5 - c))`, where `c` is the number of contributing detections.
   5 can be tuned based on the number of models in the ensemble.

5. **Uncertainty buckets.**  
   - **Low:** `H < 1.2`  
   - **Medium:** `1.2 <= H < 1.6`  
   - **High:** `H >= 1.6`  

## Citation
If you find DriveSOTIF useful in your research, please cite:
```
@article{huang2025drivesotif,
  title={DriveSOTIF: Advancing SOTIF Through Multimodal Large Language Models},
  author={Huang, Shucheng and Shi, Freda and Sun, Chen and Zhong, Jiaming and Ning, Minghao and Yang, Yufeng and Lu, Yukun and Wang, Hong and Khajepour, Amir},
  journal={IEEE Transactions on Vehicular Technology},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements
We would like to acknowledge the financial support
of the Natural Sciences and Engineering Research Council of
Canada (NSERC) and Vector Institute. We also acknowledge
the research grant provided by OpenAI and computing resources provided by the Vector Institute.


## License:
The PeSOTIF and CADC dataset are accessed under CC BY-NC-SA 4.0 license. 
Proprietary LLMs, such as GPT4, GPT4o, etc. are accessed under their licenses, terms and conditions. 
Open-source LLM models and LVLM, such as Blip, Blip2, LLAVA, Qwen2-VL, etc., are accessed under their corresponding licenses.
