This folder includes scripts to generate ImageNet Adversarial Text Regions (ImageNet-Atr) dataset, which is used as the evaluation set in our paper ["Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness", 2023](https://arxiv.org/abs/2305.05095).

To generate the dataset by your own: see [add_attack_tfrecord.py](add_attack_tfrecord.py). It does not rely on other AXLearn libraries. Instead, we can run the code on a laptop with just the current folder.

The dataset is available on GCP as a Tensorflow Dataset: `gs://axlearn-public/tensorflow_datasets/imagenet2012_ocr_attack/`.


## Recognition results of different CLIP models

| Model            | Num. Training | ImageNet2012 Top-1 Acc | ImageNet_Atr  Top-1 Acc|
| ---------------- | -----------   |----------------------- |------------- |
| OpenAI CLIP B16  | 400M          | 68.35%       | 31.65%       |
| OpenCLIP B16     | 400M          | 66.99%       | 29.55%       |
| Our CLIP B16     | 1.1B          | 68.66%       | 35.73%       |
| Our Filter-based CLIP B16 | 0.7B | 70.77%       | 68.78%       |

See [zeroshot_eval_with_opensource_clip.ipynb](zeroshot_eval_with_opensource_clip.ipynb) for evaluation code.
