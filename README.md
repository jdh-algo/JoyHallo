# JoyHallo: Digital human model for Mandarin

<br>
<div align='left'>
    <a href='https://jdh-algo.github.io/JoyHallo'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/pdf/2409.13268'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/jdh-algo/JoyHallo-v1'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://huggingface.co/spaces/jdh-algo/JoyHallo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow'></a>
</div>
<br>

## 📖 Introduction

In audio-driven video generation, creating Mandarin videos presents significant challenges. Collecting comprehensive Mandarin datasets is difficult, and the complex lip movements in Mandarin further complicate model training compared to English. In this study, we collected 29 hours of Mandarin speech video from JD Health International Inc. employees, resulting in the jdh-Hallo dataset. This dataset includes a diverse range of ages and speaking styles, encompassing both conversational and specialized medical topics. To adapt the JoyHallo model for Mandarin, we employed the Chinese wav2vec2 model for audio feature embedding. A semi-decoupled structure is proposed to capture inter-feature relationships among lip, expression, and pose features. This integration not only improves information utilization efficiency but also accelerates inference speed by 14.3%. Notably, JoyHallo maintains its strong ability to generate English videos, demonstrating excellent cross-language generation capabilities.

## 🎬 Videos-Mandarin-Woman

https://github.com/user-attachments/assets/389e053f-e0c4-433c-8c60-80f9181d3f9c

## 🎬 Videos-Mandarin-Man

https://github.com/user-attachments/assets/1694efd9-2577-4bb5-ada4-7aa711d016a6

## 🎬 Videos-English

https://github.com/user-attachments/assets/d6b2efea-be76-442e-a8aa-ea0eef8b5f12

## 🧳 Framework

![Network](assets/network.png "Network")

## ⚙️ Installation

System requirements:

- Tested on Ubuntu 20.04, Cuda 11.3
- Tested GPUs: A100

Create environment:

```bash
# 1. Create base environment
conda create -n joyhallo python=3.10 -y
conda activate joyhallo

# 2. Install requirements
pip install -r requirements.txt

# 3. Install ffmpeg
sudo apt-get update  
sudo apt-get install ffmpeg -y
```

## 🎒 Prepare model checkpoints

### 1. Download base checkpoints

Use the following command to download the base weights:

```shell
git lfs install
git clone https://huggingface.co/fudan-generative-ai/hallo pretrained_models
```

### 2. Download chinese-wav2vec2-base model

Use the following command to download the `chinese-wav2vec2-base` model:

```shell
cd pretrained_models
git lfs install
git clone https://huggingface.co/TencentGameMate/chinese-wav2vec2-base 
```

### 3. Download JoyHallo model

```bash
git lfs install
git clone https://huggingface.co/jdh-algo/JoyHallo-v1 pretrained_models/joyhallo
```

For convenience, we have uploaded the model weights to **Hugging Face**.

|  Model  |  Dataset  |                     Hugging Face                     |
| :------: | :-------: | :--------------------------------------------------: |
| JoyHallo | jdh-Hallo | [JoyHallo](https://huggingface.co/jdh-algo/JoyHallo-v1) |

### 4. pretrained_models contents

The final `pretrained_models` directory should look like this:

```text
./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- face_analysis/
|   `-- models/
|       |-- face_landmarker_v2_with_blendshapes.task
|       |-- 1k3d68.onnx
|       |-- 2d106det.onnx
|       |-- genderage.onnx
|       |-- glintr100.onnx
|       `-- scrfd_10g_bnkps.onnx
|-- hallo/
|   `-- net.pth
|-- joyhallo/
|   `-- net.pth
|-- motion_module/
|   `-- mm_sd_v15_v2.ckpt
|-- sd-vae-ft-mse/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- stable-diffusion-v1-5/
|   `-- unet/
|       |-- config.json
|       `-- diffusion_pytorch_model.safetensors
|-- wav2vec/
|   `-- wav2vec2-base-960h/
|       |-- config.json
|       |-- feature_extractor_config.json
|       |-- model.safetensors
|       |-- preprocessor_config.json
|       |-- special_tokens_map.json
|       |-- tokenizer_config.json
|       `-- vocab.json
`-- chinese-wav2vec2-base/
    |-- chinese-wav2vec2-base-fairseq-ckpt.pt
    |-- config.json
    |-- preprocessor_config.json
    `-- pytorch_model.bin
```

## 🚧 Data requirements

**Image**:

- Cropped to square shape.
- Face should be facing forward and occupy 50%-70% of the image.

**Audio**:

- Use `wav` format.
- Mandarin, English or mixed, with clear audio and suitable background music.

**Note**: These requirements apply to **both training and inference processes**.

## 🚀 Inference

### 1. Inference with command line

Use the following command to perform inference:

```bash
sh joyhallo-infer.sh
```

**Kindly remind**: If you want to improve the inference speed, you can change the `inference_steps` from **40** to **15** in `configs/inference/inference.yaml`. That will enhance the efficiency immediately. You can decrease that even more, but you may get a worse result. You can try changing `cfg_scale` together.

Modify the parameters in `configs/inference/inference.yaml` to specify the audio and image files you want to use, as well as switch between models. The inference results will be saved in `opts/joyhallo`. The parameters in `inference.yaml` are explained as follows:

* audio_ckpt_dir: Path to the model weights.
* ref_img_path: Path to the reference images.
* audio_path: Path to the reference audios.
* output_dir: Output directory.
* exp_name: Output file folder name.

### 2. Inference with web demo

Use the following command to start web demo:

```bash
sh joyhallo-app.sh
```

The demo will be create at [http://127.0.0.1:7860](http://127.0.0.1:7860).

## ⚓️ Train or fine-tune JoyHallo

You have two options when training or fine-tuning the model: start from **Stage 1** or only train  **Stage 2** .

### 1. Use the following command to start training from Stage 1

```
sh joyhallo-alltrain.sh
```

This will automatically start training both stages (including Stage 1 and Stage 2), and you can adjust the training parameters by referring to `configs/train/stage1_alltrain.yaml` and `configs/train/stage2_alltrain.yaml`.

### 2. Use the following command to train only Stage 2

```
sh joyhallo-train.sh
```

This will start training from  **Stage 2** , and you can adjust the training parameters by referring to `configs/train/stage2.yaml`.

## 🎓 Prepare training data

### 1. Prepare the data in the following directory structure, ensuring that the data meets the requirements mentioned earlier

```text
jdh-Hallo/
|-- videos/
|   |-- 0001.mp4
|   |-- 0002.mp4
|   |-- 0003.mp4
|   `-- 0004.mp4
```

### 2. Use the following command to process the dataset

```bash
# 1. Extract features from videos.
python -m scripts.data_preprocess --input_dir jdh-Hallo/videos --step 1 -p 1 -r 0
python -m scripts.data_preprocess --input_dir jdh-Hallo/videos --step 2 -p 1 -r 0

# 2. Get jdh-Hallo dataset.
python scripts/extract_meta_info_stage1.py -r jdh-Hallo -n jdh-Hallo
python scripts/extract_meta_info_stage2.py -r jdh-Hallo -n jdh-Hallo
```

**Note**: Execute steps 1 and 2 sequentially as they perform different tasks. Step 1 converts videos into frames, extracts audio from each video, and generates the necessary masks. Step 2 generates face embeddings using InsightFace and audio embeddings using Chinese wav2vec2, and requires a GPU. For parallel processing, use the `-p` and `-r` arguments. The `-p` argument specifies the total number of instances to launch, dividing the data into `p` parts. The `-r` argument specifies which part the current process should handle. You need to manually launch multiple instances with different values for `-r`.

## 💻 Comparison

### 1. Accuracy comparison in Mandarin

|  Model  | IQA $\uparrow$ | VQA $\uparrow$ | Sync-C $\uparrow$ | Sync-D $\downarrow$ | Smooth $\uparrow$ | Subject $\uparrow$ | Background $\uparrow$ |
| :------: | :--------------: | :--------------: | :----------------: | :------------------: | :----------------: | :-----------------: | :--------------------: |
|  Hallo  | **0.7865** |      0.8563      |       5.7420       |  **13.8140**  |       0.9924       |       0.9855       |    **0.9651**    |
| JoyHallo |      0.7781      | **0.8566** |  **6.1596**  |       14.2053       |  **0.9925**  |  **0.9864**  |         0.9627         |

Notes: The evaluation metrics used here are from the following repositories, and the results are for reference purposes only:

- IQA and VQA: [Q-Align](https://github.com/Q-Future/Q-Align)
- Sync-C and Sync-D: [Syncnet](https://github.com/joonson/syncnet_python)
- Smooth, Subject, and Background: [VBench](https://github.com/Vchitect/VBench)

### 2. Accuracy comparison in English

|  Model  | IQA $\uparrow$ | VQA $\uparrow$ | Sync-C $\uparrow$ | Sync-D $\downarrow$ | Smooth $\uparrow$ | Subject $\uparrow$ | Background $\uparrow$ |
| :------: | :--------------: | :--------------: | :----------------: | :------------------: | :----------------: | :-----------------: | :--------------------: |
|  Hallo  | **0.7779** |      0.8471      |       4.4093       |  **13.2340**  |       0.9921       |       0.9814       |    **0.9649**    |
| JoyHallo | **0.7779** | **0.8537** |  **4.7658**  |       13.3617       |  **0.9922**  |  **0.9838**  |         0.9622         |

### 3. Inference efficiency comparison

|                              | JoyHallo | Hallo |   Improvement   |
| :---------------------------: | :------: | :----: | :-------------: |
| GPU Memory (512*512, step 40) |  19049m  | 19547m | **2.5%** |
|  Inference Speed (16 frames)  |   24s   |  28s  | **14.3%** |

## 📝 Citations

If you find our work helpful, please consider citing us:

```
@misc{shi2024joyhallo,
  title={JoyHallo: Digital human model for Mandarin}, 
  author={Sheng Shi and Xuyang Cao and Jun Zhao and Guoxin Wang},
  year={2024},
  eprint={2409.13268},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2409.13268}, 
}
```

## 🤝 Acknowledgments

We would like to thank the contributors to the [Hallo](https://github.com/fudan-generative-vision/hallo), [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), [Chinese-wav2vec2](https://github.com/TencentGameMate/chinese_speech_pretrain), [Q-Align](https://github.com/Q-Future/Q-Align), [Syncnet](https://github.com/joonson/syncnet_python), [VBench](https://github.com/Vchitect/VBench), and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) repositories, for their open research and extraordinary work.
