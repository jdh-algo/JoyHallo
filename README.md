# JoyHallo: Digital human model for Mandarin

<br>
<div align='left'>
    <a href='https://jdh-algo.github.io/JoyHallo/#/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://huggingface.co/spaces/jdh-algo/JoyHallo-v1/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
</div>
<br>

## 📖 Introduction

In the field of speech-driven video generation, creating Mandarin videos presents significant challenges. Collecting comprehensive Mandarin datasets is difficult, and Mandarin's complex lip shapes further complicate model training compared to English. Our research involved collecting 29 hours of Mandarin speech video from employees at JD Health International Inc., resulting in the jdh-Hallo dataset. This dataset features a wide range of ages and speaking styles, including both conversational and specialized medical topics. To adapt the JoyHallo model for Mandarin, we utilized the Chinese-wav2vec 2.0 model for audio feature embedding. Additionally, we enhanced the Hierarchical Audio-Driven Visual Synthesis module by integrating a Cross Attention mechanism, which aggregates information from lip, expression, and pose features. This integration not only improves information utilization efficiency but also accelerates inference speed by 14.3%. The moderate coupling of information enables the model to learn relationships between facial features, addressing issues of unnatural appearance. These advancements lead to more precise alignment between audio inputs and visual outputs, enhancing the quality and realism of synthesized videos. It is noteworthy that JoyHallo maintains its strong ability to generate English videos, demonstrating excellent cross-language generation capabilities.

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

For convenience, we have uploaded the model weights to both **Huggingface** and **JD Cloud**.

|  Model  |  Dataset  |                              Huggingface                              |                                       JD Cloud                                       |         Description         |
| :------: | :-------: | :-------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :-------------------------: |
| JoyHallo | jdh-hallo | [JoyHallo](https://huggingface.co/spaces/jdh-algo/JoyHallo-v1/tree/main) | [JoyHallo](https://medicine-ai.s3.cn-north-1.jdcloud-oss.com/JoyHallo/joyhallo/net.pth) | Suitable for JoyHallo model |
| ch-Hallo | jdh-hallo | [ch-Hallo](https://huggingface.co/spaces/jdh-algo/JoyHallo-v1/tree/main) | [ch-Hallo](https://medicine-ai.s3.cn-north-1.jdcloud-oss.com/JoyHallo/ch-hallo/net.pth) |  Suitable for Hallo model  |

### 4. pretrained_models contents

The final `pretrained_models` directory should look like this:

```text
./pretrained_models/
|-- audio_separator/
|   |-- download_checks.json
|   |-- mdx_model_data.json
|   |-- vr_model_data.json
|   `-- Kim_Vocal_2.onnx
|-- ch-hallo/
|   `-- net.pth
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
- Face should be facing forward and occupy 50%-70% of the image area.

**Audio**:

- Audio in `wav` format.
- Mandarin or English, with clear audio and suitable background music.

Notes: These requirements apply to **both training and inference processes**.

## 🚀 Inference

Use the following command to perform inference:

```bash
sh joyhallo-infer.sh
```

Modify the parameters in `configs/inference/inference.yaml` to specify the audio and image files you want to use, as well as switch between models. The inference results will be saved in `opts/joyhallo`. The parameters in `inference.yaml` are explained as follows:

* audio_ckpt_dir: Path to the model weights.
* ref_img_path: Path to the reference images.
* audio_path: Path to the reference audios.
* output_dir: Output directory.
* exp_name: Output file folder name.

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
joyhallo/
|-- videos/
|   |-- 0001.mp4
|   |-- 0002.mp4
|   |-- 0003.mp4
|   `-- 0004.mp4
```

### 2. Use the following command to process the dataset

```bash
python -m scripts.data_preprocess --input_dir joyhallo/videos --step 1
python -m scripts.data_preprocess --input_dir joyhallo/videos --step 2
```

## 💻 Comparison

### 1. Accuracy comparison in Mandarin

|  Model  | Sync-C $\uparrow$ | Sync-D $\downarrow$ | Smooth $\uparrow$ | Subject $\uparrow$ | Background $\uparrow$ |
| :------: | :----------------: | :------------------: | :----------------: | :-----------------: | :--------------------: |
|  Hallo  |       5.7420       |  **13.8140**  |       0.9924       |       0.9855       |    **0.9651**    |
| JoyHallo |  **6.1596**  |       14.2053       |  **0.9925**  |  **0.9864**  |         0.9627         |

Notes: The evaluation metrics used here are from the following repositories, and the results are for reference purposes only:

- Sync-C and Sync-D: [Syncnet](https://github.com/joonson/syncnet_python)
- Smooth, Subject, and Background: [VBench](https://github.com/Vchitect/VBench)

### 2. Inference efficiency comparison

|                              | JoyHallo | Hallo |   Improvement   |
| :---------------------------: | :------: | :----: | :-------------: |
| GPU Memory (512*512, step 40) |  19049m  | 19547m | **2.5%** |
|  Inference Speed (16 frames)  |   24s   |  28s  | **14.3%** |

## 📝 Citations

If you find our work helpful, please consider citing us:

```
@misc{JoyHallo2024,
  title={JoyHallo: Digital human model for Mandarin},
  author={Sheng Shi and Xuyang Cao and Jun Zhao and Guoxin Wang},
  year={2024},
  url={https://huggingface.co/spaces/jdh-algo/JoyHallo-v1}
}
```

## 🤝 Acknowledgments

We would like to thank the contributors to the [Hallo](https://github.com/fudan-generative-vision/hallo), [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), [Chinese-wav2vec2](https://github.com/TencentGameMate/chinese_speech_pretrain), [Syncnet](https://github.com/joonson/syncnet_python), [VBench](https://github.com/Vchitect/VBench), and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) repositories, for their open research and extraordinary work.
