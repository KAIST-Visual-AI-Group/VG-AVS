<div align="center">
<h2>Toward Ambulatory Vision: Learning Visually-Grounded Active View Selection</h2>

[**Juil Koo***](https://63days.github.io) · [**Daehyeon Choi***](https://choidaedae.github.io) · [**Sangwoo Youn***]() · [**Phillip Y. Lee**](https://phillipinseoul.github.io/) · [**Minhyuk Sung**](https://mhsung.github.io) 

(* Equal Contribution)

KAIST

<span style="font-size: 1.5em;"><b>arXiv 2025</b></span>

<a href="https://arxiv.org/abs/2512.13250"><img src='https://img.shields.io/badge/arXiv-AmbulatoryVision-red' alt='Paper PDF'></a>
<a href='https://active-view-selection.github.io'><img src='https://img.shields.io/badge/Project_Page-AmbulatoryVision-green' alt='Project Page'></a>
<a href='https://huggingface.co/daehyeonchoi/VGAVS-model'><img src='https://img.shields.io/badge/🤗%20Hugging%20Face-AVS_Model-yellow' alt='AVS-Model'></a>
<a href='https://huggingface.co/datasets/daehyeonchoi/VGAVS-Dataset'><img src='https://img.shields.io/badge/🤗%20Hugging%20Face-AVS_Dataset-blue' alt='AVS-Dataset'></a>
<img src="./assets/vgavs_teaser.png" alt="VGAVS Teaser" width="100%">

## TL;DR
<i>We introduce Visually Grounded Active View Selection (VG-AVS) Framework, enabling embodied agents to actively adjust their viewpoint for better Visual Question Answering using only current visual cues, achieving state-of-the-art performance on synthetic and real-world benchmarks.</i>

</div>

## Release Checklist
✅ Pretrained (SFT, SFT+GRPO) model checkpoint. **(01.05)**

✅ AVS-ProcTHOR & AVS-HM3D dataset, training/inference/evaluation code. **(01.02)**



## Code
### 1. Environment Setup

We tested our code in CUDA 12.8 with NVIDIA H200 GPUs. However, it might work in different CUDA environment and GPU device. 

#### Conda Environment
Clone this repository:
```
git clone https://github.com/KAIST-Visual-AI-Group/VG-AVS.git
cd VG-AVS
```

```
# initialize virtual environment. we used conda.
conda create --name avs python=3.11 -y 
conda activate avs

# Firstly, install torch fit with your gpu. We used 2.8.0+cu128.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 

# install other libraries.
bash setup.sh 
```

### 2. Download data
#### ProcTHOR
We release the training data (ProcTHOR) and evaluation data (ProcTHOR) in [huggingface](https://huggingface.co/datasets/daehyeonchoi/VGAVS-Dataset), so please download and move these files in your project folder.
```
# move data samples in 'data' folder
mv avs_procthor_train.tar.gz avs_procthor_existence.tar.gz avs_procthor_counting.tar.gz avs_procthor_state.tar.gz ./data/

# extract files from tar file 
tar -xvf avs_procthor_train.tar.gz
tar -xvf avs_procthor_existence.tar.gz
tar -xvf avs_procthor_counting.tar.gz
tar -xvf avs_procthor_state.tar.gz
```

#### HM3D
For the case of HM3D dataset, please download the datasets with this offiical instruction at first. ([Habitat-Matterport3D](https://github.com/matterport/habitat-matterport-3dresearch))
```
# authorize yourself and download 'v0.2/val' splits. 
mv hm3d-val-semantic-configs-v0.2.tar hm3d-val-semantic-annots-v0.2.tar hm3d-val-habitat-v0.2.tar hm3d-val-glb-v0.2.tar ./data/hm3d/val/

# extract files from tar file 
cd ./data/hm3d/val 
tar -xvf hm3d-val-semantic-configs-v0.2.tar
tar -xvf hm3d-val-semantic-annots-v0.2.tar
tar -xvf hm3d-val-habitat-v0.2.tar
tar -xvf hm3d-val-glb-v0.2.tar
```

Then additionally download data snapshot from [HF repository](https://huggingface.co/datasets/daehyeonchoi/VGAVS-Dataset), then move it into "data" folder. 
```
# move data samples in 'data' folder
mv avs_hm3d.tar.gz ./data/
```

Finally, the folder structure is like below:
```
data/
├── hm3d/
│   └── val/
│       ├── 00800-TEEsavR23oF/
│       └── 00YYY-zzzzzzzzzzz/
├── avs_procthor_train/
├── avs_procthor_existence/
├── avs_procthor_counting/
├── avs_procthor_state/
├── avs_hm3d/
├── avs_hm3d_overall.jsonl
└──...
```

### 3. Setting Simulation Environment
#### Habitat-Sim 
```
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# building habitat-sim from source. it would take minutes.
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install . -v
```

#### Sanity Check 
Our framework utilizes two different types of simulation environment (AI2-THOR and HM3D), so before running the code, please check each environment works properly in your setting. 

Please follow [notebook/environment_check.ipynb](https://github.com/KAIST-Visual-AI-Group/VG-AVS/blob/master/notebook/environment_check.ipynb).


### 4. Download Pretrained model
We provide our pretrained model checkpoint in [HF repository](https://huggingface.co/daehyeonchoi/VGAVS-model/tree/main). Download this model into your directory and set `MODEL_PATH` variable in your running script. 


### 5. Run
Before running training or evaluation scripts, you need to configure the following paths and API keys.

#### Configuration

##### Required Paths
| Variable | Description | Example |
|----------|-------------|---------|
| `PROJECT_ROOT` | Root directory of the project | `/home/user/VG-AVS` |
| `DATA_JSONL` | Path to training/evaluation JSONL file | `${PROJECT_ROOT}/data/avs_procthor_train.jsonl` |
| `IMG_ROOT` | Root directory containing images | `${PROJECT_ROOT}/data` |
| `MODEL_PATH` | Path to trained model (for evaluation) | `${PROJECT_ROOT}/src/open-r1-multimodal/output/grpo-procthor` |

##### API Keys (for Evaluation)
The evaluation scripts use LLM APIs for the verifier model. Set these environment variables:
```bash
export GEMINI_API_KEY="your_gemini_api_key"   # Required for Gemini verifier
export OPENAI_API_KEY="your_openai_api_key"   # Optional, for GPT verifier
```

### Tutorial    
You can easliy test our framework in ProcTHOR environment. 
```bash
bash src/open-r1-multimodal/run_scripts/test_procthor_single_sample.sh
```

### Training 
With eight H200 GPUs, SFT training takes about 1 hour, and GRPO training takes about 20 hours.

**1. SFT Training (Supervised Fine-tuning):**
```bash
bash src/open-r1-multimodal/run_scripts/run_sft_procthor_active_qa.sh
```

**2. GRPO Training (Reinforcement Learning):**

Please note that you should set '$MODEL_PATH' variable as SFT pre-trained model's.
```bash
# Set required paths
bash src/open-r1-multimodal/run_scripts/run_grpo_procthor_active_qa.sh 
```

### Evaluation   
**ProcTHOR Evaluation:**
```bash
# Set required paths and API keys
bash src/open-r1-multimodal/run_scripts/test_procthor_action_accuracy.sh
```

**HM3D Evaluation:**
```bash
bash src/open-r1-multimodal/run_scripts/test_hm3d_action_accuracy.sh
```

## Contact
If you run into any issues, please open a GitHub issue or email us at daehyeonchoi@kaist.ac.kr.

## Acknowledgement
Our implementation is built upon amazing projects including [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [AI2-THOR](https://ai2thor.allenai.org/), [Habitat-Sim](https://github.com/facebookresearch/habitat-sim). We greatly thank all authors and contributors for open-sourcing their code and model checkpoints.
