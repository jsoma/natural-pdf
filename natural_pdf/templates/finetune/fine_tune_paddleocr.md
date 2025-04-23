# Fine-tuning a PaddleOCR Recognition Model with Your Exported Data

This notebook guides you through fine-tuning a PaddleOCR text recognition model using the dataset you exported from `natural-pdf`.

**Goal:** Improve OCR accuracy on your specific documents (e.g., handle unique fonts, languages, or styles).

**Environment:** This notebook is designed to run on Google Colab with a GPU runtime.

## 1. Setup Environment

First, let's install the necessary libraries: PaddlePaddle (GPU version) and PaddleOCR.

```python
# Check GPU availability (Recommended: Select Runtime -> Change runtime type -> GPU)
!nvidia-smi
```

```python
# Install PaddlePaddle GPU version
# Visit https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/linux-pip_en.html
# for the correct command based on your CUDA version.
# CUDA versions are backwards-compatible, so you don't have to worry about
# I mostly just go to https://www.paddlepaddle.org.cn/packages/stable/
# and see what the most recent version that kinda matches mine is 
# e.g. colab is CUDA 12.4, there's a "123" directory, I use that.
!pip install --quiet paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# Install PaddleOCR and its dependencies
!pip install --quiet paddleocr
```

```python
# Verify PaddlePaddle installation and GPU detection
import paddle
print("PaddlePaddle version:", paddle.__version__)
print("GPU available:", paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print("Number of GPUs:", paddle.device.cuda.device_count())
    print("Current GPU:", paddle.device.get_device())
```

## 2. Upload and Unzip Your Dataset

Use the file browser on the left panel of Colab to upload the `.zip` file you created using the `PaddleOCRRecognitionExporter`. Then, unzip it.

```python
# Replace 'your_exported_data.zip' with the actual filename you uploaded
!unzip -q your_exported_data.zip -d finetune_data

# List the contents to verify
!ls finetune_data
```

You should see `images/`, `dict.txt`, `train.txt`, and `val.txt` (or `label.txt`) inside the `finetune_data` directory.

## 3. Prepare Training Configuration

PaddleOCR uses YAML files for configuration. We'll create one based on a standard recognition config, modified for fine-tuning with our dataset.

**Key Parameters to potentially adjust:**

*   `Global.pretrained_model`: Path or URL to the pre-trained model you want to fine-tune. Using a model pre-trained on a large dataset (like English or multilingual) is crucial. See PaddleOCR Model List for options.
*   `Global.save_model_dir`: Where to save checkpoints during training.
*   `Global.epoch_num`: Number of training epochs. Start small (e.g., 10-50) for fine-tuning and increase if needed based on validation performance.
*   `Optimizer.lr.learning_rate`: Learning rate. Fine-tuning often requires a smaller learning rate than training from scratch (e.g., 1e-4, 5e-5).
*   `Train.dataset.data_dir`: Path to the directory containing the `images/` folder.
*   `Train.dataset.label_file_list`: Path to your `train.txt`.
*   `Train.loader.batch_size_per_card`: Batch size. Adjust based on GPU memory.
*   `Eval.dataset.data_dir`: Path to the directory containing the `images/` folder.
*   `Eval.dataset.label_file_list`: Path to your `val.txt`.
*   `Eval.loader.batch_size_per_card`: Batch size for evaluation.
*   `Architecture...`: Ensure the architecture matches the `pretrained_model`.
*   `Loss...`: Ensure the loss function matches the `pretrained_model`.

```python
# Choose a pre-trained model (check PaddleOCR docs for latest/best models)
#PRETRAINED_MODEL_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/latin_PP-OCRv4_rec_train.tar"
PRETRAINED_MODEL_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar"

# Download and extract the pre-trained model
!wget -q {PRETRAINED_MODEL_URL} -O pretrained_model.tar
!tar -xf pretrained_model.tar

# Find the actual directory name (it might vary slightly)
PRETRAINED_MODEL_DIR = !find . -maxdepth 1 -type d -name '*_rec*' | head -n 1
PRETRAINED_MODEL_DIR = PRETRAINED_MODEL_DIR[0]
print(f"Using Pretrained Model Dir: {PRETRAINED_MODEL_DIR}")
```

```python
num_classes = len([line for line in open("finetune_data/dict.txt", encoding="utf-8")])
num_classes
```

```python
lengths = []
with open("finetune_data/train.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            lengths.append(len(parts[1]))

# Basic stats
print("Max length:", max(lengths))
print("95th percentile:", sorted(lengths)[int(len(lengths) * 0.95)])
print("99th percentile:", sorted(lengths)[int(len(lengths) * 0.99)])
print("99.9th percentile:", sorted(lengths)[int(len(lengths) * 0.999)])

buffered_max_length = int(sorted(lengths)[int(len(lengths) * 0.999)] * 1.1)
buffered_max_length
```

```python
import shutil
from datetime import datetime

MAX_ALLOWED = buffered_max_length
removed = 0
cleaned_lines = []

with open("finetune_data/train.txt", encoding="utf-8") as f:
  original_lines = f.readlines()

for i, line in enumerate(original_lines):
  parts = line.strip().split(maxsplit=1)
  if len(parts) == 2 and len(parts[1]) > MAX_ALLOWED:
    removed += 1
    print(f"⚠️ Line {i} exceeds max_text_length: {len(parts[1])} chars: {parts[1]}")
  else:
    cleaned_lines.append(line)

if removed > 0:
  print(f"Removed {removed} of {len(original_lines)}. Backing up original, writing clean copy.")
  shutil.copy("finetune_data/train.txt", "finetune_data/train_backup.txt")

  with open("finetune_data/train.txt", "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)
else:
  print("Found 0 long lines")
```

```python
# Training configuration for PaddleOCR Recognition Fine-tuning
yaml_content = f"""
Global:
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 50
  save_model_dir: ./output/finetune_rec/
  save_epoch_step: 5
  eval_batch_step: [0, 200]
  cal_metric_during_train: true
  pretrained_model: {PRETRAINED_MODEL_DIR}/best_accuracy
  checkpoints: null
  save_inference_dir: null
  use_visualdl: false
  infer_img: doc/imgs_words/en/word_1.png
  character_dict_path: finetune_data/dict.txt
  max_text_length: {buffered_max_length}
  infer_mode: false
  use_space_char: true
  save_res_path: ./output/rec/predicts_rec.txt

Optimizer:
  name: AdamW
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.00005   # 5e-5 for batch_size=64
    warmup_epoch: 3
  regularizer:
    name: L2
    factor: 0.00005

Architecture:
  model_type: rec
  algorithm: SVTR
  Transform: null
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: svtr
    dims: 64
    depth: 2
    hidden_dims: 120
    use_guide: False
  Head:
    name: CTCHead
    fc_decay: 0.00001
    out_channels: {num_classes + 1}

Loss:
  name: CTCLoss

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./finetune_data/
    label_file_list: ["./finetune_data/train.txt"]
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - CTCLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./finetune_data/
    label_file_list: ["./finetune_data/val.txt"]
    ratio_list: [1.0]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - CTCLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
"""

with open("finetune_rec.yml", "w", encoding="utf-8") as fp:
  fp.write(yaml_content)
```

## 4. Clone PaddleOCR Repository and Start Training

We need the PaddleOCR repository for its training scripts.

```python
# Clone the PaddleOCR repository (using main branch)
!git clone https://github.com/PaddlePaddle/PaddleOCR.git --depth 1 paddleocr_repo
```

```python
# Start training!
# -c points to our config file
# -o Override specific config options if needed (e.g., Global.epoch_num=10)
!python paddleocr_repo/tools/train.py -c ../finetune_rec.yml
```

Training will begin, printing logs and saving checkpoints to the directory specified in `Global.save_model_dir` (`./output/finetune_rec/` in the example). Monitor the accuracy (`acc`) and loss on the training and validation sets. Stop training early if validation accuracy plateaus or starts to decrease.

## 5. Export Best Model for Inference

Once training is complete, find the best checkpoint (usually named `best_accuracy.pdparams`) in the output directory and convert it into an inference model.

```python
# Find the best model checkpoint
BEST_MODEL_PATH = "output/finetune_rec/best_accuracy" # Path relative to paddleocr_repo dir

# Export the model for inference
!python paddleocr_repo/tools/export_model.py \
    -c finetune_rec.yml \
    -o Global.pretrained_model="{BEST_MODEL_PATH}" \
    Global.save_inference_dir="inference_model"
```

This will create an `inference_model` directory containing `inference.pdmodel`, `inference.pdiparams`, and potentially other files needed for deployment.

## 6. Test Inference (Optional)

You can use the exported inference model to predict text on new images.

```python
from paddleocr import PaddleOCR
from IPython.display import Image, display
import random

ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    rec_model_dir='inference_model',
    rec_algorithm='SVTR_LCNet',
    rec_image_shape='3,48,320',
    rec_char_dict_path='finetune_data/dict.txt',
    use_gpu=True
)

# Pick one random image from val.txt
with open("finetune_data/val.txt", encoding="utf-8") as f:
    line = random.choice([l.strip() for l in f if l.strip()])
img_path, ground_truth = line.split(maxsplit=1)

# Run inference
result = ocr.ocr(img_path, det=False)
prediction = result[0][0][1]['text'] if result else '[No result]'

# Display
display(Image(filename=img_path))
print(f"GT:  {ground_truth}")
print(f"Pred: {prediction}")
```

Compare the predicted text with the ground truth in your label file.

---

You now have a fine-tuned PaddleOCR recognition model tailored to your data! You can download the `inference_model` directory from Colab for use in your applications. 