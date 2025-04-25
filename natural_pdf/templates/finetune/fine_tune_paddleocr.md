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

Depending on how you train, you may or may not need to know how many characters are in your alphabet.

```python
num_classes = len([line for line in open("finetune_data/dict.txt", encoding="utf-8")])
num_classes
```

You need to set a maximum length for your pieces of text – if you plan ahead you can cut them up in other ways, but the easiest route is to pick the 99th or 99.9th percentile to avoid outliers. In my first test the 95th percentile was 17, 99.9th was 41, and absolute max was 138! It would have wasted a lot of memory and energy if we'd centered everything around 138-character words.

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
MAX_ALLOWED = buffered_max_length
MIN_ALLOWED = 3
removed = 0
cleaned_lines = []

with open("finetune_data/train.txt", encoding="utf-8") as f:
  original_lines = f.readlines()

for i, line in enumerate(original_lines):
  parts = line.strip().split(maxsplit=1)
  if len(parts) == 2 and len(parts[1]) > MAX_ALLOWED:
    removed += 1
    print(f"⚠️ Line {i} exceeds max_text_length: {len(parts[1])} chars: {parts[1]}")
  elif len(parts[1]) < MIN_ALLOWED:
    removed += 1
    print(f"⚠️ Line {i} under min_text_length: {len(parts[1])} chars: {parts[1]}")
  elif "Sorry, I can't" in parts[1]:
    removed += 1
    print(f"⚠️ Line {i} was not OCR'd correctly")
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

You'll also notice it catches a lot of "Sorry, I can't process the image. Please upload the image again." and the like.

**And now it's configuration time!** We ignore almost all of the [suggestions from PaddleOCR's documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/model_train/finetune.html) because for some reason they get me ~40% while copying the [PPOCRv3 yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/configs/rec/PP-OCRv3/multi_language/latin_PP-OCRv3_rec.yml) gets me up to ~80%.

This creates a `finetune_rec.yml` file that controls how the training process will go.

```python
yaml_content = f"""
Global:
  use_gpu: true
  epoch_num: 120
  log_smooth_window: 20
  print_batch_step: 50
  save_model_dir: ./output/finetune_rec/
  save_epoch_step: 5
  eval_batch_step: [0, 200]  # Evaluate every 200 steps
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
    learning_rate: 0.00005
    warmup_epoch: 3
  regularizer:
    name: L2
    factor: 0.00005

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform: null
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
    last_pool_kernel_size: [2, 2]
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: svtr
            dims: 64
            depth: 2
            hidden_dims: 120
            use_guide: True
          Head:
            fc_decay: 0.00001
      - SARHead:
          enc_dim: 512
          max_text_length: {buffered_max_length}

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - SARLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
  ignore_space: false

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./finetune_data/
    label_file_list: ["./finetune_data/train.txt"]
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - MultiLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ["image", "label_ctc", "label_sar", "length", "valid_ratio"]
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
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - MultiLabelEncode:
      - SVTRRecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ["image", "label_ctc", "label_sar", "length", "valid_ratio"]
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

We need the PaddleOCR repository for its training scripts. Once we have it we'll point it at our `finetune_rec.yml` and set it in action.

```python
# Clone the PaddleOCR repository (using main branch)
!git clone https://github.com/PaddlePaddle/PaddleOCR.git --depth 1 paddleocr_repo
```

```python
# Remove any existing trained model
!rm -rf output

# Start training!
# -c points to our config file
# -o Override specific config options if needed (e.g., Global.epoch_num=10)
!python paddleocr_repo/tools/train.py -c finetune_rec.yml
```

Training will begin, printing logs and saving checkpoints to the directory specified in `Global.save_model_dir` (`./output/finetune_rec/` in the example). Monitor the accuracy (`acc`) and loss on the training and validation sets. You can stop training early if validation accuracy plateaus or starts to decrease.

## 5. Export Best Model for Inference

Once training is complete, find the best checkpoint (usually named `best_accuracy.pdparams`) in the output directory and convert it into an inference model. The line below should automatically find the best model.

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
img_path = "finetune_data/" + img_path

# Run inference
result = ocr.ocr(img_path, det=False)
prediction = result[0][0][1] if result else '[No result]'

# Display
display(Image(filename=img_path))
print(f"GT:  {ground_truth}")
print(f"Pred: {prediction}")
```

Compare the predicted text with the ground truth in your label file.

## 7. Package and Distribute Your Model

Once you have successfully fine-tuned and tested your model, you'll want to package it for easy distribution and use. A properly packaged model should include all necessary files to use it with Natural PDF:

````python
import shutil
import os

# Create a distribution directory
dist_dir = "my_paddleocr_model_distribution"
os.makedirs(dist_dir, exist_ok=True)

# Copy the inference model
shutil.copytree("inference_model", os.path.join(dist_dir, "inference_model"))

# Copy the dictionary file (critical for text recognition)
shutil.copy("finetune_data/dict.txt", os.path.join(dist_dir, "dict.txt"))

# Create a simple README
with open(os.path.join(dist_dir, "README.md"), "w") as f:
    f.write("""# Custom PaddleOCR Model

## Model Information
- Trained for: [describe your document type/language]
- Base model: [e.g., "PaddleOCR v3 Latin"]
- Training date: [date]
- Epochs trained: [number of epochs]
- Final accuracy: [accuracy percentage]

## Usage with Natural PDF

    from natural_pdf import PDF
    from natural_pdf.ocr import PaddleOCROptions

    # Configure OCR with this model
    paddle_opts = PaddleOCROptions(
        rec_model_dir="path/to/inference_model",
        rec_char_dict_path="path/to/dict.txt",
    )

    # Use in your PDF processing
    pdf = PDF("your-document.pdf")
    page = pdf.pages[0]
    ocr_elements = page.apply_ocr(engine='paddle', options=paddle_opts)
""")

# Zip everything up

shutil.make_archive(dist_dir, 'zip', dist_dir)
print(f"Model distribution package created: {dist_dir}.zip")
````

### Essential Components

Your distribution package must include:

1. **Inference Model Directory**: Contains the trained model files (`inference.pdmodel`, `inference.pdiparams`, etc.)
2. **Character Dictionary**: The `dict.txt` file used during training that maps character IDs to actual characters
3. **Documentation**: A README with usage instructions and model information

### Usage Notes

When sharing your model with others, advise them to:

1. Extract all files while maintaining the directory structure
2. Use the `PaddleOCROptions` class to configure Natural PDF with the model paths
3. Understand model limitations (specific languages, document types, etc.)

You now have a fine-tuned PaddleOCR recognition model tailored to your data! The model can be distributed and used to improve OCR accuracy on similar documents in your application.

--- 