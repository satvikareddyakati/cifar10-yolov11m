# 🌟 YOLOv11m-CLS on CIFAR-10

### **Blur-Robust Image Classification Project**

A compact, reproducible project to train, evaluate, and run inference using **Ultralytics YOLOv11-CLS** on **CIFAR-10**, enhanced with strong regularization for **blur robustness**.

---

## 🚀 Features

* **Model:** YOLOv11m-CLS (ImageNet-pretrained)
* **Dataset:** CIFAR-10, upsampled from **32×32 → 160×160**
* **Device:** Auto-selects: Apple MPS → CUDA → CPU
* **Robustness Techniques:**

  * Cosine LR
  * Label smoothing
  * Dropout
  * Mixup
  * CutMix
  * Random erasing
* **Includes:** Blur-robustness demo + complete evaluation suite

---

## 📁 Repository Structure

```
.
├── yolo11_cifar10_robust.py      # Train + blur demo
├── yolo11_eval.py                # Train/Val/Test accuracy summary
├── yolo11_infer.py               # Single-image inference
├── test.py                       # Test evaluation + confusion matrices
└── runs/classify/...             # All Ultralytics outputs
```

---

## 🛠️ Environment Setup

Tested with:

* Python **3.13.7**
* PyTorch **2.9.0**
* Ultralytics **8.3.226**
* Hardware: **Apple M4 (MPS)**

### Install (Minimal)

```bash
python -m venv yolov11-env
source yolov11-env/bin/activate
pip install -U ultralytics
```

Ultralytics installs a compatible PyTorch automatically.

---

## 📦 Dataset

* Training automatically downloads CIFAR-10 using:

  ```python
  ```

data="cifar10"

```
- Default dataset root for evaluation scripts:

```

/Users/satvikaakati/datasets/cifar10

````

Update paths in scripts if stored elsewhere.

---

## 🏋️ Training

Run YOLOv11m-CLS with imgsz=160 and strong regularization:

```bash
python /Users/satvikaakati/Desktop/yolo_project/yolo11_cifar10_robust.py
````

Results saved to:

```
runs/classify/train*/
```

Includes:

* `args.yaml`
* `results.csv`, `results.png`
* `confusion_matrix.png`, `confusion_matrix_normalized.png`
* `weights/best.pt`

---

## 📊 Evaluation

### ✅ One-Shot Evaluation (Train/Val/Test)

```bash
python /Users/satvikaakati/Desktop/yolo_project/yolo11_eval.py
```

Example output:

```json
{
  "train_top1": 0.9816,
  "train_top5": 0.9998,
  "val_top1_best": 0.9496,
  "val_top5_best": 0.9993,
  "test_top1": 0.9497,
  "test_top5": 0.9993
}
```

---

### 🎯 Test-Only Evaluation (+ Confusion Matrices)

```bash
python /Users/satvikaakati/Desktop/yolo_project/test.py \
  --model /Users/satvikaakati/Desktop/yolo_project/runs/classify/train2/weights/best.pt \
  --data /Users/satvikaakati/datasets/cifar10 \
  --imgsz 160
```

Outputs:

* `test_metrics.json`
* `confusion_matrix.png`
* `confusion_matrix_normalized.png`

---

## 🖼️ Inference Example

```bash
python /Users/satvikaakati/Desktop/yolo_project/yolo11_infer.py
```

Modify the `source` parameter inside the script to test custom images.

---

## 📈 Results Summary (Best Run: `train2`)

| Metric               | Score  |
| -------------------- | ------ |
| **Train Top-1**      | 0.9816 |
| **Train Top-5**      | 0.9998 |
| **Validation Top-1** | 0.9496 |
| **Validation Top-5** | 0.9993 |
| **Test Top-1**       | 0.9497 |
| **Test Top-5**       | 0.9993 |

✨ *Validation and Test match closely — demonstrating strong generalization with no overfitting.*

---

## 📤 Suggested Submission Package

* **Source files:**

  * `yolo11_cifar10_robust.py`
  * `yolo11_eval.py`
  * `yolo11_infer.py`
  * `test.py`
* **Training artifacts (train2):**

  * `results.csv`, `results.png`
  * `args.yaml`
  * `weights/best.pt`
  * Confusion matrices
* **Test outputs:**

  * `test_metrics.json`
  * `confusion_matrix.png`
  * `confusion_matrix_normalized.png`
  * Optional prediction samples

---

## 🔁 Reproducibility Tips

* Reduce `batch=` in the training script if you hit memory issues.
* Force device manually: `'cpu'`, `'mps'`, or `'0'` (CUDA).
* Keep `imgsz=160` for metric consistency.

---

## 🎉 Acknowledgements

This project uses **Ultralytics YOLOv11-CLS**, open-source under the AGPL license.
