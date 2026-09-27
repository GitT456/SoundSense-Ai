# 🎧 SoundSense AI
### Intelligent Multi-Label Environmental Sound Recognition with Deep Learning

<p align="center">
  <strong>Can AI understand what it hears?</strong><br/>
  SoundSense AI transforms raw audio into meaningful sound events using a hybrid
  <strong>CNN + BiLSTM + Attention</strong> deep learning architecture.
</p>

<p align="center">
  <a href="https://www.kaggle.com/code/tahazarei/sound-sense">Kaggle Notebook</a>
  ·
  <a href="https://github.com/GitT456">Taha Zarei</a>
  ·
  <a href="https://github.com/maedehbabaei-dev">Maedeh Babaei</a>
</p>

---

## 🌌 Overview

**SoundSense AI** is an end-to-end deep learning project for **multi-label environmental sound event recognition**.

Unlike a traditional classifier that assumes one input belongs to exactly one class, SoundSense is designed for real-world audio where **multiple sound events can occur at the same time**.

For example, a single recording might contain:

> 🗣️ Speech + 👏 Applause + 🎵 Music

Instead of forcing the recording into one category, SoundSense can assign **multiple sound labels with independent confidence scores**.

The complete pipeline is:

```text
🎙️ Audio
   │
   ▼
🔧 Audio Preprocessing
   │
   ▼
🎼 Mel Spectrogram
   │
   ▼
🧠 CNN
   │
   ▼
⏱️ BiLSTM
   │
   ▼
🎯 Attention Pooling
   │
   ▼
📊 Multi-Label Classification
   │
   ▼
🔢 Sigmoid + Class Thresholds
   │
   ▼
🏷️ Sound Events + Confidence Scores
   │
   ▼
⚡ FastAPI
   │
   ▼
🌐 Web Interface
```

The project is developed as a **Deep Learning course project** with an emphasis on understanding the complete machine-learning lifecycle:

**Dataset → Preprocessing → Model Design → Training → Evaluation → Model Export → API → Web Application**

---

# ✨ What Makes SoundSense Different?

Many beginner audio classification projects follow a simple:

```text
Audio → Spectrogram → CNN → One Label
```

SoundSense is designed around a more realistic problem:

```text
Audio → Spectrogram → CNN → Temporal Modeling → Attention → Multiple Labels
```

The architecture combines three complementary ideas:

| Component | Main Role |
|---|---|
| 🎼 Mel Spectrogram | Converts audio into a representation that exposes useful frequency patterns |
| 👁️ CNN | Learns local time-frequency patterns |
| ⏱️ BiLSTM | Models temporal relationships across the audio sequence |
| 🎯 Attention | Learns which parts of the sequence are more informative |
| 🔢 Sigmoid | Produces an independent probability for every sound class |
| ⚖️ Weighted BCE | Addresses the highly imbalanced multi-label learning problem |
| 🧩 SpecAugment | Improves robustness through spectrogram-level augmentation |
| 🎚️ Per-class thresholds | Converts probabilities into final labels more appropriately than a fixed threshold |

---

# 🎯 Project Goal

The goal is to build a model capable of answering:

> **"What sounds are present in this recording?"**

rather than:

> **"Which single category does this recording belong to?"**

This distinction is important because environmental audio is naturally complex.

A recording can contain:

- human speech
- animals
- vehicles
- tools
- household sounds
- musical instruments
- natural sounds
- mechanical sounds
- and several of these at the same time

Therefore, SoundSense treats the task as **multi-label sound event classification**.

---

# 🗂️ Dataset — FSD50K

SoundSense uses **FSD50K (Freesound Dataset 50K)**, an open, human-labeled dataset of sound events created from Freesound.

FSD50K contains:

| Property | Value |
|---|---:|
| Audio clips | **51,197** |
| Total duration | **108.3 hours** |
| Sound classes | **200** |
| Development clips | **40,966** |
| Evaluation clips | **10,231** |
| Average labels / clip | **1.22** |
| Clip duration | **0.3–30 seconds** |
| Source | Freesound |
| Ontology | AudioSet subset |
| Annotation | Human-labeled |

The official dataset is organized into development and evaluation sets, with the evaluation split designed separately from development data.

### Official Resources

- **FSD50K official website:** https://fsannotator.upf.edu/fsd/release/FSD50K/
- **FSD50K on Zenodo:** https://zenodo.org/records/4060432
- **Original paper:** https://doi.org/10.1109/TASLP.2021.3133208
- **Project Kaggle notebook:** https://www.kaggle.com/code/tahazarei/sound-sense

---

# 🧠 Why FSD50K?

FSD50K was selected because it represents a substantially more realistic sound-recognition problem than a simple single-label audio dataset.

### 1. Multi-label audio

One recording may contain multiple sound events.

### 2. Large vocabulary

The dataset contains **200 classes**, making the classification problem significantly more challenging.

### 3. Variable-duration audio

Clips range from approximately **0.3 to 30 seconds**.

### 4. Class imbalance

Some sound events appear much more frequently than others.

### 5. Human annotations

The dataset is manually labeled, making it suitable for supervised sound-event recognition research.

### 6. Hierarchical sound ontology

The 200 classes are organized using a subset of the **AudioSet ontology**, including categories such as human sounds, sounds of things, animals, natural sounds, and music.

---

# 🔬 The Core Problem

A normal multi-class classifier might produce:

```text
Dog → 92%
Cat → 5%
Bird → 3%
```

and select only one class.

SoundSense instead produces independent probabilities:

```text
Dog      → 92%
Speech   → 81%
Vehicle  → 67%
Music    → 12%
Applause → 4%
```

Then class-specific decision thresholds are applied to determine which events are considered present.

This is why the final layer uses:

```text
Sigmoid
```

instead of:

```text
Softmax
```

### Softmax

Softmax assumes classes compete with one another.

```text
P(class_1) + P(class_2) + ... = 1
```

That is suitable for many single-label classification problems.

### Sigmoid

Sigmoid treats each class independently:

```text
Speech   → independent probability
Applause → independent probability
Music    → independent probability
Dog      → independent probability
```

This is much more appropriate for multi-label sound event recognition.

---

# 🎼 From Audio to Mel Spectrogram

A neural network does not directly understand an audio waveform as easily as humans do.

SoundSense therefore transforms the waveform into a **Mel Spectrogram**.

Conceptually:

```text
Raw Waveform
     │
     ▼
Short-Time Fourier Transform
     │
     ▼
Frequency Representation
     │
     ▼
Mel Frequency Mapping
     │
     ▼
Mel Spectrogram
```

A Mel Spectrogram represents:

- **X-axis:** time
- **Y-axis:** frequency on the Mel scale
- **Intensity:** energy

This turns an audio signal into a time-frequency representation that can be processed similarly to an image.

That allows CNNs to detect local patterns in the spectrogram.

---

# 🏗️ Model Architecture

SoundSense uses a hybrid architecture:

```text
                    AUDIO
                      │
                      ▼
              Mel Spectrogram
                      │
                      ▼
              ┌───────────────┐
              │      CNN      │
              │               │
              │  64 channels  │
              │ 128 channels  │
              │ 256 channels  │
              └───────┬───────┘
                      │
                      ▼
             Feature Sequence
                      │
                      ▼
              ┌───────────────┐
              │    BiLSTM     │
              │   2 layers    │
              │ hidden = 128  │
              └───────┬───────┘
                      │
                      ▼
            Attention Pooling
                      │
                      ▼
                  Dropout
                      │
                      ▼
             Fully Connected
                      │
                      ▼
                  Sigmoid
                      │
                      ▼
             200 Class Scores
```

---

# 👁️ CNN — Learning Local Audio Patterns

The CNN operates on the Mel Spectrogram.

It learns local patterns such as:

- frequency structures
- short acoustic events
- harmonics
- transient patterns
- time-frequency textures

The current architecture uses three main convolutional stages:

```text
1 → 64 → 128 → 256
```

with:

- Convolution
- Batch Normalization
- ReLU
- Max Pooling

The CNN acts as the **local feature extractor**.

---

# ⏱️ BiLSTM — Understanding Temporal Structure

Sound is not only about **what** happens.

It is also about **when** it happens.

For example:

```text
Door opens
     ↓
Footsteps
     ↓
Speech
     ↓
Door closes
```

A CNN can detect local patterns, but temporal relationships are also important.

The extracted CNN features are therefore converted into a sequence and processed by a **Bidirectional LSTM**.

The BiLSTM reads the sequence in both directions:

```text
Forward:
t1 → t2 → t3 → t4

Backward:
t4 → t3 → t2 → t1
```

This allows the model to use information from both temporal directions.

---

# 🎯 Attention Pooling

Not every moment of an audio recording is equally informative.

For example, in a 10-second clip:

```text
0s ─────────────── 10s
     ↑
     important event
```

Attention learns to assign larger weights to informative parts of the sequence.

Conceptually:

```text
CNN Features
     │
     ▼
BiLSTM Sequence
     │
     ▼
Attention Scores
     │
     ▼
Weighted Representation
     │
     ▼
Classifier
```

This allows the model to focus more strongly on relevant temporal regions instead of treating every time step equally.

---

# ⚙️ Training Strategy

The project includes several techniques designed around the actual difficulties of the dataset.

## Weighted Binary Cross Entropy

FSD50K is highly imbalanced.

If some classes appear much more frequently than others, a model can become biased toward common labels.

SoundSense therefore uses:

```text
BCEWithLogitsLoss
+
Class-specific positive weights
```

This gives rarer positive classes more influence during training.

---

# 🎛️ SpecAugment

SoundSense also uses spectrogram-level augmentation.

Instead of changing the raw waveform directly, parts of the spectrogram can be masked.

Conceptually:

```text
Original Spectrogram
        ↓
Time Masking
        +
Frequency Masking
        ↓
Augmented Spectrogram
```

The objective is to reduce overfitting and encourage the network to learn more robust acoustic representations.

---

# 📏 Input Normalization

The audio is converted into a consistent representation before entering the model.

The current preprocessing pipeline includes:

- mono conversion
- resampling
- fixed-duration handling
- Mel Spectrogram extraction
- conversion to decibel scale
- per-sample normalization

This is especially important because FSD50K contains audio with different durations and recording conditions.

---

# ⏱️ Variable-Length Audio

FSD50K contains clips with different durations.

Neural networks generally require compatible tensor shapes within a batch.

SoundSense therefore uses a fixed input duration and:

- crops longer recordings
- pads shorter recordings
- uses sliding-window inference for longer recordings when appropriate

This creates a consistent model input while preserving the ability to process longer audio during inference.

---

# 📊 Evaluation Metrics

Accuracy is not the main metric for this problem.

Why?

Because this is:

- multi-label
- imbalanced
- 200-class
- sound-event recognition

The project therefore considers metrics such as:

### Micro Precision

Measures precision over all label decisions collectively.

### Micro Recall

Measures how many of the actual positive events were detected.

### Micro F1

Balances precision and recall.

### Macro F1

Computes F1 across classes and gives each class equal importance.

### Mean Average Precision (mAP)

mAP is particularly important for comparing multi-label sound-event recognition systems.

---

# 📈 Current Experimental Result

The current reported experiment produced:

| Metric | Result |
|---|---:|
| Micro F1 | **0.4192** |
| Macro F1 | **0.3220** |
| Micro Precision | **0.3033** |
| Micro Recall | **0.6787** |
| mAP | **0.3633** |

### Interpreting the result

The current model has substantially higher recall than precision:

```text
Recall    ≈ 0.679
Precision ≈ 0.303
```

This indicates that the model is relatively aggressive when predicting positive sound events: it detects many real events, but also produces a significant number of false positives.

This observation is useful because it directly motivates further work on:

- decision thresholds
- calibration
- class imbalance
- false-positive reduction
- evaluation on the official evaluation split

> **Important:** the reported `mAP = 0.3633` is the result of the current experiment. It should only be compared numerically with the FSD50K paper's official evaluation score if the same evaluation protocol and split are used.

---

# 📚 Comparison With the Original FSD50K Paper

The original FSD50K paper reported the following baseline results on its official evaluation protocol:

| Architecture | Parameters | mAP |
|---|---:|---:|
| CRNN | 0.96M | 0.417 |
| VGG-like | 0.27M | **0.434** |
| ResNet-18 | 11.3M | 0.373 |
| DenseNet-121 | 12.5M | 0.425 |

Source: Fonseca et al., *FSD50K: An Open Dataset of Human-Labeled Sound Events*, IEEE/ACM TASLP, 2022.

The **0.434 VGG-like score is the original paper baseline**, not a claim of current state of the art.

Later work has reported higher results on FSD50K, so the original paper baseline should be understood as a historical benchmark rather than the ceiling of the task.

---

# 🧪 What Is Our Contribution?

SoundSense is not presented as a claim of inventing a new state-of-the-art architecture.

The project contribution is instead the construction of a complete, practical system around the sound-event recognition problem.

### Model-level contribution

The project combines:

```text
CNN
 +
BiLSTM
 +
Attention
 +
Weighted BCE
 +
SpecAugment
 +
Per-class thresholding
```

into one end-to-end pipeline.

### Engineering contribution

The trained model is intended to move beyond a notebook into an actual application:

```text
Trained Model
      ↓
Model Export
      ↓
FastAPI
      ↓
Prediction Endpoint
      ↓
Web Interface
```

This makes the project an **end-to-end Deep Learning system**, rather than only a training experiment.

---

# 🔬 Research vs. Project Innovation

It is important to distinguish these two concepts.

### Research novelty

A genuinely new architecture, loss function, algorithm, or scientific method that advances the field.

### Project innovation

A meaningful combination and implementation of established techniques to solve a complete problem effectively.

SoundSense focuses on the second category.

The project demonstrates how established deep learning components can be combined into a practical audio AI application.

---

# 🧩 End-to-End System

The complete system is designed as:

```text
                         ┌──────────────────┐
                         │   User Audio     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Audio Processing │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Mel Spectrogram  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       CNN        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     BiLSTM       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Attention     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Multi-label Head │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Predictions   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     FastAPI      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Web Interface  │
                         └──────────────────┘
```

---

# ⚡ API Layer

The application layer is designed around **FastAPI**.

The core endpoint is:

```http
POST /predict
```

The client sends an audio file to the API.

The backend then:

1. receives the audio
2. preprocesses it
3. generates the Mel Spectrogram
4. loads the trained model
5. performs inference
6. applies the stored class thresholds
7. returns predicted sound events and confidence scores

A conceptual response looks like:

```json
{
  "filename": "example.wav",
  "predictions": [
    {
      "label": "Speech",
      "confidence": 0.91
    },
    {
      "label": "Applause",
      "confidence": 0.78
    }
  ]
}
```

The exact response schema may evolve with the final API implementation.

---

# 🌐 Web Application

The final application is intended to provide a simple interface for interacting with the trained model.

The user can:

```text
Upload / Record Audio
        ↓
       Analyze
        ↓
AI Inference
        ↓
Detected Sound Events
        ↓
Confidence Visualization
```

The goal is to hide the complexity of the deep learning pipeline behind an intuitive user experience.

---

# 🧰 Technology Stack

### Deep Learning

- Python
- PyTorch
- CNN
- BiLSTM
- Attention Pooling
- BCEWithLogitsLoss
- Weighted BCE
- SpecAugment
- Automatic Mixed Precision

### Audio Processing

- Librosa
- SoundFile
- Mel Spectrogram
- STFT
- Resampling
- Fixed-duration processing

### Data Science

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Deployment

- FastAPI
- Uvicorn
- HTML
- CSS
- JavaScript

### Training Environment

- Kaggle Notebook
- GPU acceleration

---

# 📁 Suggested Repository Structure

A clean repository structure for the project is:

```text
SoundSense/
│
├── README.md
│
├── notebooks/
│   └── sound-sense.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── inference.py
│   └── utils.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   ├── soundsense_best.pth
│   ├── config.json
│   ├── classes.json
│   └── thresholds.json
│
├── requirements.txt
│
└── .gitignore
```

> Large model files should not be committed directly to Git when they exceed practical repository limits. Git LFS or an external model-storage solution can be used instead.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/GitT456/SoundSense.git
cd SoundSense
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the API

From the project root:

```bash
uvicorn api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🌐 Running the Web Interface

After starting the API, open the project's web interface according to the final frontend configuration.

The frontend communicates with the prediction API and displays:

- detected sound events
- confidence scores
- multiple simultaneous predictions

---

# 🧪 Reproducing the Training Experiment

The main experiment is available through Kaggle:

**SoundSense AI — Kaggle Notebook**

https://www.kaggle.com/code/tahazarei/sound-sense

The recommended workflow is:

```text
1. Open Kaggle Notebook
2. Attach FSD50K dataset
3. Enable GPU
4. Configure training parameters
5. Build dataset
6. Generate Mel Spectrograms
7. Train CNN + BiLSTM + Attention
8. Evaluate validation performance
9. Freeze model configuration
10. Optimize thresholds on validation data
11. Evaluate on the official evaluation split
12. Export model artifacts
```

---

# 💾 Model Artifacts

For reproducible deployment, inference should use the **same model configuration that was used during training**.

The deployment package should preserve:

```text
Model Weights
+
Model Configuration
+
Class Mapping
+
Decision Thresholds
```

This is important because loading only the neural-network weights is not sufficient if inference depends on:

- sample rate
- audio duration
- Mel parameters
- number of classes
- class ordering
- thresholds

---

# 🔐 Reproducibility

The project should preserve the following information for reproducibility:

```text
Random Seed
Sample Rate
Audio Duration
n_mels
n_fft
hop_length
f_min
f_max
CNN Channels
LSTM Hidden Size
LSTM Layers
Dropout
Learning Rate
Weight Decay
Batch Size
Number of Epochs
Loss Function
Class Weights
Augmentation Settings
Decision Thresholds
```

A saved `config.json` is recommended for this purpose.

---

# ⚠️ Important Scientific Considerations

## 1. Multi-label imbalance

Not all 200 classes occur equally often.

Therefore, accuracy alone can be misleading.

## 2. Weak / incomplete labels

FSD50K contains clip-level annotations, and the paper discusses label noise and annotation limitations.

A missing label does not necessarily mean that the corresponding sound is physically absent from the recording.

## 3. Variable duration

Real recordings are not naturally the same length.

The model therefore needs a deliberate strategy for padding, cropping, and long-audio inference.

## 4. Metric comparability

A model's mAP should not be compared directly with a published number unless:

- the same evaluation split is used
- the evaluation protocol is compatible
- the metric implementation is comparable
- preprocessing differences are understood

This project intentionally keeps that distinction explicit.

---

# 📌 Current Limitations

The current implementation has several limitations that provide clear directions for future work.

### Dataset scale

The full FSD50K release is larger than the subset that can conveniently be used under limited local/storage/compute constraints.

### Compute

Full-scale training and extensive hyperparameter experimentation require significant GPU time.

### Long audio

Very long recordings require efficient windowing and aggregation.

### Class imbalance

Rare sound classes remain challenging.

### Label ambiguity

Some acoustic events are inherently difficult to distinguish.

### Calibration

A high confidence score does not automatically mean that the probability is perfectly calibrated.

---

# 🔮 Future Improvements

Possible next steps include:

### Audio foundation models

Experiment with pretrained audio encoders and transfer learning.

### Transformer-based temporal modeling

Replace or augment the recurrent stage with an audio transformer.

### Better calibration

Apply temperature scaling or other calibration methods.

### Advanced threshold optimization

Optimize thresholds with respect to the desired operating metric and application behavior.

### More systematic ablation studies

Compare:

```text
CNN
CNN + BiLSTM
CNN + BiLSTM + Attention
```

and separately evaluate:

```text
Baseline Loss
       vs
Weighted BCE

No Augmentation
       vs
SpecAugment
```

This helps identify which components actually contribute to performance.

### Full-dataset training

Train on the complete available FSD50K data under a reproducible experimental protocol.

### Pretrained models

Compare the custom architecture against modern pretrained audio representations.

---

# 🧪 Recommended Ablation Study

A useful scientific experiment for SoundSense is:

| Experiment | CNN | BiLSTM | Attention | Weighted BCE | SpecAugment |
|---|:---:|:---:|:---:|:---:|:---:|
| Baseline | ✓ | — | — | — | — |
| + BiLSTM | ✓ | ✓ | — | — | — |
| + Attention | ✓ | ✓ | ✓ | — | — |
| + Weighted BCE | ✓ | ✓ | ✓ | ✓ | — |
| Full Model | ✓ | ✓ | ✓ | ✓ | ✓ |

The primary comparison metric should be **mAP**, with F1, precision and recall used as complementary metrics.

This turns model development into a measurable experiment instead of simply adding techniques because they sound useful.

---

# 🎓 Why This Is an End-to-End Deep Learning Project

SoundSense covers the major stages of a real machine-learning system:

```text
Problem Definition
      ↓
Dataset Selection
      ↓
Data Understanding
      ↓
Audio Preprocessing
      ↓
Feature Representation
      ↓
Architecture Design
      ↓
Training
      ↓
Evaluation
      ↓
Error Analysis
      ↓
Model Export
      ↓
API Development
      ↓
Web Application
```

The project therefore goes beyond:

> "I trained a neural network."

It demonstrates the transition from an ML experiment to a usable AI system.

---

# 👥 Team

## Taha Zarei

**Machine Learning / Deep Learning / Backend**

GitHub:  
https://github.com/GitT456

Kaggle:  
https://www.kaggle.com/tahazarei

---

## Maedeh Babaei

**Machine Learning / Deep Learning / Project Development**

GitHub:  
https://github.com/maedehbabaei-dev

---

# 🤝 Collaboration

This project was developed collaboratively by:

**Taha Zarei × Maedeh Babaei**

The repository is intended to demonstrate both the technical implementation and the collaborative development process of a Deep Learning project.

---

# 📚 References

### FSD50K Dataset

Fonseca, E., Favory, X., Pons, J., Font, F., & Serra, X.

**FSD50K: An Open Dataset of Human-Labeled Sound Events.**

IEEE/ACM Transactions on Audio, Speech, and Language Processing, Vol. 30, 2022, pp. 829–852.

DOI:

https://doi.org/10.1109/TASLP.2021.3133208

Dataset:

https://zenodo.org/records/4060432

Official project page:

https://fsannotator.upf.edu/fsd/release/FSD50K/

---

# 📝 Citation

If you use the FSD50K dataset in a research project, please cite the original paper:

```bibtex
@article{fonseca2022FSD50K,
  title={FSD50K: An Open Dataset of Human-Labeled Sound Events},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={829--852},
  year={2022},
  publisher={IEEE}
}
```

---

# 📎 Project Links

| Resource | Link |
|---|---|
| 🎧 Kaggle Notebook | https://www.kaggle.com/code/tahazarei/sound-sense |
| 👨‍💻 Taha Zarei | https://github.com/GitT456 |
| 👩‍💻 Maedeh Babaei | https://github.com/maedehbabaei-dev |
| 📚 FSD50K Official | https://fsannotator.upf.edu/fsd/release/FSD50K/ |
| 💾 FSD50K Zenodo | https://zenodo.org/records/4060432 |
| 📄 Original Paper | https://doi.org/10.1109/TASLP.2021.3133208 |

---

# ⭐ Final Note

SoundSense AI was built around a simple question:

> **Can a machine listen to a complex environment and understand the sounds happening inside it?**

The project explores that question through a complete Deep Learning pipeline:

**Audio → Mel Spectrogram → CNN → BiLSTM → Attention → Multi-Label Prediction → API → Web Application**

The focus is not only on achieving a metric, but on understanding **why the problem is difficult, how each architectural component addresses a specific challenge, how the model should be evaluated fairly, and how a trained model can become a usable application.**

---

<p align="center">
  <strong>🎧 SoundSense AI</strong><br/>
  <em>Listen. Understand. Recognize.</em>
</p>
