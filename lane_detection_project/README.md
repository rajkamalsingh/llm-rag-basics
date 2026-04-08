# Real-World Lane Detection using Classical Computer Vision

##  Overview
This project explores how **classical computer vision techniques** perform for lane detection in real-world driving scenarios. Instead of focusing on perfect accuracy, the goal is to analyze how edge-based methods behave under challenging conditions such as shadows, lighting variations, motion blur, and cluttered backgrounds.

---

## Objective
- Apply **edge detection techniques** for lane detection  
- Evaluate performance on **real-world noisy data**  
- Analyze **successes and failure cases**  
- Improve robustness through preprocessing and parameter tuning  

---

## Methodology

### Pipeline
The lane detection system follows this pipeline:

1. **Grayscale Conversion** – simplify image representation  
2. **Histogram Equalization** – improve contrast in low-light conditions  
3. **Gaussian Blur** – reduce noise  
4. **Canny Edge Detection** – detect edges  
5. **Region of Interest (ROI)** – focus on road area  
6. **Hough Line Transform** – detect lane lines  
7. **Slope Filtering** – remove irrelevant edges  

---

##  Baseline vs Improved Pipeline

### Baseline
- Grayscale → Blur → Canny → ROI → Hough  

### Improved
- Grayscale → Equalization → Blur → Adaptive Canny → Morphology → ROI → Hough → Slope Filtering  

---

## Experiments

### 1. Gaussian Blur Study
- Tested kernel sizes: (3×3), (5×5), (9×9)  
- Trade-off between noise reduction and edge preservation  

### 2. Canny Threshold Sensitivity
- Tested thresholds: (50,150), (100,200), (150,300)  
- Observed over-detection vs under-detection  

### 3. Real-World Stress Testing
Tested on:
- Bright sunlight  
- Shadows  
- Motion blur  
- Cluttered road scenes  

### 4. Baseline vs Improved Comparison
- Compared robustness across different conditions  
- Evaluated visual quality and consistency  

---

## Results

- Moderate blur (5×5) provided best balance  
- Adaptive thresholds improved performance across lighting conditions  
- Improved pipeline reduced noise and false detections  
- Performance degraded in extreme conditions  

---

## Failure Analysis

- **Shadows:** detected as false edges  
- **Bright light:** weak lane visibility due to low contrast  
- **Motion blur:** fragmented edges  
- **Curved lanes:** not well captured by straight-line detection  

---

## Improvements

- Histogram equalization for contrast enhancement  
- Adaptive Canny thresholds for dynamic edge detection  
- Morphological operations to clean edges  
- Slope filtering to remove irrelevant lines  

---

## Project Structure
```lane-detection-project/
│
├── src/
│ ├── main.py
│ ├── utils.py
│
├── outputs/ # experiment results (ignored in repo)
├── data/ # input videos (ignored in repo)
│
├── README.md
└── requirements.txt
```

---

## Key Takeaway
Classical computer vision techniques can perform well in controlled environments but struggle in real-world conditions due to sensitivity to lighting, noise, and environmental variability.

---

## Future Work
- Improve detection for curved lanes  
- Incorporate color-based lane filtering  
- Explore hybrid approaches with learning-based methods  

---
##  How to Run

### Clone the Repository
```bash
git clone https://github.com/rajkamalsingh/llm-rag-basics.git
cd lane-detection-project
```
---
### Add Input Video
Place your input video inside the data/ folder
 - Example:data/input_video.mp4
---
### Run the Lane Detection Script
```
python src/mains.py
```
---
### Output
 - Processed video / frames will be saved in the outputs/ folder
 - Intermediate outputs (edges, ROI, etc.) may also be saved for analysis

---
### Optional: Run Experiments

To run specific experiments (blur, thresholds, comparison), modify parameters in:
```
src/mains.py
src/exp_blur.py
src/exp_canny
```
#### Example:
 - Change Gaussian blur kernel size
 - Adjust Canny thresholds
 - Switch between baseline and improved pipeline

---
## Author
**Raj Kamal Singh**