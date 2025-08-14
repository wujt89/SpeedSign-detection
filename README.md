### Speed Limit Sign Detection

A classical computer-vision pipeline for detecting and classifying speed‑limit traffic signs. The system combines circular proposal generation (Hough transform) with local feature descriptors (SIFT) aggregated via a Bag‑of‑Visual‑Words (BoVW) vocabulary and a Random Forest classifier.

### Key capabilities
- **Circle proposal filtering**: Removes nested/duplicate circle proposals to yield stable regions of interest.
- **BoVW vocabulary learning**: Builds a SIFT vocabulary from positive examples and saves it to `voc.npy`.
- **Descriptor extraction**: Computes BoVW descriptors for candidate boxes in both training and inference.
- **Classification**: Trains a `RandomForestClassifier` to distinguish `speedlimit` vs `other`.
- **Two modes of operation**:
  - **classify**: Classify user-provided regions in test images.
  - **detect**: Automatically propose circular regions via Hough Circles and classify them.

### Requirements
- Python 3.8+
- Packages:
  - `opencv-contrib-python` (SIFT and BoW modules)
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `pandas`

Install in a fresh environment:

```bash
pip install opencv-contrib-python numpy scikit-learn matplotlib pandas
```

### Expected dataset layout
This project expects training and test data to be located one directory above the project root:

```text
../train/
  images/               # training images
  annotations/          # PASCAL VOC-style XML files
../test/
  images/               # test images
  annotations/          # PASCAL VOC-style XML files
```

Each XML annotation should contain object entries with `name` equal to `speedlimit` for positive instances and bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`).

### Running
From the project directory:

```bash
python main.py
```

The script will:
1) Load annotations for `train` and `test` sets.
2) Learn a SIFT BoVW vocabulary from positive regions and save it to `voc.npy`.
3) Extract descriptors for the training set and train a Random Forest classifier.
4) Prompt for an operating mode: `classify` or `detect`.

### Interactive usage
- **classify** mode:
  1) Enter number of files to process.
  2) For each file, enter the image filename (from `../test/images/`).
  3) Enter how many regions you want to classify within that image.
  4) For each region, provide coordinates as space‑separated integers: `xmin xmax ymin ymax`.
  5) The program prints `speedlimit` or `other` for each region.

- **detect** mode:
  - The script runs circle detection on each test image using `cv2.HoughCircles` with `cv2.HOUGH_GRADIENT_ALT`, converts circles to bounding boxes, extracts BoVW descriptors, classifies them, and prints:
    - image filename
    - number of detections predicted as `speedlimit`
    - bounding box coordinates for each detection (`xmin xmax ymin ymax`)

### Implementation overview (functions)
- `checkCircle(x1, y1, x2, y2, r1, r2)`: Helper to suppress circles fully contained within another circle.
- `loadAndCirclePhoto(path)`: Reads an image, runs Hough Circles, returns candidate boxes.
- `checkAndDrawRedCircles(circles, actual_img, is_empty)`: Filters circle proposals and converts them to bounding boxes.
- `load(trainOrTest, xmlFolder)`: Loads image names and boxes from XML; also samples random negative boxes.
- `learn(data)`: Trains a BoVW vocabulary from positive regions (and random crops) and saves `voc.npy`.
- `extractSinglePhotoClassify(name, coordinates)`: Extracts a BoVW descriptor for a single region in a single test image.
- `extractDetect(data, path)`: For each test image, generates circle proposals and extracts descriptors for each box.
- `extract(data, path)`: Extracts descriptors for all annotated boxes in the dataset.
- `train(data)`: Trains a `RandomForestClassifier` on BoVW descriptors.
- `predictSinglePhotoClassify(rf, desc)`: Prints a label prediction for a single region.
- `predictDetect(rf, data)`: Prints detections for each test image and returns enriched metadata.
- `predictUniversal(rf, data)`: Adds predictions for all annotated boxes (utility).
- `evaluate(data)`: Prints overall classification accuracy given predicted vs. true labels.
- `evaluateDetect(data)`: Utility for drawing predicted boxes; can be adapted for further evaluation.

### Notes
- SIFT and BoW functionality require `opencv-contrib-python`.
- The BoVW vocabulary is saved as `voc.npy` in the project root and reused by extraction functions.
- Hough Circle parameters (e.g., `param1`, `param2`, `minRadius`, `maxRadius`) are tuned empirically and may need adjustment for other datasets.

