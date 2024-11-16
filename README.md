# SAM-Masking-Tool
This is a SAM-based interactive image annotation tool that allows users to create binary masks on images through point prompts, with category management and mask visualization capabilities. It's a tool developed based on [this project](https://github.com/wudi-ldd/SAM-Annotation).

## Visualization of SAM Mask

<img src='./images/sample.jpg' width='500'> <img src='./images/mask.jpg' width='500'>

## Prerequisites

The following packages are recommended to run the scripts:

- PyTorch and OpenCV

- pillow == 9.1.1

## Model Usage Instructions

**Step 1**: Download the pre-trained weights of the SAM model and place them in the `checkpoints` folder from [official site](https://github.com/facebookresearch/segment-anything) or [this link](https://drive.google.com/file/d/1GCQ1aCbn6_cO6jDLshYtFMUvxV98jqZ-/view?usp=sharing).

**Step 2**: Modify the following lines in `utils\file_functions.py` according to the chosen weight type:
```python
   sam_checkpoint = "checkpoints\sam_vit_l_0b3195.pth"
   model_type = "vit_l"
```

## Shortcuts

- **Left Mouse Click**: Click on areas of interest.
- **Right Mouse Click**: Click on areas not of interest.
- **Z**: Undo to the previous mouse click state.
- **E**: Finish annotation and save the current mask state.
- **Space**: Proceed to the next image.

## Installation
Download and run directly
  ```python
    python main.py
  ```

## Acknowledgment
We thank the following repos for providing helpful components/functions in our work.

- [SAM-Annotation](https://github.com/wudi-ldd/SAM-Annotation)

