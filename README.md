# HPEval-Human-Pose-Estimation
Group project for UTSA Artificial Intelligence graduate class

In this project we will implement/evaluate different approaches to human pose estimation.

## DeepPose

Requires linux for `pycocoutils` library <br>
Implementation based on https://arxiv.org/abs/1312.4659

### Training
1. Download COCO 2017 Val and Train datasets (https://cocodataset.org/#download)
2. Update config.ini to point to the appropriate unzipped directories
3. Install needed python dependencies via `python -m pip install -r requirements.txt` (We used python3.8)
4. From base of project directory run the following command
```
python src/deep_pose/main.py --config_file config.ini
```

### Visualization
Once training is complete, you can run the model on images and visualize where the model thinks the joints are.
Use the following command
```
python src/deep_pose/visualize.py --image_path <path to your image> --model_path <path to model outputted by training step>
```
### References
Below is a list of websites referenced for this implementation:
- https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/
- https://sebastianraschka.com/faq/docs/training-loop-in-pytorch.html
- https://github.com/mitmul/deeppose?utm_source=catalyzex.com
- https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4/
