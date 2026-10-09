# Explainable Human Activity Recognition (CNN-LSTM + Grad-CAM + Streamlit)

## Run order
1. `python -m pip install -r requirements.txt`
2. Put the KTH videos in class folders, e.g. `kth_videos/boxing/*.avi`, `kth_videos/handclapping/*.avi`, ...
3. `python prepare_data.py --data kth_videos`
   Reads every video densely (~12.5 fps), detects the person on every 8th frame, interpolates the box in between,
   crops + resizes to 224x224, extracts MobileNetV2 features for every frame -> `cache/features.npz`.
   Resumable: finished videos are cached in `cache/videos/`.
4. `python train.py`     trains the LSTM on random 16-frame clips -> `outputs/har_model.pt`
5. `python evaluate.py`  test accuracy per VIDEO (average over sliding clips), confusion matrix, `test_predictions.csv`
6. `python -m streamlit run app.py`

## What changed vs. the first version
- Clips of 16 *consecutive* sampled frames (1.3 s) instead of 16 frames spread over the whole video, so the LSTM actually sees
  motion cycles (clap / wave). Many clips per video = far more training data.
- Box position/size/speed added as extra LSTM input (cropping removes how fast the person moves).
- Video prediction = average over all sliding clips. LSTM output averaged over time. Label smoothing + input dropout.
- Split is by person (standard KTH protocol); test persons are never used for training or model selection.
