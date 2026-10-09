"""Step 4: Streamlit app.   Run:  python -m streamlit run app.py"""
import hashlib
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from common import NUM_FRAMES, PersonDetector, encode_crops, get_device, preprocess_video, to_tensor
from featstore import predict_video_probs
from gradcam import explain, overlay
from model import load_model

st.set_page_config(page_title="Explainable HAR", layout="wide")
st.title("Explainable Human Activity Recognition")
st.caption("MobileNetV2 + LSTM with Grad-CAM")

ckpt = st.sidebar.text_input("Model checkpoint", "outputs/har_model.pt")
if not os.path.exists(ckpt):
    st.error(f"Checkpoint not found: {ckpt}. Run prepare_data.py and train.py first.")
    st.stop()


@st.cache_resource
def load_all(path):
    dev = get_device()
    model, classes = load_model(path, dev)
    return model, classes, PersonDetector(dev), dev


@st.cache_resource(show_spinner=False, max_entries=2)
def analyse(digest, _data, suffix, path):
    model, classes, detector, dev = load_all(path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(_data)
        tmp = f.name
    try:
        pre = preprocess_video(tmp, detector)
    finally:
        os.remove(tmp)
    cnn = encode_crops(model.encoder, pre["crops"], dev)
    mean, std = model.box_mean.cpu().numpy(), model.box_std.cpu().numpy()
    starts, wp = predict_video_probs(model.head, cnn, (pre["boxf"] - mean) / std, dev)
    probs = wp.mean(0)                              # video prediction = mean over all clips
    pred = int(probs.argmax())
    best = int(wp[:, pred].argmax())                # clip that supports the prediction most
    return dict(classes=classes, probs=probs, pred=pred, starts=starts, wp=wp, best=best, pre=pre, cams={})


up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if up is None:
    st.info("Upload a short video of someone walking, jogging, running, boxing, handwaving or clapping.")
    st.stop()

data = up.getvalue()
digest = hashlib.md5(data).hexdigest()
st.video(data)
with st.spinner("Detecting person, extracting features, running LSTM on all clips..."):
    r = analyse(digest, data, os.path.splitext(up.name)[1] or ".mp4", ckpt)

model, classes, _, _ = load_all(ckpt)
pre, starts, wp = r["pre"], r["starts"], r["wp"]
L, W = len(pre["crops"]), len(starts)
pretty = lambda c: c.replace("hand", "hand ").title()

c1, c2, c3 = st.columns(3)
c1.metric("Predicted activity", pretty(classes[r["pred"]]))
c2.metric("Confidence", f"{r['probs'][r['pred']]:.1%}")
c3.metric("Frames analysed", f"{L}  ({W} clips of {NUM_FRAMES})")
st.caption(f"{int((wp.argmax(1) == r['pred']).sum())} of {W} clips agree with the prediction. "
           f"Person detected in {pre['n_det']} of {pre['n_key']} checked frames (boxes in between are interpolated).")

st.subheader("Class probability over the video")
st.line_chart({pretty(c): wp[:, i] for i, c in enumerate(classes)})

clip = r["best"] if W == 1 else st.slider("Clip (Grad-CAM is computed for this clip)", 1, W, r["best"] + 1, key=f"clip_{digest}") - 1
t = st.slider("Frame in clip", 1, NUM_FRAMES, 1, key=f"frame_{digest}") - 1
s = starts[clip]

if clip not in r["cams"]:
    with st.spinner("Computing Grad-CAM..."):
        _, _, cams = explain(model, to_tensor(pre["crops"][s:s + NUM_FRAMES]), pre["boxf"][s:s + NUM_FRAMES], r["pred"])
        r["cams"][clip] = [overlay(c, m) for c, m in zip(pre["crops"][s:s + NUM_FRAMES], cams)]

x1, y1, x2, y2 = pre["boxes"][s + t]
ann = pre["frames"][s + t].copy()
cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)

a, b, c = st.columns(3)
a.image(ann, caption=f"Frame {s + t + 1}: person box", use_container_width=True)
b.image(pre["crops"][s + t], caption="Cropped & resized (224x224)", use_container_width=True)
c.image(r["cams"][clip][t], caption=f"Grad-CAM for '{pretty(classes[r['pred']])}'", use_container_width=True)

st.subheader("Average class probabilities")
st.bar_chart({pretty(k): float(v) for k, v in zip(classes, r["probs"])})
