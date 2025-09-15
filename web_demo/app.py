import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.models.multitask_model import MultiTaskModel
from src.utils.visualization import overlay_mask

st.set_page_config(page_title="Pathology Multitask Demo", layout="wide")
st.title("病理多任务演示：分割 + 分类")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel().to(device)
    model.eval()
    return model, device


model, device = load_model()

up = st.file_uploader("上传一张 Patch (PNG/JPG)", type=["png", "jpg", "jpeg"])
if up is not None:
    img = Image.open(up).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_logits, cls_logits = model(x)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
        cls_prob = torch.sigmoid(cls_logits)[0, 0].item()
    st.write(f"分类概率（肿瘤）：{cls_prob:.3f}")
    overlay = overlay_mask((arr * 255).astype(np.uint8), seg_prob, alpha=0.5)
    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="原图", use_column_width=True)
    with c2:
        st.image(overlay, caption="叠加分割", use_column_width=True)
else:
    st.info("请上传一张病理 Patch 图像以进行推理演示。")
