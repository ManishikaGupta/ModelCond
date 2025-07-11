import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageStat, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from transformers import CLIPProcessor, CLIPModel
import os
from datetime import datetime
import cv2
import csv
from fpdf import FPDF

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Augmentation functions
def add_noise(image, noise_level=20):
    np_image = np.array(image)
    noise = np.random.randint(-noise_level, noise_level, np_image.shape).astype(np.int16)
    noisy_image = np.clip(np_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

# Quality metric calculations
def calculate_sharpness(image):
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    laplacian = np.var(np.gradient(img_array))
    return laplacian

def calculate_brightness(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.mean[0] / 255.0

def calculate_contrast(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return stat.stddev[0] / 255.0

# Defect detection using OpenCV
def detect_defects(image):
    def detect_cracks_overlay(gray_img, original_img, output_path):
        edges = cv2.Canny(gray_img, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = original_img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(output_path, overlay)
        crack_pixels = np.sum(edges > 128)
        return crack_pixels / (gray_img.shape[0] * gray_img.shape[1])

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_score = sum([cv2.contourArea(c) for c in contours]) / (gray.shape[0] * gray.shape[1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logged_images", exist_ok=True)
    crack_overlay_path = f"logged_images/{timestamp}_crack_overlay.jpg"
    crack_score = detect_cracks_overlay(gray, cv_image, crack_overlay_path)
    return defect_score + crack_score

def show_augmented_versions(image):
    blurred = add_blur(image, 2)
    noisy = add_noise(image, 20)
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[1].imshow(blurred)
    axs[1].set_title("Blurred")
    axs[2].imshow(noisy)
    axs[2].set_title("Noisy")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    os.makedirs("logged_images", exist_ok=True)
    plt.savefig("logged_images/augmentations_preview.png")

def log_to_csv(timestamp, score, category, sharpness, brightness, contrast, defect_score):
    os.makedirs("logs", exist_ok=True)
    csv_file = os.path.join("logs", "assessment_log.csv")
    headers = ["Timestamp", "Score", "Category", "Sharpness", "Brightness", "Contrast", "Defect Score"]
    row = [timestamp, score, category, f"{sharpness:.2f}", f"{brightness:.2f}", f"{contrast:.2f}", f"{defect_score:.4f}"]
    write_headers = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(headers)
        writer.writerow(row)

def generate_pdf_summary(timestamp, score, category, sharpness, brightness, contrast, defect_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Walmart Product Condition Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Timestamp: {timestamp}", ln=True)
    pdf.cell(200, 10, txt=f"Category: {category}", ln=True)
    pdf.cell(200, 10, txt=f"Condition Score: {score}/10", ln=True)
    pdf.cell(200, 10, txt=f"Sharpness: {sharpness:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Brightness: {brightness:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Contrast: {contrast:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Defect Score: {defect_score:.4f}", ln=True)
    os.makedirs("logs", exist_ok=True)
    pdf.output(f"logs/{timestamp}_report.pdf")

def compute_condition_score(image, category):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        feature_norm = torch.norm(image_features).item()

    sharpness = calculate_sharpness(image)
    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    defect_score = detect_defects(image)

    weights = dict(sharpness=0.3, brightness=0.3, contrast=0.2, feature=0.1)
    penalty_factor = 6

    if category == "Clothes":
        weights = dict(sharpness=0.35, brightness=0.25, contrast=0.15, feature=0.1)
        penalty_factor = 10
    elif category == "Shoes":
        weights = dict(sharpness=0.25, brightness=0.2, contrast=0.25, feature=0.2)
        penalty_factor = 7
    elif category == "Bags":
        weights = dict(sharpness=0.3, brightness=0.25, contrast=0.2, feature=0.15)
        penalty_factor = 8
    elif category == "Electronics":
        weights = dict(sharpness=0.2, brightness=0.2, contrast=0.3, feature=0.2)
        penalty_factor = 5

    base_score = (
        weights['sharpness'] * min(1.0, sharpness / 50.0) +
        weights['brightness'] * brightness +
        weights['contrast'] * contrast +
        weights['feature'] * min(1.0, feature_norm)
    ) * 10

    penalty = min(defect_score * penalty_factor, 5)
    final_score = max(0.0, base_score - penalty)

    return round(final_score, 1), sharpness, brightness, contrast, defect_score

def predict_condition(image, simulate_damage=False, category="Clothes"):
    original_image = image.copy()
    if simulate_damage:
        image = add_blur(image, radius=2)
        image = add_noise(image, noise_level=30)

    score, sharpness, brightness, contrast, defect_score = compute_condition_score(image, category)

    if score <= 3:
        verdict = "❌ Poor condition. Recommend Recycle ♻️"
    elif score <= 7:
        verdict = "⚠️ Average condition. Consider Recycle or Donate."
    else:
        verdict = "✅ Good condition. Not recommended for recycling yet."

    os.makedirs("logged_images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_image = original_image.convert("RGB")
    original_image.save(f"logged_images/{timestamp}_score_{score}_{category}.jpg")

    show_augmented_versions(original_image)
    log_to_csv(timestamp, score, category, sharpness, brightness, contrast, defect_score)
    generate_pdf_summary(timestamp, score, category, sharpness, brightness, contrast, defect_score)

    metrics_report = (
        f"\nCategory: {category}\n"
        f"Sharpness: {sharpness:.2f}\n"
        f"Brightness: {brightness:.2f}\n"
        f"Contrast: {contrast:.2f}\n"
        f"Defect Score: {defect_score:.4f}"
    )

    return f"🧵 Condition Score: {score}/10\n{verdict}{metrics_report}"

def dashboard_view():
    try:
        df = pd.read_csv("logs/assessment_log.csv")
        df = df.tail(10)

        # Display table
        st.dataframe(df)

        # Plot trend
        st.subheader("Condition Score Trend")
        fig, ax = plt.subplots()
        df_sorted = df.sort_values("Timestamp")
        ax.plot(df_sorted["Timestamp"], df_sorted["Score"].astype(float), marker='o')
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Score")
        ax.set_xticklabels(df_sorted["Timestamp"], rotation=45, ha='right', fontsize=8)
        st.pyplot(fig)

        # Download links
        for _, row in df.iterrows():
            img_path = f"logged_images/{row['Timestamp']}_score_{row['Score']}_{row['Category']}.jpg"
            pdf_path = f"logs/{row['Timestamp']}_report.pdf"
            st.markdown(f"🖼️ **{row['Timestamp']}** | [{row['Category']}] | Score: {row['Score']}")
            if os.path.exists(img_path):
                st.image(img_path, width=150)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    b64_pdf = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{os.path.basename(pdf_path)}">📄 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Dashboard not available: {e}")

# STREAMLIT UI
st.set_page_config(page_title="Walmart Universal Product Condition Estimator")
tab1, tab2 = st.tabs(["Condition Estimator", "Assessment Dashboard"])

with tab1:
    st.title("Walmart Universal Product Condition Estimator")
    st.markdown("Upload a product image. The model analyzes its condition, detects defects, and logs outputs to CSV/PDF.")

    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])
    simulate_damage = st.checkbox("Simulate Damage (Blur + Noise)")
    category = st.selectbox("Product Category", ["Clothes", "Shoes", "Bags", "Electronics"], index=0)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Condition"):
            result = predict_condition(image, simulate_damage, category)
            st.text_area("Condition Assessment", value=result, height=200)

with tab2:
    st.title("Assessment Dashboard")
    if st.button("Load Dashboard"):
        dashboard_view()
