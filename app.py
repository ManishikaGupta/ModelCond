import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageStat, ImageFilter
import numpy as np
from pandas import read_csv
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
import os
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import csv
from fpdf import FPDF

"""# Step 1: Load CLIP Model"""

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

"""# Step 2: Preprocessing & Augmentation"""

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def add_noise(image, noise_level=20):
    np_image = np.array(image)
    noise = np.random.randint(-noise_level, noise_level, np_image.shape).astype(np.int16)
    noisy_image = np.clip(np_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

"""# Step 3: Utility Functions"""

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
    return (stat.stddev[0] / 255.0)

def detect_defects(image):
    # Crack detection for electronics using contour + visual overlay
    def detect_cracks_overlay(gray_img, original_img, output_path):
        edges = cv2.Canny(gray_img, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = original_img.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # avoid tiny noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite(output_path, overlay)
        crack_pixels = np.sum(edges > 128)
        return crack_pixels / (gray_img.shape[0] * gray_img.shape[1])

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_score = sum([cv2.contourArea(c) for c in contours]) / (gray.shape[0] * gray.shape[1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    pdf.output(f"logs/{timestamp}_report.pdf")

"""# Step 4: Condition Scoring Using CLIP Embeddings + Quality Metrics

"""

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
    else:
        weights = dict(sharpness=0.3, brightness=0.3, contrast=0.2, feature=0.1)
        penalty_factor = 6

    base_score = (
        weights['sharpness'] * min(1.0, sharpness / 50.0) +
        weights['brightness'] * brightness +
        weights['contrast'] * contrast +
        weights['feature'] * min(1.0, feature_norm)
    ) * 10

    penalty = min(defect_score * penalty_factor, 5)
    final_score = max(0.0, base_score - penalty)

    return round(final_score, 1), sharpness, brightness, contrast, defect_score

"""# Step 5: Full Pipeline with Logging and PDF Export"""

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
    original_image.save(f"logged_images/{timestamp}_score_{score}_{category}.jpg")
    show_augmented_versions(original_image)
    log_to_csv(timestamp, score, category, sharpness, brightness, contrast, defect_score)
    generate_pdf_summary(timestamp, score, category, sharpness, brightness, contrast, defect_score)

    metrics_report = (
        f"\nCategory: {category}\n"
        f"Sharpness: {sharpness:.2f}\n"
        f"Brightness: {brightness:.2f}\n"
        f"Contrast: {contrast:.2f}\n"
        f"Defect Score (Holes/Threads): {defect_score:.4f}"
    )

    return f"🧵 Condition Score: {score}/10\n{verdict}{metrics_report}"

"""# Step 6: Gradio Interface"""

def dashboard_view():
    try:
        df = read_csv("logs/assessment_log.csv")
        df['Image'] = df['Timestamp'].apply(lambda ts: f"logged_images/{ts}_score_" + df['Score'].astype(str) + "_" + df['Category'] + ".jpg")
        preview_df = df.tail(10)
        html = "<table><tr>" + "".join(f"<th>{col}</th>" for col in preview_df.columns if col != 'Image') + "<th>Image</th><th>PDF</th></tr>"
        for _, row in preview_df.iterrows():
            html += "<tr>" + "".join(f"<td>{row[col]}</td>" for col in preview_df.columns if col != 'Image')
            img_path = row['Image']
            pdf_path = f"logs/{row['Timestamp']}_report.pdf"
            if os.path.exists(img_path):
                html += f'<td><img src="file/{img_path}" width="100"></td>'
            else:
                html += '<td>Image not found</td>'
            if os.path.exists(pdf_path):
                html += f'<td><a href="file/{pdf_path}" download>Download PDF</a></td>'
            else:
                html += '<td>No PDF</td>'
            html += "</tr>"
        html += "</table>"

        # Add line chart of scores over time
        try:
            fig, ax = plt.subplots()
            df_sorted = df.sort_values("Timestamp")
            ax.plot(df_sorted["Timestamp"].tail(20), df_sorted["Score"].astype(float).tail(20), marker='o')
            ax.set_title("Condition Score Trend")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Score")
            ax.set_xticklabels(df_sorted["Timestamp"].tail(20), rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.getvalue()).decode()
            html += f'<br><h3>Condition Score Trend</h3><img src="data:image/png;base64,{chart_base64}" width="600">'
        except Exception as chart_err:
            html += f"<p>Could not generate chart: {chart_err}</p>"
        return HTML(html)
    except Exception as e:
        return f"Dashboard not available: {e}"
# Assume you have these defined already
# from your_model_file import predict_condition, dashboard_view

st.set_page_config(page_title="Walmart Universal Product Condition Estimator")

# Tabs
tab1, tab2 = st.tabs(["Condition Estimator", "Assessment Dashboard"])

with tab1:
    st.title("Walmart Universal Product Condition Estimator")
    st.markdown("Upload a product image. The model analyzes its condition, detects defects (including cracks), and recommends recycling. Logs outputs to CSV/PDF.")

    # Inputs
    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])
    simulate_damage = st.checkbox("Simulate Damage (Blur + Noise)")
    category = st.selectbox("Product Category", ["Clothes", "Shoes", "Bags", "Electronics"], index=0)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Show preview
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Condition"):
            result = predict_condition(image, simulate_damage, category)
            st.text_area("Condition Assessment", value=result, height=200)

with tab2:
    st.title("Assessment Dashboard")
    st.markdown("View the latest 10 logged assessments with scores and metrics.")

    if st.button("Load Dashboard"):
        dashboard_output = dashboard_view()
        st.text_area("Dashboard Output", value=dashboard_output, height=400)
