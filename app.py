import streamlit as st
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import base64
from io import BytesIO
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Page config
st.set_page_config(page_title="Sneaker Vision AI", layout="wide", page_icon="👟")

# Custom CSS with sneaker theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #000000;
        background-attachment: fixed;
    }
    
    [data-testid="stHeader"] {display: none;}
    
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 35px;
        padding: 3rem 2.5rem;
        box-shadow: 0 25px 80px rgba(0,0,0,0.6);
        max-width: 1400px;
        margin: 2rem auto;
        backdrop-filter: blur(10px);
    }
    
    .floating-sneaker {
        position: fixed;
        width: 15vw;
        height: 15vw;
        z-index: 1000;
        opacity: 0.1;
        animation: float 8s ease-in-out infinite;
        pointer-events: none;

        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        font-size: 12vw;
    }
    
    .sneaker-1 {
        top: 10%;
        left: 5%;
        animation-delay: 0s;
    }
    
    .sneaker-2 {
        top: 10%;
        right: 5%;
        animation-delay: 1s;
    }
    
    .sneaker-3 {
        top: 50%;
        left: 3%;
        animation-delay: 2s;
    }
    
    .sneaker-4 {
        top: 50%;
        right: 3%;
        animation-delay: 3s;
    }
    
    .sneaker-5 {
        bottom: 10%;
        left: 5%;
        animation-delay: 4s;
    }
    
    .sneaker-6 {
        bottom: 10%;
        right: 5%;
        animation-delay: 5s;
    }
    
    @keyframes float {
        0% {
            transform: translateY(0px) translateX(0px) rotate(0deg) scale(1);
        }
        25% {
            transform: translateY(-50px) translateX(20px) rotate(-15deg) scale(1.1);
        }
        50% {
            transform: translateY(-80px) translateX(-30px) rotate(20deg) scale(0.9);
        }
        75% {
            transform: translateY(-40px) translateX(25px) rotate(-10deg) scale(1.05);
        }
        100% {
            transform: translateY(0px) translateX(0px) rotate(0deg) scale(1);
        }
    }
    
    .title {
        text-align: center;
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
        letter-spacing: -2px;
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .analyze-btn button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1.2rem 3rem;
        font-size: 1.3rem;
        font-weight: 700;
        border-radius: 20px;
        width: 100%;
        transition: all 0.4s ease;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .analyze-btn button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .result-card {
        padding: 2.5rem;
        border-radius: 25px;
        animation: slideUp 0.6s ease-out, glow 2s ease-in-out infinite;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    .checkmark {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: block;
        margin: 0 auto 1rem auto;
        box-shadow: none; /* remove glow so circle is solid */
        animation: none;
    }

    .checkmark-circle {
        fill: #000000;   /* full black circle */
        stroke: none;
    }

    .checkmark-check {
        stroke: #ffffff; /* white tick */
        stroke-width: 4;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    
    @keyframes stroke {
        100% {
            stroke-dashoffset: 0;
        }
    }
    
    @keyframes scale {
        0%, 100% {
            transform: none;
        }
        50% {
            transform: scale3d(1.1, 1.1, 1);
        }
    }
    
    @keyframes fill {
        100% {
            box-shadow: inset 0px 0px 0px 30px rgba(255,255,255,0.2);
        }
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-100%) translateY(-100%) rotate(45deg);
        }
        100% {
            transform: translateX(100%) translateY(100%) rotate(45deg);
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        50% {
            box-shadow: 0 0 40px rgba(102, 126, 234, 0.6);
        }
    }

    .confidence-bar {
        width: 100%;
        height: 15px;
        background: rgba(255,255,255,0.25);
        border-radius: 15px;
        overflow: hidden;
        margin: 1.5rem 0;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.7) 0%, rgba(255,255,255,1) 100%);
        border-radius: 15px;
        transition: width 1.2s cubic-bezier(0.65, 0, 0.35, 1);
        animation: fillBar 1.2s ease-out;
        box-shadow: 0 0 10px rgba(255,255,255,0.5);
    }

    @keyframes fillBar {
        from {
            width: 0% !important;
        }
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 3rem;
        padding: 2.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-radius: 25px;
        border: 2px solid rgba(102, 126, 234, 0.1);
    }

    .stat-box {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.4s ease;
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .stat-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .stat-box:hover::before {
        left: 100%;
    }

    .stat-box:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.25);
        border: 2px solid rgba(102, 126, 234, 0.3);
    }

    .stat-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-5px);
        }
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 0.95rem;
        color: #6b7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .image-container {
        background: white;
        padding: 2rem;
        border-radius: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 3px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.2);
    }

    .image-container img {
        border-radius: 20px;
        width: 100% !important;
        height: auto !important;
        max-height: 500px;
        object-fit: contain;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .top-predictions {
        margin-top: 2rem;
        padding: 1.5rem;
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .prediction-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .prediction-item:hover {
        background: rgba(255,255,255,0.3);
        transform: translateX(5px);
    }
    
    .prediction-name {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .prediction-percent {
        font-weight: 700;
        font-size: 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 1rem;
    }

    .stFileUploader {
        margin-bottom: 2rem;
    }
    
    .stFileUploader > div {
        border: 3px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
        background: rgba(0, 0, 0, 0.3);
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.05);
    }

</style>
""", unsafe_allow_html=True)

# Add floating sneaker emojis
st.markdown("""
<div class="floating-sneaker sneaker-1">👟</div>
<div class="floating-sneaker sneaker-2">👟</div>
<div class="floating-sneaker sneaker-3">👟</div>
<div class="floating-sneaker sneaker-4">👟</div>
<div class="floating-sneaker sneaker-5">👟</div>
<div class="floating-sneaker sneaker-6">👟</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'sneaker_counts' not in st.session_state:
    st.session_state.sneaker_counts = {}

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        # Replace with your model path
        model_path = "Aadyasingh/sneaker_classifier"  # or use Hugging Face: "your-username/sneaker-classifier"
        
        processor = ViTImageProcessor.from_pretrained(model_path)
        # CRITICAL: Load model with output_attentions=True to enable attention visualization
        model = ViTForImageClassification.from_pretrained(model_path, output_attentions=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Make sure your model is in './sneakers_top10_best' or update the path")
        return None, None, None

# Generate comprehensive visualizations
def generate_visualizations(model, image, target_class, device, processor):
    """Generate multiple visualization types"""
    visualizations = {}
    
    try:
        # Prepare inputs
        inputs = processor(images=image, return_tensors="pt").to(device)
        img_array = np.array(image.resize((224, 224)))
        
        # 1. ATTENTION ROLLOUT
        try:
            with torch.no_grad():
                outputs = model(pixel_values=inputs['pixel_values'], output_attentions=True)
                
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attentions = outputs.attentions
                    
                    # Attention rollout
                    num_tokens = attentions[0].size(-1)
                    result = torch.eye(num_tokens, device=device)
                    
                    for attention_layer in reversed(attentions):
                        attention_heads_fused = attention_layer.mean(dim=1)[0]
                        current_attention = attention_heads_fused + torch.eye(num_tokens, device=device)
                        current_attention = current_attention / (current_attention.sum(dim=-1, keepdim=True) + 1e-12)
                        result = torch.matmul(current_attention, result)
                    
                    mask = result[0, 1:].cpu().numpy()
                    mask = mask / (mask.sum() + 1e-12)
                    
                    num_patches = int(np.sqrt(len(mask)))
                    attention_map = mask.reshape(num_patches, num_patches)
                    
                    # Create attention overlay
                    attention_resized = cv2.resize(attention_map, (224, 224))
                    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-12)
                    
                    # Apply colormap
                    heatmap = cm.hot(attention_resized)[:, :, :3] * 255
                    heatmap = heatmap.astype(np.uint8)
                    attention_overlay = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
                    
                    visualizations['attention'] = Image.fromarray(attention_overlay)
                    visualizations['attention_raw'] = Image.fromarray((cm.hot(attention_resized)[:, :, :3] * 255).astype(np.uint8))
        except Exception as e:
            print(f"Attention rollout failed: {e}")
        
        # 2. GRADIENT SALIENCY
        try:
            model.zero_grad()
            inputs_grad = processor(images=image, return_tensors="pt").to(device)
            inputs_grad['pixel_values'].requires_grad = True
            
            outputs = model(**inputs_grad)
            target_score = outputs.logits[0, target_class]
            target_score.backward()
            
            gradients = inputs_grad['pixel_values'].grad.data[0].cpu().numpy()
            saliency = np.abs(gradients).max(axis=0)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-12)
            
            # Create gradient overlay
            heatmap = cm.hot(saliency)[:, :, :3] * 255
            heatmap = heatmap.astype(np.uint8)
            gradient_overlay = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
            
            visualizations['gradient'] = Image.fromarray(gradient_overlay)
            visualizations['gradient_raw'] = Image.fromarray((cm.hot(saliency)[:, :, :3] * 255).astype(np.uint8))
            
            # 3. COMBINED
            if 'attention' in visualizations:
                attention_resized = cv2.resize(attention_map, (224, 224))
                attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-12)
                combined = (attention_resized + saliency) / 2
                
                heatmap = cm.hot(combined)[:, :, :3] * 255
                heatmap = heatmap.astype(np.uint8)
                combined_overlay = cv2.addWeighted(img_array, 0.4, heatmap, 0.6, 0)
                
                visualizations['combined'] = Image.fromarray(combined_overlay)
                visualizations['combined_raw'] = Image.fromarray((cm.hot(combined)[:, :, :3] * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"Gradient saliency failed: {e}")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return visualizations

# Title
st.markdown('<div class="title">SNEAKER VISION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Identify 10 Iconic Sneaker Models with Precision</div>', unsafe_allow_html=True)

# Center the upload section
st.markdown('<div style="max-width: 600px; margin: 0 auto;">', unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("Drop your sneaker image here", type=['png', 'jpg', 'jpeg'], label_visibility="visible")

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Auto-analyze on upload
    with st.spinner('Analyzing your sneaker...'):
        # Load model
        processor, model, device = load_model()
        
        if processor is None or model is None:
            st.error("❌ Failed to load the model. Please check the model path.")
        else:
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]
                prediction = logits.argmax(dim=-1).item()
                confidence = probs[prediction].item() * 100
            
            # Get predicted sneaker name
            predicted_sneaker = model.config.id2label[prediction]
            
            # Get all predictions sorted
            all_probs, all_indices = torch.sort(probs, descending=True)

            # time.sleep(0.3)  # Brief pause for effect

            # --- NEW: apply confidence threshold ----
            THRESHOLD = 85.0  # percent
            display_label = predicted_sneaker if confidence > THRESHOLD else "Other"

            # Update session state counts using display label
            st.session_state.predictions_made += 1
            st.session_state.sneaker_counts[display_label] = st.session_state.sneaker_counts.get(display_label, 0) + 1
            
            st.session_state.result = {
                'sneaker': display_label,
                'predicted_sneaker': predicted_sneaker,
                'confidence': confidence,
                'all_predictions': [(model.config.id2label[idx.item()], prob.item() * 100) 
                        for prob, idx in zip(all_probs, all_indices)],
                'image': image,
                'visualizations': {}
            }
            
            # Generate all visualizations
            visualizations = generate_visualizations(model, image, prediction, device, processor)
            if visualizations:
                st.session_state.result['visualizations'] = visualizations

    # Display Results
    if hasattr(st.session_state, 'result'):
        result = st.session_state.result
        
        # Choose gradient based on confidence
        if result['confidence'] >= 90:
            gradient = "linear-gradient(135deg, #000000 0%, #1a1a1a 100%)"
            status = "HIGHLY CONFIDENT"
        elif result['confidence'] >= 70:
            gradient = "linear-gradient(135deg, #1a1a1a 0%, #333333 100%)"
            status = "CONFIDENT"
        else:
            gradient = "linear-gradient(135deg, #333333 0%, #4d4d4d 100%)"
            status = "UNCERTAIN"
        
        # Main Result Card
        st.markdown(f"""
        <div class="result-card" style="background: {gradient}; color: white; margin: 3rem auto; max-width: 800px;">
            <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52" aria-hidden="true" role="img">
                <circle class="checkmark-circle" cx="26" cy="26" r="25" />
                <path class="checkmark-check" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
            </svg>
            <div style="font-size:2.5rem; font-weight:900; margin-bottom: 0.5rem; text-align: center; text-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                {result['sneaker']}
            </div>
            <div style="text-align: center;">
                <span class="badge" style="background: rgba(255,255,255,0.15); color: white;">{status}</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {result['confidence']:.0f}%;"></div>
            </div>
            <p style="font-size: 1.3rem; font-weight: 700; text-align: center; margin-top: 0.5rem;">
                Confidence: {result['confidence']:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Report Section
        st.markdown('<div style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
        
        # Visual Explanation as expandable section
        with st.expander("Visual Explanation: Why This Prediction?", expanded=False):
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">
                    See what the model focuses on when making decisions
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get visualizations
            vis = result.get('visualizations', {})
            
            if vis:
                # Row 1: Original, Attention Map, Gradient Map
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h3 style="font-size: 1.8rem; font-weight: 700; color: #ffffff;">
                        Visualization Overview
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Original Image
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(result['image'])
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Attention Map
                        </h4>
                        <p style="font-size: 0.85rem; color: #cccccc;">Where the model looks</p>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    if 'attention_raw' in vis:
                        st.image(vis['attention_raw'])
                    else:
                        st.info("Attention map not available")
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Gradient Map
                        </h4>
                        <p style="font-size: 0.85rem; color: #cccccc;">Most influential pixels</p>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    if 'gradient_raw' in vis:
                        st.image(vis['gradient_raw'])
                    else:
                        st.info("Gradient map not available")
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                # Row 2: Overlays
                st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h3 style="font-size: 1.8rem; font-weight: 700; color: #ffffff;">
                        Heatmap Overlays
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Attention Overlay
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    if 'attention' in vis:
                        st.image(vis['attention'])
                        st.markdown("""
                        <p style="text-align: center; color: #cccccc; font-size: 0.85rem; margin-top: 1rem;">
                            Transformer attention patterns
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Not available")
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Gradient Overlay
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    if 'gradient' in vis:
                        st.image(vis['gradient'])
                        st.markdown("""
                        <p style="text-align: center; color: #cccccc; font-size: 0.85rem; margin-top: 1rem;">
                            Gradient-based saliency
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Not available")
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <h4 style="font-size: 1.3rem; font-weight: 700; color: #ffffff;">
                            Combined Analysis
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)
                    #st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    if 'combined' in vis:
                        st.image(vis['combined'])
                        st.markdown("""
                        <p style="text-align: center; color: #cccccc; font-size: 0.85rem; margin-top: 1rem;">
                            Attention + Gradient fusion
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Not available")
                    #st.markdown('</div>', unsafe_allow_html=True)
                
                # Interpretation Guide
                st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.05); 
                            border-radius: 25px; padding: 2rem; border: 2px solid rgba(255, 255, 255, 0.1);">
                    <h3 style="text-align: center; font-size: 1.8rem; font-weight: 700; color: #ffffff; margin-bottom: 1.5rem;">
                        How to Interpret These Visualizations
                    </h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
                            <h4 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 0.5rem;">Attention Map</h4>
                            <p style="color: #cccccc; font-size: 0.95rem;">
                                Shows which image patches the Vision Transformer focuses on. 
                                Brighter areas = more attention from the model.
                            </p>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
                            <h4 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 0.5rem;">Gradient Map</h4>
                            <p style="color: #cccccc; font-size: 0.95rem;">
                                Highlights pixels that, if changed, would most affect the prediction. 
                                Red/hot colors = high importance.
                            </p>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
                            <h4 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 0.5rem;">Combined</h4>
                            <p style="color: #cccccc; font-size: 0.95rem;">
                                Merges attention and gradient information for a comprehensive view 
                                of what features drive the prediction.
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Visualizations could not be generated. This may be due to model configuration.")


# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #ffffff;">
    <p style="font-size: 0.9rem; font-weight: 600;">
        Nike Air Force 1 Low / Nike Air Jordan 1 High / Nike Dunk Low / Adidas Stan Smith / Adidas Superstar / Converse Chuck Taylor High / Vans Old Skool / New Balance 550 / Yeezy Boost 350 V2 / Nike Air Max 90
    </p>

</div>
""", unsafe_allow_html=True)