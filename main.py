import streamlit as st
from ollama import chat as ollama_chat
from groq import Groq
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from PIL import Image
import io
import os
from dotenv import load_dotenv
import base64
import time
import google.generativeai as genai

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ImageAnalysis(BaseModel):
    """Unified structure for image analysis results"""
    description: str
    key_points: List[str]
    detected_objects: List[str]
    detected_text: Optional[str] = None

def analyze_image_ollama(image_path: str) -> ImageAnalysis:
    """Analyze image using Ollama's vision model"""
    try:
        response = ollama_chat(
            model='llama3.2-vision',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a precise image analysis system.'
                },
                {
                    'role': 'user',
                    'content': 'Analyze this image and provide: 1) A description 2) Key points 3) List of objects 4) Any text detected',
                    'images': [image_path]
                }
            ],
            options={'temperature': 0}
        )

        # Process Ollama's response into structured format
        content = response.message.content
        
        # Basic parsing of the response
        sections = content.split('\n\n')
        description = sections[0] if sections else "No description available"
        key_points = [point.strip('- ') for point in content.split('\n') if point.startswith('-')]
        
        # Extract objects and text if mentioned
        objects = []
        detected_text = None
        
        for section in sections:
            if 'object' in section.lower():
                objects = [obj.strip('- ') for obj in section.split('\n') if obj.strip('- ')]
            if 'text' in section.lower():
                detected_text = section.split(':')[-1].strip()

        return ImageAnalysis(
            description=description,
            key_points=key_points if key_points else ["No key points identified"],
            detected_objects=objects if objects else ["No objects specifically identified"],
            detected_text=detected_text
        )
    except Exception as e:
        raise Exception(f"Error in Ollama image analysis: {str(e)}")

def analyze_image_groq(image_path: str, groq_client: Groq) -> ImageAnalysis:
    """Analyze image using Groq's vision model"""
    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        
        image_data_url = f"data:image/jpeg;base64,{image_data}"
        
        # First message to get description and overview
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a detailed description of this image. Focus on what you see and any notable aspects."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        initial_description = completion.choices[0].message.content
        
        # Second message to get specific details
        completion_details = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List the following for this image:\n1. Key objects present\n2. Any text visible in the image\n3. Important observations"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        details_content = completion_details.choices[0].message.content
        
        # Process the responses
        key_points = []
        objects = []
        detected_text = None
        
        # Extract information from the details response
        for line in details_content.split('\n'):
            line = line.strip('*- ')
            if line.lower().startswith(('object', 'present', 'see', 'contain')):
                objects.append(line)
            elif line.lower().startswith(('text', 'word', 'written')):
                detected_text = line
            elif len(line) > 10:  # Arbitrary length to filter out headers
                key_points.append(line)
        
        # Clean up the lists
        objects = [obj for obj in objects if len(obj) > 0]
        key_points = [point for point in key_points if len(point) > 0]
        
        # Create structured output
        return ImageAnalysis(
            description=initial_description,
            key_points=key_points if key_points else ["No specific key points identified"],
            detected_objects=objects if objects else ["No specific objects identified"],
            detected_text=detected_text
        )
        
    except Exception as e:
        raise Exception(f"Error in Groq image analysis: {str(e)}")

def analyze_image_gemini(image_path: str, temperature: float = 0.2) -> ImageAnalysis:
    """Analyze image using Google's Gemini model"""
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Model configuration
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Initialize model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-1219",
            generation_config=generation_config,
        )

        # Upload and analyze image
        image_file = genai.upload_file(image_path, mime_type="image/jpeg")
        
        # Start chat session with specific prompt
        chat = model.start_chat(history=[])
        response = chat.send_message([
            image_file,
            """Analyze this image and provide the following in a clear structure:
            1. A detailed description
            2. Key points or observations
            3. List of detected objects
            4. Any text visible in the image
            
            Please be specific and detailed in your analysis."""
        ])

        # Parse response into sections
        content = response.text
        sections = content.split('\n\n')
        
        # Basic parsing of the response
        description = sections[0] if sections else "No description available"
        
        # Extract key points, objects, and text
        key_points = []
        objects = []
        detected_text = None
        
        for section in sections:
            if 'key point' in section.lower() or 'observation' in section.lower():
                key_points.extend([point.strip('- ') for point in section.split('\n') if point.strip('- ')])
            elif 'object' in section.lower():
                objects.extend([obj.strip('- ') for obj in section.split('\n') if obj.strip('- ')])
            elif 'text' in section.lower():
                detected_text = section.split(':')[-1].strip()

        return ImageAnalysis(
            description=description,
            key_points=key_points if key_points else ["No key points identified"],
            detected_objects=objects if objects else ["No objects specifically identified"],
            detected_text=detected_text
        )
        
    except Exception as e:
        raise Exception(f"Error in Gemini image analysis: {str(e)}")

def main():
    st.set_page_config(
        page_title="Image Analysis Comparison",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Custom CSS for consistent styling
    st.markdown("""
        <style>
        .small-font {
            font-size: 14px !important;
            margin: 0px !important;
            padding: 0px !important;
        }
        .sidebar .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        .results-container {
            padding: 1rem;
            margin-top: 1rem;
        }
        .metrics-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar configuration and user actions
    with st.sidebar:
        # Branding
        st.title("🖼️ Image Analyzer")
        st.markdown("<p class='small-font'>Compare image analysis across different AI models</p>", 
                   unsafe_allow_html=True)
        
        st.divider()
        
        # Configuration section
        st.markdown("### Configuration")
        provider = st.selectbox(
            "Select Provider",
            ["ollama", "groq", "gemini"],
            help="Choose between local (Ollama) or cloud (Groq/Gemini) processing"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Higher values make the output more creative, lower values make it more focused"
            )
            max_tokens = st.slider(
                "Max Tokens",
                min_value=256,
                max_value=2048,
                value=1024,
                step=256,
                help="Maximum number of tokens in the response"
            )
        
        # Check for required API keys
        if provider == "groq" and not GROQ_API_KEY:
            st.error("Groq API key not found. Please add GROQ_API_KEY to your .env file")
            return
        elif provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            st.error("Gemini API key not found. Please add GEMINI_API_KEY to your .env file")
            return

        st.divider()
        
        # Image upload section
        st.markdown("### Upload Image")
        image_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        analyze_button = st.button("Analyze Image", type="primary", disabled=not image_file)

    # Main content area - Results only
    if image_file:
        # Show small preview of uploaded image in sidebar
        with st.sidebar:
            st.markdown("### Preview")
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        if analyze_button:
            try:
                start_time = time.time()
                with st.spinner(f"Analyzing image with {provider.upper()}..."):
                    # Save image temporarily
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format=image.format)
                        tmp.write(img_byte_arr.getvalue())
                        tmp_path = tmp.name

                    # Analyze based on selected provider
                    if provider == "ollama":
                        analysis = analyze_image_ollama(tmp_path)
                    elif provider == "groq":
                        groq_client = Groq(api_key=GROQ_API_KEY)
                        analysis = analyze_image_groq(tmp_path, groq_client)
                    else:  # gemini
                        analysis = analyze_image_gemini(tmp_path, temperature)

                    processing_time = time.time() - start_time

                    # Display performance metrics
                    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric(
                            "Processing Time",
                            f"{processing_time:.2f}s"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            "Provider",
                            provider.upper()
                        )
                    
                    with metrics_col3:
                        st.metric(
                            "Temperature",
                            temperature
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Display results in main area with consistent styling
                    st.markdown("## Analysis Results")
                    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
                    
                    st.markdown("#### Description")
                    st.markdown(f"<p class='small-font'>{analysis.description}</p>", 
                              unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Key Points")
                        for point in analysis.key_points:
                            st.markdown(f"<p class='small-font'>• {point}</p>", 
                                      unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### Detected Objects")
                        for obj in analysis.detected_objects:
                            st.markdown(f"<p class='small-font'>• {obj}</p>", 
                                      unsafe_allow_html=True)
                    
                    if analysis.detected_text:
                        st.markdown("#### Detected Text")
                        st.markdown(f"<p class='small-font'>{analysis.detected_text}</p>", 
                                  unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Cleanup
                    Path(tmp_path).unlink()

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    else:
        # Empty state message in main area
        st.markdown(
            """
            <div style='text-align: center; padding: 50px;'>
                <p class='small-font'>👈 Upload an image from the sidebar to begin analysis</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
