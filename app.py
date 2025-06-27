import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Your model classes (with fixed discriminator)
class UpConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, dropout=0.0):
        super(UpConvBlock, self).__init__()
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, 4, 2, 1),
            nn.InstanceNorm2d(op_sz),
            nn.ReLU(),
        ]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
    
    def forward(self, x, enc_ip):
        x = nn.Sequential(*(self.layers))(x)
        op = torch.cat((x, enc_ip), 1)
        return op

class DownConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, norm=True, dropout=0.0):
        super(DownConvBlock, self).__init__()
        self.layers = [nn.Conv2d(ip_sz, op_sz, 4, 2, 1)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
    
    def forward(self, x):
        op = nn.Sequential(*(self.layers))(x)
        return op

class UNetGenerator(nn.Module):
    def __init__(self, chnls_in=3, chnls_op=3):
        super(UNetGenerator, self).__init__()
        self.down_conv_layer_1 = DownConvBlock(chnls_in, 64, norm=False)
        self.down_conv_layer_2 = DownConvBlock(64, 128)
        self.down_conv_layer_3 = DownConvBlock(128, 256)
        self.down_conv_layer_4 = DownConvBlock(256, 512, dropout=0.5)
        self.down_conv_layer_5 = DownConvBlock(512, 512, dropout=0.5)
        self.down_conv_layer_6 = DownConvBlock(512, 512, dropout=0.5)
        self.down_conv_layer_7 = DownConvBlock(512, 512, dropout=0.5)
        self.down_conv_layer_8 = DownConvBlock(512, 512, norm=False, dropout=0.5)
        self.up_conv_layer_1 = UpConvBlock(512, 512, dropout=0.5)
        self.up_conv_layer_2 = UpConvBlock(1024, 512, dropout=0.5)
        self.up_conv_layer_3 = UpConvBlock(1024, 512, dropout=0.5)
        self.up_conv_layer_4 = UpConvBlock(1024, 512, dropout=0.5)
        self.up_conv_layer_5 = UpConvBlock(1024, 256)
        self.up_conv_layer_6 = UpConvBlock(512, 128)
        self.up_conv_layer_7 = UpConvBlock(256, 64)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv_layer_1 = nn.Conv2d(128, chnls_op, 4, padding=1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        enc1 = self.down_conv_layer_1(x)
        enc2 = self.down_conv_layer_2(enc1)
        enc3 = self.down_conv_layer_3(enc2)
        enc4 = self.down_conv_layer_4(enc3)
        enc5 = self.down_conv_layer_5(enc4)
        enc6 = self.down_conv_layer_6(enc5)
        enc7 = self.down_conv_layer_7(enc6)
        enc8 = self.down_conv_layer_8(enc7)
        dec1 = self.up_conv_layer_1(enc8, enc7)
        dec2 = self.up_conv_layer_2(dec1, enc6)
        dec3 = self.up_conv_layer_3(dec2, enc5)
        dec4 = self.up_conv_layer_4(dec3, enc4)
        dec5 = self.up_conv_layer_5(dec4, enc3)
        dec6 = self.up_conv_layer_6(dec5, enc2)
        dec7 = self.up_conv_layer_7(dec6, enc1)
        final = self.upsample_layer(dec7)
        final = self.zero_pad(final)
        final = self.conv_layer_1(final)
        return self.activation(final)

class Pix2PixDiscriminator(nn.Module):
    def __init__(self, chnls_in=3):
        super(Pix2PixDiscriminator, self).__init__()
        
        def disc_conv_block(chnls_in, chnls_op, norm=True):
            layers = [nn.Conv2d(chnls_in, chnls_op, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(chnls_op))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.lyr1 = disc_conv_block(chnls_in * 2, 64, norm=False)
        self.lyr2 = disc_conv_block(64, 128)
        self.lyr3 = disc_conv_block(128, 256)
        self.lyr4 = disc_conv_block(256, 512)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.final_conv = nn.Conv2d(512, 1, 4, padding=1)
    
    def forward(self, real_image, translated_image):
        ip = torch.cat((real_image, translated_image), 1)
        op = self.lyr1(ip)
        op = self.lyr2(op)
        op = self.lyr3(op)
        op = self.lyr4(op)
        op = self.zero_pad(op)
        op = self.final_conv(op)
        return op

# Global model variable
generator = None

def load_model(model_file, model_type):
    """Load the Pix2Pix generator model"""
    global generator
    try:
        generator = UNetGenerator(chnls_in=3, chnls_op=3)
        
        if model_file is not None:
            # Load custom model
            model_state = torch.load(model_file.name, map_location='cpu')
            generator.load_state_dict(model_state)
            status = f"✅ Custom model loaded successfully!"
        else:
            # Use randomly initialized model for demo
            status = f"⚠️ Using randomly initialized model (upload a trained model for real results)"
        
        generator.eval()
        return status
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def transform_image(input_image, model_file, model_type, image_size):
    """Transform input image using Pix2Pix model"""
    if input_image is None:
        return None, "Please upload an image first!"
    
    try:
        # Load model if not already loaded or if new model uploaded
        if generator is None or model_file is not None:
            status = load_model(model_file, model_type)
        else:
            status = "Using previously loaded model"
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Convert PIL to tensor
        input_tensor = transform(input_image).unsqueeze(0)
        
        # Generate output
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Post-process output
        output_tensor = (output_tensor + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # Convert back to PIL Image
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
        
        return output_image, status
        
    except Exception as e:
        return None, f"❌ Error processing image: {str(e)}"

def create_demo():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #FF6B35, #F7931E) !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        transform: scale(1.05) !important;
        transition: all 0.2s !important;
    }
    """
    
    with gr.Blocks(css=css, title="Pix2Pix Image Translation", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Pix2Pix Image Translation
            Transform your images using conditional GANs! Upload an image and watch the magic happen ✨
            """,
            elem_id="header"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")
                input_image = gr.Image(
                    type="pil", 
                    label="Upload your image",
                    height=300
                )
                
                # Model configuration
                gr.Markdown("### ⚙️ Configuration")
                model_type = gr.Dropdown(
                    choices=["Sketch to Photo", "Day to Night", "Satellite to Map", "Custom Model"],
                    value="Sketch to Photo",
                    label="Transformation Type"
                )
                
                model_file = gr.File(
                    label="Upload Custom Model (.pth/.pt)",
                    file_types=[".pth", ".pt"],
                    visible=False
                )
                
                image_size = gr.Slider(
                    minimum=128,
                    maximum=512,
                    step=64,
                    value=256,
                    label="Image Size"
                )
                
                # Transform button
                transform_btn = gr.Button(
                    "🚀 Transform Image", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📤 Output")
                output_image = gr.Image(
                    label="Transformed Image",
                    height=300
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3
                )
        
        # Show/hide custom model upload based on selection
        def update_model_visibility(choice):
            return gr.update(visible=(choice == "Custom Model"))
        
        model_type.change(
            update_model_visibility,
            inputs=[model_type],
            outputs=[model_file]
        )
        
        # Transform button click event
        transform_btn.click(
            transform_image,
            inputs=[input_image, model_file, model_type, image_size],
            outputs=[output_image, status_text]
        )
        
        # Information tabs
        with gr.Tabs():
            with gr.Tab("ℹ️ About Pix2Pix"):
                gr.Markdown(
                    """
                    **Pix2Pix** is a conditional Generative Adversarial Network (cGAN) that learns to map from input images to output images.
                    
                    ### How it works:
                    - **U-Net Generator**: Uses skip connections to preserve fine details
                    - **PatchGAN Discriminator**: Focuses on local image patches for realistic textures
                    - **Loss Function**: Combines adversarial loss with L1 loss for pixel-level accuracy
                    
                    ### Common Applications:
                    - 🎨 Sketch to photorealistic images
                    - 🌅 Day to night scene conversion
                    - 🗺️ Satellite imagery to maps
                    - 🎨 Image colorization
                    - 🏠 Architectural sketches to renderings
                    
                    ### Model Architecture:
                    - **Input/Output**: 3-channel RGB images
                    - **Generator**: U-Net with 8 encoder and 7 decoder layers
                    - **Discriminator**: PatchGAN with 4 convolutional layers
                    """
                )
            
            with gr.Tab("🏋️ Training Guide"):
                gr.Markdown(
                    """
                    ### Training Your Own Pix2Pix Model
                    
                    **1. Data Preparation:**
                    ```python
                    # Prepare paired training data (input-output image pairs)
                    # Images should be aligned and of same size
                    dataset_structure/
                    ├── train/
                    │   ├── A/  # Input images
                    │   └── B/  # Target images
                    └── test/
                        ├── A/
                        └── B/
                    ```
                    
                    **2. Training Loop:**
                    ```python
                    # Loss functions
                    criterion_GAN = nn.MSELoss()
                    criterion_L1 = nn.L1Loss()
                    lambda_L1 = 100  # L1 loss weight
                    
                    # Training
                    for epoch in range(num_epochs):
                        for real_A, real_B in dataloader:
                            # Train Generator
                            fake_B = generator(real_A)
                            pred_fake = discriminator(real_A, fake_B)
                            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                            loss_L1 = criterion_L1(fake_B, real_B)
                            loss_G = loss_GAN + lambda_L1 * loss_L1
                            
                            # Train Discriminator
                            pred_real = discriminator(real_A, real_B)
                            pred_fake = discriminator(real_A, fake_B.detach())
                            loss_D = (criterion_GAN(pred_real, torch.ones_like(pred_real)) + 
                                     criterion_GAN(pred_fake, torch.zeros_like(pred_fake))) * 0.5
                    ```
                    
                    **3. Training Tips:**
                    - Use paired training data for best results
                    - Train for 100-200 epochs typically
                    - Use learning rate scheduling
                    - Monitor both G and D losses
                    - Save model checkpoints regularly
                    
                    **4. Save Your Model:**
                    ```python
                    torch.save(generator.state_dict(), 'pix2pix_generator.pth')
                    ```
                    """
                )
            
            with gr.Tab("📝 Examples"):
                gr.Markdown(
                    """
                    ### Popular Pix2Pix Applications
                    
                    **🎨 Sketch to Photo:**
                    - Input: Hand-drawn sketches or edge maps
                    - Output: Photorealistic images
                    - Use case: Art creation, concept visualization
                    
                    **🌆 Day to Night:**
                    - Input: Daytime cityscape photos
                    - Output: Nighttime scenes with lighting
                    - Use case: Architectural visualization, film production
                    
                    **🛰️ Satellite to Map:**
                    - Input: Satellite imagery
                    - Output: Google Maps style images
                    - Use case: Cartography, urban planning
                    
                    **🎨 Colorization:**
                    - Input: Grayscale images
                    - Output: Colorized versions
                    - Use case: Historical photo restoration
                    
                    ### Tips for Best Results:
                    - Use high-quality, well-aligned training pairs
                    - Ensure consistent lighting and style in training data
                    - Train for sufficient epochs (patience is key!)
                    - Experiment with different loss weights
                    """
                )
        
        gr.Markdown(
            """
            ---
            💡 **Note**: This demo uses a randomly initialized model for demonstration. 
            Upload your own trained model for real transformations!
            """,
            elem_id="footer"
        )
    
    return demo

# Initialize model
generator = UNetGenerator(chnls_in=3, chnls_op=3)
generator.eval()

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,  # Set to True for public sharing
        debug=True,
        show_error=True
    )
