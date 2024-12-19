# img2img-turbo (CycleGan-Turbo | Style transfer)
(Tested on Python 3.8.10 | cuda 12.1 | 24GB VRAM - A good GPU is needed to run the training)

This repository implements CycleGAN-Turbo for unpaired image-to-image translation. It enables transforming images between a source and target domain while maintaining important features and spatial consistency. This approach is useful for tasks like style transfer/domain adaptation (e.g.: transforming synthetic images to look more realistic).

1. **Create a virtual environment**:
   <details>
      
   First, create and activate a virtual environment inside the cloned img2img-turbo dir:
   
   ```bash
   python3 -m venv cycle_venv
   source cycle_venv/bin/activate
   ```
   </details>
2. **Install rust/cargo and networkx into your cycle_venv**:
   <details>
      
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   export PATH="$HOME/.cargo/bin:$PATH"
   source ~/.bashrc
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> cycle_venv/bin/activate
   source cycle_venv/bin/activate
   pip install networkx
   ```
   Verify that rust/cargo is installed:
   
   ```bash
   rustc --version
   cargo --version
   ```
   </details>
   
3. **Install the correct torch version for your cuda**:
   <details>
   
   Change the last part ("cu121") depending on your system cuda version (I use cuda 12.1)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    
   ```
   </details>
   
4. **Install the required dependencies**:
   <details>
      
   ```bash
   pip install -r requirements.txt
   ```
   This will install all the required packages to run the scripts.

   If not done yet, make sure you are logged into huggingface so the pretrained model can be downloaded automatically:
   ```bash
   huggingface-cli login
   ```
   Now set up the accelerate configs:
   ```bash
   accelerate config
   ```
   or if you don't want to choose the setup:
   ```bash
   accelerate config default
   ```
   You can experiment with using fp16 (mixed precision) for the training which should speed it up, but the script doesn't seem to be optimized to use fp16 so it might slow it down instead.

   </details>
5. **Prepare your data and train the model**:
   <details>
      
   Make a `data/` folder in the root. Add a new folder for your case e.g. `weeds/`. Place your synthetic images here in a folder named 'synthetic' and your real images in a folder named 'real'. 
   Then run the script below to start the patching and training. You will be asked some prompts to answer (in general I train for 25 000 epochs). 

   ```bash
   prepare_and_train_cyclegan.sh
   ```
   </details>

6. **Run inference**:
   <details>
   
   After training has completed (takes about a day depending on your GPU), run the inference script (see command below) on the original resolution images folder (will now be named 'train_A_original'). 
   If the image has a high resolution and you do not have a good GPU you can run it on the patched folder ('train_A') instead and run the `inference_and_repatch.sh` script. 
   The repatching also applies blending to reduce the visibility of the separate patches but there will still be some artifacts because each patch is augmented separately. 
   In general I would recommend to do the inference on the original resolution images if your GPU allows it. 

   ```bash
   inference.sh
   ```

   The results will be in outputs/*your folder name*

   </details>

# General info about the style transfer and examples:
<details>
   
[**Paper**](https://arxiv.org/abs/2403.12036) | [**Sketch2Image Demo**](https://huggingface.co/spaces/gparmar/img2img-turbo-sketch) 
#### **Quick start:** [**Running Locally**](#getting-started) | [**Gradio (locally hosted)**](#gradio-demo) | [**Training**](#training-with-your-own-data)

### Cat Sketching
<p align="left" >
<img src="https://raw.githubusercontent.com/GaParmar/img2img-turbo/main/assets/cat_2x.gif" width="800" />
</p>

### Fish Sketching
<p align="left">
<img src="https://raw.githubusercontent.com/GaParmar/img2img-turbo/main/assets/fish_2x.gif"  width="800" />
</p>


We propose a general method for adapting a single-step diffusion model, such as SD-Turbo, to new tasks and domains through adversarial learning. This enables us to leverage the internal knowledge of pre-trained diffusion models while achieving efficient inference (e.g., for 512x512 images, 0.29 seconds on A6000 and 0.11 seconds on A100). 

Our one-step conditional models **CycleGAN-Turbo** and **pix2pix-turbo** can perform various image-to-image translation tasks for both unpaired and paired settings. CycleGAN-Turbo outperforms existing GAN-based and diffusion-based methods, while pix2pix-turbo is on par with recent works such as ControlNet for Sketch2Photo and Edge2Image, but with one-step inference. 

[One-Step Image Translation with Text-to-Image Models](https://arxiv.org/abs/2403.12036)<br>
[Gaurav Parmar](https://gauravparmar.com/), [Taesung Park](https://taesung.me/), [Srinivasa Narasimhan](https://www.cs.cmu.edu/~srinivas/), [Jun-Yan Zhu](https://github.com/junyanz/)<br>
CMU and Adobe, arXiv 2403.12036

<br>
<div>
<p align="center">
<img src='assets/teaser_results.jpg' align="center" width=1000px>
</p>
</div>




## Results

### Paired Translation with pix2pix-turbo
**Edge to Image**
<div>
<p align="center">
<img src='assets/edge_to_image_results.jpg' align="center" width=800px>
</p>
</div>

<!-- **Sketch to Image**
TODO -->
### Generating Diverse Outputs
By varying the input noise map, our method can generate diverse outputs from the same input conditioning.
The output style can be controlled by changing the text prompt.
<div> <p align="center">
<img src='assets/gen_variations.jpg' align="center" width=800px>
</p> </div>

### Unpaired Translation with CycleGAN-Turbo

**Day to Night**
<div> <p align="center">
<img src='assets/day2night_results.jpg' align="center" width=800px>
</p> </div>

**Night to Day**
<div><p align="center">
<img src='assets/night2day_results.jpg' align="center" width=800px>
</p> </div>

**Clear to Rainy**
<div>
<p align="center">
<img src='assets/clear2rainy_results.jpg' align="center" width=800px>
</p>
</div>

**Rainy to Clear**
<div>
<p align="center">
<img src='assets/rainy2clear.jpg' align="center" width=800px>
</p>
</div>
<hr>


## Method
**Our Generator Architecture:**
We tightly integrate three separate modules in the original latent diffusion models into a single end-to-end network with small trainable weights. This architecture allows us to translate the input image x to the output y, while retaining the input scene structure. We use LoRA adapters in each module, introduce skip connections and Zero-Convs between input and output, and retrain the first layer of the U-Net. Blue boxes indicate trainable layers. Semi-transparent layers are frozen. The same generator can be used for various GAN objectives.
<div>
<p align="center">
<img src='assets/method.jpg' align="center" width=900px>
</p>
</div>


## Getting Started
**Environment Setup**
- We provide a [conda env file](environment.yml) that contains all the required dependencies.
    ```
    conda env create -f environment.yaml
    ```
- Following this, you can activate the conda environment with the command below. 
  ```
  conda activate img2img-turbo
  ```
- Or use virtual environment:
  ```
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
**Paired Image Translation (pix2pix-turbo)**
- The following command takes an image file and a prompt as inputs, extracts the canny edges, and saves the results in the directory specified.
    ```bash
    python src/inference_paired.py --model_name "edge_to_image" \
        --input_image "assets/examples/bird.png" \
        --prompt "a blue bird" \
        --output_dir "outputs"
    ```
    <table>
    <th>Input Image</th>
    <th>Canny Edges</th>
    <th>Model Output</th>
    </tr>
    <tr>
    <td><img src='assets/examples/bird.png' width="200px"></td>
    <td><img src='assets/examples/bird_canny.png' width="200px"></td>
    <td><img src='assets/examples/bird_canny_blue.png' width="200px"></td>
    </tr>
    </table>
    <br>

- The following command takes a sketch and a prompt as inputs, and saves the results in the directory specified.
    ```bash
    python src/inference_paired.py --model_name "sketch_to_image_stochastic" \
    --input_image "assets/examples/sketch_input.png" --gamma 0.4 \
    --prompt "ethereal fantasy concept art of an asteroid. magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy" \
    --output_dir "outputs"
    ```
    <table>
    <th>Input</th>
    <th>Model Output</th>
    </tr>
    <tr>
    <td><img src='assets/examples/sketch_input.png' width="400px"></td>
    <td><img src='assets/examples/sketch_output.png' width="400px"></td>
    </tr>
    </table>
    <br>

**Unpaired Image Translation (CycleGAN-Turbo)**
- The following command takes a **day** image file as input, and saves the output **night** in the directory specified.
    ```
    python src/inference_unpaired.py --model_name "day_to_night" \
        --input_image "assets/examples/day2night_input.png" --output_dir "outputs"
    ```
    <table>
    <th>Input (day)</th>
    <th>Model Output (night)</th>
    </tr>
    <tr>
    <td><img src='assets/examples/day2night_input.png' width="400px"></td>
    <td><img src='assets/examples/day2night_output.png' width="400px"></td>
    </tr>
    </table>

- The following command takes a **night** image file as input, and saves the output **day** in the directory specified.
    ```
    python src/inference_unpaired.py --model_name "night_to_day" \
        --input_image "assets/examples/night2day_input.png" --output_dir "outputs"
    ```
    <table>
    <th>Input (night)</th>
    <th>Model Output (day)</th>
    </tr>
    <tr>
    <td><img src='assets/examples/night2day_input.png' width="400px"></td>
    <td><img src='assets/examples/night2day_output.png' width="400px"></td>
    </tr>
    </table>

- The following command takes a **clear** image file as input, and saves the output **rainy** in the directory specified.
    ```
    python src/inference_unpaired.py --model_name "clear_to_rainy" \
        --input_image "assets/examples/clear2rainy_input.png" --output_dir "outputs"
    ```
    <table>
    <th>Input (clear)</th>
    <th>Model Output (rainy)</th>
    </tr>
    <tr>
    <td><img src='assets/examples/clear2rainy_input.png' width="400px"></td>
    <td><img src='assets/examples/clear2rainy_output.png' width="400px"></td>
    </tr>
    </table>

- The following command takes a **rainy** image file as input, and saves the output **clear** in the directory specified.
    ```
    python src/inference_unpaired.py --model_name "rainy_to_clear" \
        --input_image "assets/examples/rainy2clear_input.png" --output_dir "outputs"
    ```
    <table>
    <th>Input (rainy)</th>
    <th>Model Output (clear)</th>
    </tr>
    <tr>
    <td><img src='assets/examples/rainy2clear_input.png' width="400px"></td>
    <td><img src='assets/examples/rainy2clear_output.png' width="400px"></td>
    </tr>
    </table>



## Gradio Demo
- We provide a Gradio demo for the paired image translation tasks.
- The following command will launch the sketch to image locally using gradio.
    ```
    gradio gradio_sketch2image.py
    ```
- The following command will launch the canny edge to image gradio demo locally.
   ```
    gradio gradio_canny2image.py
   ```


## Training with your own data
- See the steps [here](docs/training_pix2pix_turbo.md) for training a pix2pix-turbo model on your paired data.
- See the steps [here](docs/training_cyclegan_turbo.md) for training a CycleGAN-Turbo model on your unpaired data.


## Acknowledgment
Our work uses the Stable Diffusion-Turbo as the base model with the following [LICENSE](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE).

</details>
