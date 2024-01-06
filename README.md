# RIW

## Setting Up and Running the Experiment
If you want to quickly try out our RIW method after setting up the environment, I suggest you can run 'python run.py'.

### 1. Environment Setup
To run this experiment, it's highly recommended to set up the initial environment following the [instruct-pix2pix GitHub project](https://github.com/timothybrooks/instruct-pix2pix). This ensures compatibility and smooth operation.

(The main issue in setting up the environment is the configuration of Stable Diffusion. The project mentioned above can help you set up the Diffusion editing part, and you will also obtain configuration for the encoder/decoder of the Diffusion model, which will be very helpful. The encoder/decoder is needed in the process of adding watermarks. Additionally, there are no other special problems to solve in the environment configuration for adding watermarks (RIW).")

### 2. Preparing the Data
Once you have set up the **stable diffusion** environment, it's advised to download the dataset from the link provided in this documentation. After downloading, replace `"DIR/"` in the code with your actual dataset path. This step ensures that the experiment can locate and use your dataset seamlessly.

### 3. Data Preprocessing
With everything in place, run the `prepare.py` script first. This script processes the data and prepares the images by adding watermarks. This step is crucial to ensure the quality and integrity of the data being used in the experiment.

### 4. Image Editing
For editing the images, use the `cli_edit.py` script. It provides tools and functions tailored for this purpose, allowing you to modify the images in the dataset according to the needs of the experiment.

An example regarding RIW: when you increase the alpha parameter or the encoder loss, the watermark will become visible in the edited image. For examples where the watermark remains invisible after editing, please refer to the paper.
![example image](imgs/target.png)

---

**Note**: Before running any scripts, ensure that all the necessary dependencies are installed and your Python environment is correctly configured. If you encounter any issues, refer to the troubleshooting section or open an issue in this repository.
