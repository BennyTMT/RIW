## Setting Up and Running the Experiment

### 1. Environment Setup
To run this experiment, it's highly recommended to set up the initial environment following the [instruct-pix2pix GitHub project](https://github.com/YOUR_GITHUB_LINK/instruct-pix2pix). This ensures compatibility and smooth operation.

### 2. Preparing the Data
Once you have set up the **stable diffusion** environment, it's advised to download the dataset from the link provided in this documentation. After downloading, replace `"DIR/"` in the code with your actual dataset path. This step ensures that the experiment can locate and use your dataset seamlessly.

### 3. Data Preprocessing
With everything in place, run the `prepare.py` script first. This script processes the data and prepares the images by adding watermarks. This step is crucial to ensure the quality and integrity of the data being used in the experiment.

### 4. Image Editing
For editing the images, use the `cli_edit.py` script. It provides tools and functions tailored for this purpose, allowing you to modify the images in the dataset according to the needs of the experiment.

---

**Note**: Before running any scripts, ensure that all the necessary dependencies are installed and your Python environment is correctly configured. If you encounter any issues, refer to the troubleshooting section or open an issue in this repository.
