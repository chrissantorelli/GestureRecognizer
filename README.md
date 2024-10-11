# Steps to Set Up and Run the Gesture Recognition Flask App Using Anaconda

## Prerequisites

- **Anaconda or Miniconda Installed**
- **Python 3.9**
- **Webcam**
- **Web Browser:** Latest version of Chrome, Firefox, or Edge

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/chrissantorelli/GestureRecognizer.git
cd GestureRecognizer
```
### 2. Create a Conda Environment
Create a new Conda environment named gesture_env with Python 3.9:
```bash
conda create -n gesture_env python=3.9
```
### 3. Activate Conda Environment
```bash
conda activate gesture_env
```
### 4. Upgrade pip
Before installing the requirements, ensure that pip is up to date:
```bash
python -m pip install --upgrade pip
```
### 5. Install Dependencies
Install the required packages using pip within your Conda environment:
```bash
pip install -r requirements.txt
```