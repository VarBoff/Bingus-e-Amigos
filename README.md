
# Flood Watch

## Overview

This repository contains the solution developed by Bingus-e-Amigos for the NASA Space Apps Challenge 2023. Our project focuses on utilizing Convolutional Neural Networks (CNNs) to analyze satellite images of the state of Rio Grande do Sul (RS), Brazil, and predict whether or not a particular area is flooded.

### Key Features

- Convolutional Neural Network (CNN) for image analysis.
- Satellite image dataset specific to RS.
- Flood prediction model trained on labeled satellite images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- [Python](https://www.python.org/) (3.11)
- [PyTorch](https://pytorch.org/)

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/VarBoff/Bingus-e-Amigos.git
cd your-repo
```

2. **Install dependencies:**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Run the application:**

```bash
python ./CNN.py
```

## Data

The data used for training the model can be found in the data folder, where it has been segmented into train and test, and then further segmented into 3 labels: 'adequete', 'flooded' and 'no-data'.
We also kept a folder containing the complete dataset.

## Model

The Convolutional Neural Network (CNN) architecture used for flood prediction is defined in `CNN.py`. The model has been trained on ~30 images and achieved an accuracy of 68.3% on the test set.

## License

This project is licensed under the [MIT License](LICENSE).
