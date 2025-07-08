# Project Title

## Overview
This repository contains the main Python script `VAN_MU.py` for processing and analyzing the dataset. The dataset files are stored in compressed split archives due to their large size.

## Dataset Information
The primary dataset used in this project is split into multiple compressed volumes:
- `without coupling.7z.001`
- `without coupling.7z.002`
- `without coupling.7z.003`
- `without coupling.7z.004`
- `without coupling.7z.005`
- `without coupling.7z.006`

### Important Notes:
1. **Complete Dataset Requirement**: 
   - **All split volumes must be downloaded together** to successfully extract the full dataset
   - The dataset is essential for reproducing Figure 2 and other results in the project

2. **Extraction Instructions**:
   ```bash
   # On Windows: Select all files -> Right-click -> Extract All
   # On Linux/macOS:
   zip -s 0 without coupling.7z --out full_dataset.7z
   unzip full_dataset.7z
