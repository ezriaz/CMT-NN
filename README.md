# CMT-NN

## Overview
This repository contains the main Python script `VAN_MU.py` for training the dataset. The dataset files are stored in compressed split archives (7z format) due to their large size.

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
   - **All 6 split volumes must be downloaded together** to successfully extract the full dataset
   - The dataset is essential for reproducing Figure 2 and other results in the project
   - Total uncompressed dataset size: 133MB
2. **Extraction Instructions**:
   - **Windows**:
     1. Install [7-Zip](https://www.7-zip.org/) if not already installed
     2. Select all 6 files in File Explorer
     3. Right-click → 7-Zip → "Extract to without coupling\"
   
   - **Linux/macOS**:
     ```bash
     # Install p7zip if needed:
     # Ubuntu/Debian: sudo apt install p7zip-full
     # macOS: brew install p7zip
     
     7z x "without coupling.7z.001"
     ```

## Repository Structure
- `VAN_MU.py` - Main neural network training script
- `without coupling/` - Extracted dataset directory (created after extraction)
