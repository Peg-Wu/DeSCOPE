## 🧬 scATAC Perturbation Prediction

### 🔥 Execution Flow

Follow these steps to reproduce the results:

1. **Data Tokenization**  
   Run [`tokenize.ipynb`](./tokenize.ipynb) to convert your **.h5ad** files into a **Hugging Face Dataset (Arrow format)**.
   
2. **Model Training**  
   Execute one of the following bash scripts to start training.  
   > 💡 *Tip: You can configure data paths, model architectures, and training parameters directly within these scripts.*
   
   - **Standard Training**:  
     
     ```bash
     sh train_descope.sh        # Trains the DeSCOPE model
     ```
   - **Leave-One-Out Training**:  
     ```bash
     sh train_descope_loo.sh    # Trains the DeSCOPE_LOO model
     ```
   
3. **Inference & Evaluation**  
   Run [`test.ipynb`](./test.ipynb) to perform inference using the trained checkpoint and compute evaluation metrics.