# CMCS
Enhancing LLM-based Code Generation via Cross-Model Collaboration

**The files of the CMCS framework have not been properly organized yet, and we are continuously updating it......**

1. Process SPoC files in "SPoCProcess".
2. The CMCS framework running on SPoC is in the file "CMCS on SPoC dataset".
3. The CMCS framework running on HumanEval is in the file "CMCS on HumanEval dataset".
4. The CMCS framework running on MBPP is in the file "CMCS on MBPP dataset".

## CMCS on HumanEval dataset

**Download the HumanEval dataset locally:**
```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

**Run CMCS Framework on HumanEval Dataset**

To run the CMCS framework on the HumanEval dataset, you should reference `check_completeness` from `human_eval/execution.py` in your script.

Example command:
```bash
python CMCS\ on\ HumanEval\ dataset/your_script.py
```

## CMCS on MBPP dataset

**Download the MBPP dataset locally:**
```bash
python -c "
import pandas as pd
from datasets import load_dataset
dataset = load_dataset('google-research-datasets/mbpp')
train_df = pd.DataFrame(dataset['train'])
train_df.to_parquet('mbpp_train.parquet')
"
```

**Run CMCS Framework on MBPP Dataset**

To run the CMCS framework on the MBPP dataset, you should reference `check_completeness` from `DatasetProcess/MBPP_Process/execution.py` in your script.

Example command:
```bash
python CMCS\ on\ MBPP\ dataset/your_script.py
```

## CMCS on SPoC dataset

**Download and Unzip the SPoC Dataset**

Download the SPoC dataset locally:
```bash
wget https://sumith1896.github.io/spoc/data/spoc.zip
unzip spoc.zip -d path/ && rm spoc.zip
```

**Extract Pseudo Code**

You can extract pseudo code from the SPoC dataset and save it to a `.txt` file using the `DatasetProcess/SPoCProcess/Read_pseudo_code.py` script.

Specify the dataset path and the output `.txt` file location as follows:
```bash
python DatasetProcess/SPoCProcess/Read_pseudo_code.py --input data/spoc --output output/pseudo_code.txt
```

**Run CMCS Framework on SPoC Dataset**

To run the CMCS framework on the SPoC dataset, you need to update the LLMs addresses and the test cases file path (`spoc/testcases`) in your script.

Example command:
```bash
python your_script.py
```

## Additional Information

...

```
