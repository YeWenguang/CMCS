要将 Hugging Face 数据集 `google-research-datasets/mbpp` 下载并保存为 Parquet 格式，可以使用以下命令行步骤：

1. 安装 `datasets` 和 `pandas` 库：
   ```bash
   pip install datasets pandas pyarrow
   ```

2. 使用以下命令下载并保存数据集为 Parquet 格式：

   ```bash
   python -c "
import pandas as pd
from datasets import load_dataset

# 下载数据集
dataset = load_dataset('google-research-datasets/mbpp')

# 将训练集转换为 pandas DataFrame
train_df = pd.DataFrame(dataset['train'])

# 保存为 Parquet 文件
train_df.to_parquet('mbpp_train.parquet')

print('数据集已保存为 Parquet 文件')
"
   ```

通过上述步骤，你可以将 `google-research-datasets/mbpp` 数据集下载并保存为 Parquet 文件格式。

# CMCS
Enhancing LLM-based Code Generation via Cross-Model Collaboration

**The files of the CMCS framework have not been properly organized yet,and we are continuously updating it......**

1. Process SPoC files in "SPoCProcess".
2. The CMCS framework running on SPoC is in the file "CMCS on SPoC dataset".
3. The CMCS framework running on HumanEval is in the file "HumanEval on SPoC dataset".
4. The CMCS framework running on MBPP is in the file "MBPP on SPoC dataset".


##CMCS on HumanEval dataset：

**Download the HumanEval dataset locally:**
```bash
$ git clone  https://github.com/openai/human-eval
$ pip install -e human-eval
```

**Run CMCS Framework on HumanEval Dataset**

To run the CMCS framework on the HumanEval dataset, you should reference check_completeness from human_ eval/execution. py in the py file.

Example command:

```bash
python CMCS on HumanEval dataset/your_script.py
```

##CMCS on MBPP dataset：

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

To run the CMCS framework on the SPoC dataset, you should reference check_completeness from human_ eval/execution. py in the py file.

Example command:

```bash
python CMCS on HumanEval dataset/your_script.py
```

##CMCS on SPoC dataset：

**Download and Unzip the SPoC Dataset**

For the SPoC dataset, you should https://github.com/Sumith1896/spoc Download the SPoC dataset locally:
```bash
wget  https://sumith1896.github.io/spoc/data/spoc.zip
unzip spoc.zip && mv path/ && rm spoc.zip
```


**Extract Pseudo Code**

You can extract pseudo code from the SPoC dataset and save it to a `.txt` file using the `DatasetProcess/SPoCProcess/Read_pseudo_code.py` script.

Specify the dataset path and the output `.txt` file location as follows:

```bash
python DatasetProcess/SPoCProcess/Read_pseudo_code.py --input data/spoc --output output/pseudo_code.txt
```

**Run CMCS Framework on SPoC Dataset**

To run the CMCS framework on the SPoC dataset, you need to update the LLMs addresses and the test cases file path (`spoc/testcases`) in the `.py` file.

Example command:

```bash
python your_script.py
```


## Additional Information

...
```
