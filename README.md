# CMCS
Enhancing LLM-based Code Generation via Cross-Model Collaboration

**The files of the CMCS framework have not been properly organized yet,and we are continuously updating it......**

1. Process SPoC files in "SPoCProcess".
2. The CMCS framework running on SPoC is in the file "CMCS on SPoC dataset".
3. The CMCS framework running on HumanEval is in the file "HumanEval on SPoC dataset".
4. The CMCS framework running on MBPP is in the file "MBPP on SPoC dataset".

**CMCS on SPoC dataset：**

**Download and Unzip the SPoC Dataset**

For the SPoC dataset, you should https://github.com/Sumith1896/spoc Download the SPoC dataset locally:
```bash
wget  https://sumith1896.github.io/spoc/data/spoc.zip
unzip spoc.zip && mv path/ && rm spoc.zip
```

**The above commands will:**
1. **Download the `spoc.zip` file.**
2. **Unzip the file.**
3. **Rename and move the unzipped `spoc` folder to `data`.**
4. **Delete the downloaded `spoc.zip` file.**

## Extract Pseudo Code

You can extract pseudo code from the SPoC dataset and save it to a `.txt` file using the `DatasetProcess/SPoCProcess/Read_pseudo_code.py` script.

Specify the dataset path and the output `.txt` file location as follows:

```bash
python DatasetProcess/SPoCProcess/Read_pseudo_code.py --input data/spoc --output output/pseudo_code.txt
```

**The above command will:**
1. **Extract pseudo code from the `data/spoc` directory.**
2. **Save the extracted pseudo code to the `output/pseudo_code.txt` file.**

## Run CMCS Framework on SPoC Dataset

To run the CMCS framework on the SPoC dataset, you need to update the LLMs addresses and the test cases file path (`spoc/testcases`) in the `.py` file.

Example command:

```bash
python your_script.py --llm-address your_llm_address --test-cases data/spoc/testcases
```

**The above command will:**
1. **Update the LLMs addresses in the `your_script.py` file.**
2. **Specify the path to the test cases file in the SPoC dataset.**

## Additional Information

...
```
