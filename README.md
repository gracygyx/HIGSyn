## HIG-Syn: A Multi-Granularity Network for Predicting Synergistic Drug Combinations

### Model
- Framework
  ![](https://github.com/gracygyx/HIGSyn/blob/main/Figures/Framework.jpg)

- Interaction-aware attention mechanism
  ![](https://github.com/gracygyx/HIGSyn/blob/main/Figures/Attention.jpg)

### Environment

```
pip install -r requirements.txt
```


### Train and Test

Our program is easy to train and test,  just need to run "main_DrugComb.py" for DrugComb dataset and  "main_GDSC2.py" for GDSC2 dataset. 

```
python main_DrugComb.py
```

```
python main_GDSC2.py
```

### Performance on DrugComb and GDSC2 datasets

- DrugComb dataset
  ![](https://github.com/gracygyx/HIGSyn/blob/main/Figures/DrugComb.jpg)

- GDSC2 dataset
  ![](https://github.com/gracygyx/HIGSyn/blob/main/Figures/GDSC2_result.jpg)
