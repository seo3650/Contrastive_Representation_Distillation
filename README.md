# Contrastive_Representation_Distillation
Implementation of CRD (Contrastive Representation Distillation)

## Quick Start
### Teacher training
```
python main.py --option "teacher"
```
### Student training without distillation
```
python main.py --option "student"
```
### Student training with distillation
```
python main.py --option "distill" --teacher_model [teacher_model]
```
### Test
```
python main.py --option [option] --test --prev_model [prev_model]
```

## Result
<table>
<tr>
  <th>Model</th>
  <th>Teacher</th>
  <th>Student (No Distill)</th>
  <th colspan="5">Studnet (Distill)</th>
</tr>
<tr>
  <td>Beta</td>
  <td>-</td>
  <td>-</td>
  <td>0.7</td>
  <td>0.8</td>
  <td>0.9</td>
  <td>1.0</td>
  <td>1.1</td>
</tr>
<tr>
  <td>Accuracy</td>
  <td>77.08%</td>
  <td>76%</td>
  <td>75.92%</td>
  <td>75.57%</td>
  <td>76.08%</td>
  <td>76.27%</td>
  <td>75.74%</td>
</tr>
</table>


## Reference
1. Contrastive Representation Distillation (https://arxiv.org/abs/1910.10699)
2. CIFAR-100 Dataset (https://www.cs.toronto.edu/~kriz/cifar.html)
3. ResNet implementation (https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py)
