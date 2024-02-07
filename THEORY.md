# Mathematical 
## Hypernetworks

### Why does it work?

- The concept of soft weight sharing 
- Our model is data conditioned

### Relevant models
- Hypergradient
- Model agnostic Meta Learning: Gradient Based
- Bi-level optimization
- Self-tuning networks
# Programing design

## Normalizer class

- Inheriting from torch.nn.Module and use forward method, why?
- When to know to pass the Dataset to the Normalizer class only, or add a set_stats 
function that takes a dataset and calculates the mean and std of the dataset ?