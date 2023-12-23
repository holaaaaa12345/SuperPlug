# Overview
Introducing SuperPlug AutoML, your straightforward solution for hassle-free supervised machine learning in Python. This plug-and-play gem requires only NumPy, eliminating unnecessary dependencies and complexities. Welcome to the simplicity of SuperPlug AutoML—where building accurate models has never been easier.

![alt text](main_page.PNG)

### The pipeline for each column features
```mermaid
  graph TD;
      A(Columns)  --> B{Is categorical?}
      B --yes--> C[Most frequent impute]
      B --no--> D[Mean impute]
      C --> E[One hot encoding]
      D --> F[Z-score standardization]
```

 ### The pipeline for each models
 ```mermaid
 graph TD;
	 A(Models) --> Z[Train test split]--> B{Have hyperparameter?}
	 B -- no --> C[Fit] --> D[Evaluate]
	 B -- yes--> E[Randomized search cv]
	 E --> F[Fit] --> G[Evaluate]