# Overview
Meet SuperPlug AutoML – your easygoing sidekick for Python machine learning. Forget efficiency; we're all about that NumPy life and taking the scenic route in supervised learning. Get ready for a laugh as SuperPlug AutoML turns model creation into a comedy show – because who said machine learning couldn't be entertainingly inefficient?

P.S. Developed by me, ChatGPT, Claude, and Bard (in order of contributions decreasingly). Backend is fully inspired by Scikit-Learn, effectively a distilled version of it. 

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