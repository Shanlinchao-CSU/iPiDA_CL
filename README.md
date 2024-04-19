# iPiDA-CL: A Contrastive Learning Approach for Predicting Potential PiRNA-Disease Associations
In this study, we present iPiDA-CL, identification of piRNA-disease associations based on contrastive learning.  iPiDA-CL calculates Gaussian kernel similarities between piRNA and disease pairs to generate initial embeddings. It then utilizes a parameter-sharing network, incorporating online and target strategies with data augmentation, to establish a contrastive learning framework. This framework generates embeddings for piRNAs and diseases based on association pairs, and employs a cross-prediction method to compute specific association scores.
# Requirements
- torch 1.10.1
- python 3.7.13
- numpy 1.21.6
- scikit-learn 1.0.2
# Data
In our research, we conducted an extensive evaluation and testing of our model using MNDR v3.0 as database. The dataset was carefully sorted to insure data normalization and validity, which included eliminating duplicate association. Our investigation specifically targeted associations between piRNAs and diseases of human, leading to a comprehensive dataset that encompasses 11,981 instances with empirical validation. These instances encapsulated connections involving 10,149 distinct piRNAs across 19 varied diseases, serving as the basis of our analysis.

For a more robust validation of iPiDA-CL's performance, an independent test set was meticulously curated to conduct a comprehensive evaluation of the model's capabilities. A total of 2,489 piRNA-disease association pairs were meticulously collected from relevant literature sources. This independent test set encompassed interactions involving 2,415 distinct piRNAs and 13 distinct diseases.

Since the Gaussian kernel similarity matrix of piRNA and disease is too large, and the generation process is very time-consuming, we separate this process and require users to use util/GIS.py to generate the Gaussian kernel similarity matrix before running the project.The matrix was generated from piRNA-Disease Interaction data.
# Run the demo
```
python main.py
```
