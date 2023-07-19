# A lightning model for Author Rewriting  
## UPDATA 2023/7/19
version_0.1: Applying Transformer in GAN. 
- Loss Function  
  - Discriminator Loss: $L_{d} = CELoss(\hat{y}, y)$
  - Restructure Loss: $L_{r}= CELoss(\widehat{y}, \widetilde{y})$

- \color{red}{What's wrong} 
    - We got the same output tensor for each tensors.
- \color{lime}{Cause analysis}
  - outputs of last decoder layer can be updated. We guess that Fully-Connected layer after last decoder layer caused
  the problem (the sum operation may cause this problem).
- \color{green}{Possible solutions}
  - remove the Fully-Connected layer 
  - use mean operation alternative the sum