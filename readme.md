# A lightning model for Author Rewriting  
## UPDATE 2023/8/8/  
version_0.3: GPT Style PTM available.  
In this version, we adopt GPT Style PTM task for text generation which could be represented as follows:  
$max L = p(x_t|x_{i<t}$

## UPDATE 2023/7/20/ 10:00  
version_0.2.1: Applying Transformer in GAN.  
<font face="Lumanosimo">* **Changes**</font>  
In this version, we applied LR_Scheduler in optimizers.  
Then, we increased num_worker from 1 to 16, batch_size was also increased to 512.   
## UPDATE 2023/7/19 22:22
version_0.2: Applying Transformer in GAN.  
<font face="Lumanosimo">* **Changes**</font>  
- Put GRU layer after the last Encoder layer.
- The Cosine Positional Embedding Table in Transformer\[google2017\] was used.
- The restructure loss in CAAT \[Peking University- X L Wan -2019\] was used.
- Parameters init method xavier uniform was used.
- method covert_latent_to_vocab_probability_distribution() changed F.softmax() to F.log_softmax()
- SGD optimizer was applied in generator with $lr=0.1$, and AdamW optimizer was applied in discriminator with $lr=0.1$.  
<font face="Lumanosimo" size=5 >_It worked!_</font> 
- \color{red}{Some thing was still wrong.}  
  (1) loss of discriminator didn't decrease. 

## UPDATE 2023/7/19 15:03
version_0.1: Applying Transformer in GAN. 
- Loss Function  
  - Discriminator Loss: $L_{d} = CELoss(\hat{y}, y)$
  - Restructure Loss: $L_{r}= CELoss(\widehat{y}, \widetilde{y})$

- \color{red}{What's wrong} 
    - We got the same output tensor for each tensor. All tensor will be predicted as '[PAD]'.
- \color{lime}{Cause analysis}
  - outputs of last decoder layer can be updated. We guess that Fully-Connected layer after last decoder layer caused
  the problem (the sum operation may cause this problem).
- \color{green}{Possible solutions}
  - remove the Fully-Connected layer &#x2716;
    - Reason: We haven't used Fully-Connected layer after last Decoder layer. 
  - use mean operation alternative the sum &#x2716
    - Here, situations need to be discussed.  
      (1) For Sequence classification Task, the mean operation may not influence.
      (2) As for Word Prediction Task, the mean operation do influence the result.