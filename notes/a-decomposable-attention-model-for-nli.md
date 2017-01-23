# [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933)

**Keywords:** NLI, SNLI, decomposable, attention model

## Summary

- In Natural Language Inference (NLI), a system gets a **premise** and a **hypothesis** and needs to determine if the relation between them is an **entailment**, a **contradiction**, or **neutral**

- Previous work used LSTM-based models to solve this task, while this paper aims for a simpler approach with fewer parameters

- The insight is that it can be enough to align (contradicting) words pair-wise and then aggregating that information 

  * "thunder" and "lightning" aligned with "sunny" indicate a contradiction

- Given two sentences **a** and **b** where words are represented by **external word embeddings**:

  1. **Soft-align** elements of **a** and **b** using a FFNN attention model ([Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473))
  2. **Compare** each aligned sub-phrase separately, using a FFNN
  3. **Aggregate** the results from previous step, and predict the class using a FFNN

- **Optional: Intra-Sentence Attention**. Augment the word embeddings, by encoding compositional relationships between words within each sentence ([Cheng et al., 2016](https://arxiv.org/abs/1601.06733)): `a_i_new = [ a_i ; a_intra ]` where `a_intra` is a weighted sum of all other words in the sentence, aligned to a_i

- The complexity of this model is `O(ld^2)` assuming `l < d` (l sentence lenght, d NN dimensionality), which is the same as a regular LSTM

## Notes

- The approach outperforms much more complex models on SNLI
- Pairwise comparisons can do the job for this task
  * Maybe this is true for other tasks as well
- The decomposition results in an impressive speed-up, without sacrificing performance
