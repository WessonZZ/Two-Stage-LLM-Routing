
# Two-Stage LLM Routing with Embedding Representation

This repo is the implementation of our two-stage llm routing method.


**Details:**

Before routing, we cluster all the LLM candidates into several LLM groups based on their performance and cost.

## In training:
We use two contrastive losses to learn the embedding representation of the groups and LLMs to make the similarity of query and positive samples more closer. In order to tackle the new models' coming and dropping, we also optimize a projection matrix to link the embeddings of LLMs to their performance vector. Then when new models are available, we just test them on a validation dataset to obtain their performance vectors, and multiply the vectors with the inverse of the learned projection matrix to get their embeddings, which makes it convenient to merge new models into the routing system without retraining. More details refer to [LLM_router.pdf](LLM_router.pdf) and [poster.pdf](poster.pdf).


## In inference:

- First Stage: We select the best groups via the similarity between the query and LLM groups;
- Second Stage: We choose the optimal LLM from the selected groups in the first stage;

