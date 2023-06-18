# TUNEUP: A TRAINING STRATEGY FOR IMPROVING GENERALIZATION OF GRAPH NEURAL NETWORKS

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

[TUNEUP](https://openreview.net/pdf?id=8xuFD1yCoH), authored by [Weihua Hu](https://scholar.google.co.jp/citations?user=wAFMjfkAAAAJ&hl=ja), [Kaidi Cao](https://scholar.google.com/citations?user=4Zw1PJ8AAAAJ&hl=en),[Kexin Huang](https://scholar.google.com/citations?user=ogEXTOgAAAAJ&hl=en), [Edward W Huang](https://scholar.google.com/citations?user=EqvdkCAAAAAJ&hl=en), [ Karthik Subbian](https://scholar.google.com/citations?user=6ai0lDAAAAAJ&hl=en), and [Jure Leskovec](https://scholar.google.com/citations?user=Q_kKkIUAAAAJ&hl=en) is submitted to ICLR 2023. This paper proposes a novel method to employ a curriculum learning strategy on Graph Neural Networks (GNNs) for both inductive and transductive settings to alleviate a common problem: the neglect of nodes with lower degrees (tail nodes). That is, the classiccal simplistic loss for GNNs focuses on the easier task, optimizing the loss over the nodes with higher degree (head nodes), and overlooks the nodes with low degree (tail nodes), which are esentially harder to predict. This, overall, results in suboptimal performance. To mitigate this, they propose a curriculum learning strategy with esentially two stages: (1) train the model for the easy task (performing well on head nodes), and (2) adapt this model to the harder task (performing well on tail nodes).

## 1.1. Paper summary

TUNEUP uses a two-stage training strategy to train a GNN: initially employing the default trainig strategy, then transfer the learned model to train specifically for the tail nodes. For the first stage, the base GNN simply minimizes the given supervised loss, which is likely to perform well on head nodes while poorly on tail nodes. In the second stage, TUNEUP synthesizes many tail nodes by dropping edges. And, by reusing the supervision from the dropped edges, the base GNN is finetuned to perform well on the tail nodes. 

The paper addresses three main graph learning problems: semi-supervised node classification, link prediction, and recommender systems. In the context of this project, we focused on the link prediction task.

**Link Prediction Task:** Given a graph, predict new links. That is, given a source node, predict target nodes. 
* **Graph** $G$
* **Supervision** Y: whether node $s \in V$ is linked to  a node $t \in V$ in $G$
* **GNN** $F_{\theta}$: GNN model that predicts the score for a pair of nodes: $(s,t) \in V x V$, by generating the embeddings of $s$ and $t$ and calculating a score for their concatanetion using an MLP.
* **Prediction**: $Y'$
* **Loss** L: [The Bayesian Personalized Ranking (BPR) loss](https://arxiv.org/abs/1205.2618), a contrastive loss to increase the scores for positive node pairs compared to negative ones.

**Below is the pseudocode of the overall TUNEUP Method**
```
Given: GNN Fθ, graph G, loss L, supervision Y , DropEdge ratio α.

1:  # First stage: Default training to obtain a base GNN.
2:  while θ not converged do
3:    Make prediction Y' = Fθ(G)
4:    Compute loss L(Y , Y'), compute gradient ∇θL, and update parameter θ.
5:  end while

6:  # Second stage: Fine-tuning the base GNN with increased tail supervision.
7:  while θ not converged do
8:    Synthesize tail nodes, i.e., randomly drop α of edges: G DropEdge −−−−−−→ G'.
9:    Make prediction Y' = Fθ(G').
10:   Compute loss L(Y , Y'), compute gradient ∇θL, and update parameter θ.
11: end while

```

# 2. The method and my interpretation

## 2.1. The original method

The paper proposes a method called TuneUp. Training of a backbone network is divided into 2 parts. First part as default GNN training, second part is where edges are dropped from training edges to "simulate" tail nodes.

### 2.1.2 Default GNN Training
Given the graph G, and inherent supervision


## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
