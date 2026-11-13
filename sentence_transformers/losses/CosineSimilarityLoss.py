import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer


class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity(), regularization=True, reg_strength=0.001, entropy='softmax', target=None):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.regularization = regularization
        self.reg_strength = reg_strength
        # 'softmax': as in the EAR reference implementation, entropy of a softmax over the
        #            (already normalized) attention weights
        # 'attention': entropy of the attention weights themselves
        assert entropy in ('softmax', 'attention')
        self.entropy = entropy
        # target: attention entropy per layer (nats); instead of maximizing the entropy, penalize the
        # squared distance to the target, which can lie above or below the current entropy
        self.target = None if target is None else torch.tensor(target, dtype=torch.float)
        if self.target is not None:
            self.entropy = 'attention'


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = []
        entropies = []
        for sentence_feature in sentence_features:
            model_output = self.model(sentence_feature)
            embeddings.append(model_output['sentence_embedding'])
            if self.regularization == True:
                entropies.append(self._compute_negative_entropy(torch.stack(model_output["attentions"]), model_output["attention_mask"]))
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        loss = self.loss_fct(output, labels.view(-1))
        if self.regularization == True:
            # minimizing the negative entropy maximizes the attention entropy
            # torch.stack keeps the entropy terms in the graph, so they contribute gradients
            neg_entropy = torch.stack(entropies).mean(0)  # L
            if self.target is None:
                loss = loss + self.reg_strength * neg_entropy.sum()
            else:
                loss = loss + self.reg_strength * ((-neg_entropy - self.target.to(neg_entropy.device)) ** 2).sum()
        return loss

    def _compute_negative_entropy(self, inputs: torch.torch, attention_mask: torch.torch):
        """Compute the negative entropy across layers of a network for given inputs.

        Args:
            - input: tensor. should be in the form: LBHSS
            - attention_mask. Tensor with dim: BS
        """

        #  average over attention heads
        pool_heads = inputs.mean(2)  # LBSS
        mask = attention_mask.bool()  # BS

        keys = mask[None, :, None, :]
        if self.entropy == 'softmax':
            #  exclude padded tokens as keys: masked scores get exactly zero probability
            #  (finite fill value instead of -inf, which would give nan gradients)
            scores = pool_heads.masked_fill(~keys, torch.finfo(pool_heads.dtype).min)
            p, log_p = scores.softmax(-1), scores.log_softmax(-1)
        else:
            #  attention weights over non-padded keys, renormalized
            p = pool_heads.masked_fill(~keys, 0)
            p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            log_p = p.clamp_min(1e-12).log()

        #  get the negative entropy for each token
        neg_entropy = torch.where(keys, p * log_p, torch.zeros_like(p)).sum(-1)  # LBS

        #  get the "average entropy" that traverses the layer, over non-padded tokens only
        queries = mask[None, :, :].to(neg_entropy.dtype)
        mean_entropy = (neg_entropy * queries).sum(-1) / queries.sum(-1)  # LB

        #  per layer, average over the batch (summed across the layers in forward)
        return mean_entropy.mean(1)  # L
