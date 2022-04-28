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
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity(), regularization=True, reg_strength=0.001):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation
        self.regularization = regularization
        self.reg_strength = reg_strength


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = []
        entropies = []
        for sentence_feature in sentence_features:
            model_output = self.model(sentence_feature)
            embeddings.append(model_output['sentence_embedding'])
            if self.regularization == True:
                entropy = self._compute_negative_entropy(torch.stack(model_output["attentions"]), model_output["attention_mask"])
                entropies.append(-self.reg_strength * entropy)  # scale and change the arithmetic sign
        mean_entropy = torch.nan_to_num(torch.tensor(entropies).mean())
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]) + mean_entropy)
        return self.loss_fct(output, labels.view(-1))

    def _compute_negative_entropy(self, inputs: torch.torch, attention_mask: torch.torch):
        """Compute the negative entropy across layers of a network for given inputs.

        Args:
            - input: tensor. should be in the form: LBHSS
            - attention_mask. Tensor with dim: BS
        """

        #  average over attention heads
        pool_heads = inputs.mean(2)

        batch_size = pool_heads.shape[1]
        samples_entropy = list()
        for b in range(batch_size):
            #  get inputs from non-padded tokens of the current sample
            mask = attention_mask[b]
            sample = pool_heads[:, b, mask.bool(), :]
            sample = sample[:, :, mask.bool()]

            #  get the negative entropy for each non-padded token
            neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)

            #  get the "average entropy" that traverses the layer
            mean_entropy = neg_entropy.mean(-1)

            #  store the sum across all the layers
            samples_entropy.append(mean_entropy.sum(0))

        # average over the batch
        final_entropy = torch.stack(samples_entropy).mean()
        return final_entropy