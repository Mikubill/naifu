import torch
import torch.nn as nn

class vector_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook):
        with torch.no_grad():
            codebook_sqr = torch.sum(codebook**2, dim=1)
            x_sqr = torch.sum(x**2, dim=1, keepdim=True)

            dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices = dist.min(dim=1)

            ctx.save_for_backward(indices, codebook)
            ctx.mark_non_differentiable(indices)

            nn = torch.index_select(codebook, 0, indices)
            return nn, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors

            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)


class VectorQuantize(nn.Module):
    def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
        """
        Takes an input of variable size (as long as the last dimension matches the embedding size).
        Returns one tensor containing the nearest neighbour embeddings to each of the inputs,
        with the same size as the input, vq and commitment components for the loss as a tuple
        in the second output and the indices of the quantized vectors in the third:
        quantized, (vq_loss, commit_loss), indices
        """
        super(VectorQuantize, self).__init__()

        self.codebook = nn.Embedding(k, embedding_size)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.vq = vector_quantize.apply

        self.ema_decay = ema_decay
        self.ema_loss = ema_loss
        if ema_loss:
            self.register_buffer("ema_element_count", torch.ones(k))
            self.register_buffer("ema_weight_sum", torch.zeros_like(self.codebook.weight))

    def _laplace_smoothing(self, x, epsilon):
        n = torch.sum(x)
        return (x + epsilon) / (n + x.size(0) * epsilon) * n

    def _updateEMA(self, z_e_x, indices):
        mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
        elem_count = mask.sum(dim=0)
        weight_sum = torch.mm(mask.t(), z_e_x)

        self.ema_element_count = (self.ema_decay * self.ema_element_count) + ((1 - self.ema_decay) * elem_count)
        self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)
        self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + ((1 - self.ema_decay) * weight_sum)

        self.codebook.weight.data = self.ema_weight_sum / self.ema_element_count.unsqueeze(-1)

    def idx2vq(self, idx, dim=-1):
        q_idx = self.codebook(idx)
        if dim != -1:
            q_idx = q_idx.movedim(-1, dim)
        return q_idx

    def forward(self, x, get_losses=True, dim=-1):
        if dim != -1:
            x = x.movedim(dim, -1)
        z_e_x = x.contiguous().view(-1, x.size(-1)) if len(x.shape) > 2 else x
        z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
        vq_loss, commit_loss = None, None
        if self.ema_loss and self.training:
            self._updateEMA(z_e_x.detach(), indices.detach())
        # pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
        z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
        if get_losses:
            vq_loss = (z_q_x_grd - z_e_x.detach()).pow(2).mean()
            commit_loss = (z_e_x - z_q_x_grd.detach()).pow(2).mean()

        z_q_x = z_q_x.view(x.shape)
        if dim != -1:
            z_q_x = z_q_x.movedim(-1, dim)
        return z_q_x, (vq_loss, commit_loss), indices.view(x.shape[:-1])

