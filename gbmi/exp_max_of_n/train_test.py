import torch

from gbmi.exp_max_of_n.train import MAX_OF_10_CONFIG
from gbmi.model import train_or_load_model
from gbmi.utils import ein
from gbmi.utils.testing import TestCase


class TestMaxOfN(TestCase):
    def test_gen_output(self):
        cfg = MAX_OF_10_CONFIG
        data = torch.tensor([[1, 2, 3, 4, 3, 2], [2, 3, 4, 5, 4, 3]])
        self.assertExpectedPretty(
            cfg.experiment.get_ground_truth(data), """tensor([4, 5])"""
        )

    def test_max_of_k(self):
        cfg = MAX_OF_10_CONFIG
        cfg.experiment.nth_max = 4
        data = torch.tensor([[1, 2, 3, 4, 3, 2], [2, 3, 4, 5, 4, 3]])
        self.assertExpectedPretty(
            cfg.experiment.get_ground_truth(data), """tensor([2, 3])"""
        )


class TestOneLayerTransformer(TestCase):
    def test_forward_pass(self):
        model = MAX_OF_10_CONFIG.build_model()
        model.to("cpu")
        W_U, W_E, W_pos, W_Q, W_K, W_V, W_O = [
            i.squeeze().clone().detach()
            for i in [
                model.W_U,
                model.W_E,
                model.W_pos,
                model.W_Q,
                model.W_K,
                model.W_V,
                model.W_O,
            ]
        ]
        d_head = model.cfg.d_head

        seq = torch.tensor([31, 30, 32, 33, 34, 35, 39, 37, 38, 36])
        n = len(seq)

        # Embeddings (x[i]: embedding of input token i)
        x = ein.array(lambda i: W_E[seq[i]], sizes=[n])
        self.assertExpectedPretty(x.shape, """Size((10, 32))""")

        # QK path (a[i]: attention from last element to input token i)
        a = ein.array(lambda i: (x[9] + W_pos[9]) @ W_Q @ W_K.T @ (x[i] + W_pos[i]).T)
        a_softmax = torch.softmax(a / (d_head**0.5), dim=0)
        self.assertExpectedPretty(a_softmax.shape, """Size((10,))""")

        # OV path (v[i]: OV logit contribution of input token i)
        v = ein.array(lambda i: (x[i] + W_pos[i]) @ W_V @ W_O @ W_U)
        self.assertExpectedPretty(v.shape, """Size((10, 64))""")

        # embed/unembed path (r[i]: residual logit contribution of input token i)
        r = ein.array(lambda i: (x[i] + W_pos[i]) @ W_U)
        self.assertExpectedPretty(r.shape, """Size((10, 64))""")

        # last element logits (y[i]: last element logit for input token i)
        y = ein.sum(lambda i: a_softmax[i] * v[i]) + r[9]
        self.assertExpectedPretty(y.shape, """Size((64,))""")
        self.assertExpectedPretty(y.argmax(-1), """tensor(59)""")

        # === True logits ===
        true_logits, cache = model.run_with_cache(torch.tensor(seq))
        self.assertTrue(torch.allclose(y, true_logits.squeeze()[-1], atol=1e-5))
        self.assertExpectedPretty(
            true_logits.argmax(-1).squeeze()[-1],
            """tensor(59)""",
        )
