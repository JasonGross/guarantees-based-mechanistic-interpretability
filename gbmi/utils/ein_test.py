import os, sys
import functorch
import torch

from gbmi.utils.testing import TestCase
from gbmi.utils import ein


class EinTest(TestCase):
    def test_ein_matmul(self):
        A = torch.rand(5, 3)
        B = torch.rand(3, 10)
        C = ein.array(lambda i, j: ein.sum(lambda k: A[i, k] * B[k, j]))
        self.assertTrue(torch.allclose(C, A @ B))

    def test_ein_path(self):
        # fmt: off
        table = torch.tensor(
            [[8, 1, 2],
             [7, 0, 3],
             [6, 5, 4]])
        # fmt: on

        coords = torch.tensor(
            [
                [1, 0, 0, 1, 2, 2, 2, 1, 0],  # y coords
                [1, 1, 2, 2, 2, 1, 0, 0, 0],  # x coords
            ]
        )

        path = ein.array(lambda t: table[coords[0, t], coords[1, t]])

        self.assertExpectedPretty(path, """tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])""")

    def test_ein_where(self):
        identity = ein.array(lambda i, j: torch.where(i == j, 1, 0), sizes=[3, 4])
        self.assertExpectedPretty(
            identity,
            """\
tensor([[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]])""",
        )

    def test_ein_repeat(self):
        rows = ein.array(lambda i, j: i, sizes=[3, 4])
        self.assertExpectedPretty(
            rows,
            """\
tensor([[0, 0, 0, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2]])""",
        )

    def test_ein_diagonal_indexing_works_on_square_matrices(self):
        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        diag = ein.array(lambda i: A[i, i])
        self.assertExpectedPretty(diag, """tensor([1, 5, 9])""")

    def test_ein_diagonal_indexing_works_on_non_square_matrices(self):
        A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.assertExpectedPretty(
            ein.array(lambda i: A[i, i], sizes=[3]), """tensor([1, 5, 9])"""
        )

    def test_ein_unbound_usage_after_usage_succeeds(self):
        A = torch.tensor([1, 2, 3])
        self.assertExpectedPretty(
            ein.array(lambda i: A[i] + torch.exp(i)),
            """tensor([ 2.0000,  4.7183, 10.3891])""",
        )

    def test_ein_unbound_usage_before_usage_succeeds_with_implicit_sizes(self):
        A = torch.tensor([1, 2, 3])
        self.assertExpectedPretty(
            ein.array(
                lambda i: torch.exp(i) + A[i],
            ),
            """tensor([ 2.0000,  4.7183, 10.3891])""",
        )

    def test_ein_unbound_usage_before_usage_succeeds_with_explicit_sizes(self):
        A = torch.tensor([1, 2, 3])
        self.assertExpectedPretty(
            ein.array(lambda i: torch.exp(i) + A[i], sizes=[3]),
            """tensor([ 2.0000,  4.7183, 10.3891])""",
        )

    def test_ein_index_an_indexed_array(self):
        A = torch.tensor([[0, 1], [2, 3]])
        self.assertTrue(torch.allclose(ein.array(lambda i, j: A[i][j]), A))

    # TODOs and bugs
    def test_ein_indexing_fails_if_out_of_range_positive(self):
        A = torch.tensor([0, 1, 2])
        self.assertExpectedRaisesInline(
            IndexError,
            lambda: ein.array(lambda i: 0 * A[i] + A[i + 1]),
            """index 3 is out of bounds for dimension 0 with size 3""",
        )

    def test_ein_indexing_works_if_out_of_range_negative(self):
        A = torch.tensor([0, 1, 2])
        self.assertExpectedPretty(
            ein.array(lambda i: 0 * A[i] + A[i - 1]), """tensor([2, 0, 1])"""
        )

    # TODO: add a better warning for the following
    def test_ein_unbound_index_with_no_size_errors(self):
        A = torch.tensor([[0, 1], [2, 3]])
        test_func = lambda: ein.array(lambda i, j: A[j, j])
        dimname = "dim"
        if os.name == "nt":
            try:
                test_func()
            except ValueError as e:
                if "d0" in str(e):
                    dimname = "d0"
        self.assertExpectedRaisesInline(
            ValueError,
            test_func,
            f"""dimension {dimname} is unbound""",
        )
