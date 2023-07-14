from multiprocessing_utils import *
# Do not import DGL and SAR - these modules should be
# independently loaded inside each process


@sar_test
def test_patch_dgl():
    """
    Import DGL library and SAR and check whether `patch_dgl` function
    overrides edge_softmax function in specific GNN layers implementation.
    """
    import dgl
    original_gat_edge_softmax = dgl.nn.pytorch.conv.gatconv.edge_softmax
    original_dotgat_edge_softmax = dgl.nn.pytorch.conv.dotgatconv.edge_softmax
    original_agnn_edge_softmax = dgl.nn.pytorch.conv.agnnconv.edge_softmax

    import sar
    sar.patch_dgl()

    assert original_gat_edge_softmax == dgl.nn.functional.edge_softmax
    assert original_dotgat_edge_softmax == dgl.nn.functional.edge_softmax
    assert original_agnn_edge_softmax == dgl.nn.functional.edge_softmax

    assert dgl.nn.pytorch.conv.gatconv.edge_softmax == sar.patched_edge_softmax
    assert dgl.nn.pytorch.conv.dotgatconv.edge_softmax == sar.patched_edge_softmax
    assert dgl.nn.pytorch.conv.RelGraphConv == sar.RelGraphConv
    assert dgl.nn.RelGraphConv == sar.RelGraphConv