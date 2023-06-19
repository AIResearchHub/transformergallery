

from .utils import apply_mlm_mask

def test_memory(model, dataloader):
    """
    Parameters:
        model (TransformerLM): A transformer variant of bert
        data (Dataloader): A pytorch dataloader object
    """

    for i, batch in enumerate(dataloader):


