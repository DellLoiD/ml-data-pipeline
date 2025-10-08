from typing import Dict, List


# In-memory storage for saved hyperparameter grids
SAVED_RANDOM_GRIDS: List[Dict] = []


def save_hyperparameters(random_grid: Dict) -> None:
    """Append provided hyperparameter grid to in-memory storage.

    This function is intended to be called from the UI layer when the user
    presses the "Save hyperparameters" button.
    """
    SAVED_RANDOM_GRIDS.append(random_grid)


def get_saved_hyperparameters() -> List[Dict]:
    """Return a shallow copy of saved hyperparameter grids for read-only access."""
    return list(SAVED_RANDOM_GRIDS)


