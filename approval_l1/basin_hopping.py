from typing import Optional
import numpy as np
from scipy.optimize import BasinHopping

VotingHist = np.ndarray


def anneal(votings_hists: list[VotingHist], N: int, seed: Optional[int] = None) -> VotingHist:
  
  x0 = np.sort(np.random.randint(0, N, N))
