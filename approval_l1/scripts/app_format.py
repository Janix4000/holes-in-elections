from typing import Optional
import os


def load(source: str) -> tuple[list[set[int]], int]:
    """# Summary
    Loads approval election from a file in the `app` format.

    ## Args:
        `source` (str): Path to the file with voting

    ## Returns:
        ->list[set[int]], int: Election as a list of sets (as votes) of chosen candidates ids in range 0 to N-1 and number of candidates N.
    """

    def load_raw_line(lines: list[str], line_idx: int) -> Optional[str]:
        n = len(lines)

        line = None
        while line_idx < n and line is None:
            line = lines[line_idx]
            line_idx += 1

            if line.isspace() or line.startswith('# '):
                line = None

        return line, line_idx

    with open(source, 'r') as f:
        lines = f.readlines()

        election: list[set[str]] = []
        line_idx = 0

        line, line_idx = load_raw_line(lines, line_idx)
        N = int(line.strip())
        for _ in range(N):
            line, line_idx = load_raw_line(lines, line_idx)

        line, line_idx = load_raw_line(lines, line_idx)
        _a, _b, r = tuple(map(int, line.strip().split(',')))
        for _ in range(r):
            line, line_idx = load_raw_line(lines, line_idx)
            i = line.find(',')
            t = int(line[:i])
            nums_str = line[i + 1:].strip(' {}\n').split(',')
            if nums_str == ['']:
                s = set()
            else:
                s = set(map(int, nums_str))
            election.extend([s] * t)

        return election, N


def to_histograms(election: list[set[int]], n: int) -> list[int]:
    hist = [0] * n
    for vote in election:
        for i in vote:
            hist[i] += 1

    return hist


def load_mapel(source: str) -> list[list[int]]:
    if source.endswith(".app"):
        files_paths = [source]
    else:
        files_paths = list(os.listdir(source))
    elections_histograms = []
    for path in files_paths:
        if not path.endswith(".app"):
            continue
        election, _N = app_format.load(os.path.join(source, path))
        elections_histograms.append(app_format.to_histograms(election, N))

    return elections_histograms
