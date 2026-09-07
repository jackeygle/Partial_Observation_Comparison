"""crowdcore — the core shared by all five methods.

What lives here is **method-agnostic**: where the data comes from, how the robots
move, how observations are generated, which cells are walkable, which channel is
defined on which cells. No method's model, loss, or training loop belongs here.

  config              reads config.yaml (all grid / observation / navigation / prior params)
  navigation          walkable region, A* path planning, the real ATC map
  observation_model   multi-robot sensor motion and observation generation,
                       Omega/Omega_c, X0 fill-in
  data                the CSV -> h5 -> grid data pipeline

Before this these four things lived inside 4dvarnet_enkf/, and the other two
methods imported them via a hardcoded absolute path,
`/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf` (72 places across the repo).
Moved here, the sharing relationship is explicit.
"""
