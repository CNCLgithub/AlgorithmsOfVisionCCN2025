using Pkg
using Conda
# Install pybullet to a local conda environment
Conda.add("pybullet", :AglorithmsOfVisionCCN2025)
pythonpath = joinpath(Conda.ROOTENV, "envs", "AglorithmsOfVisionCCN2025", "bin", "python")
# Re-build `PyCall` to use that environment
ENV["PYTHON"] = pythonpath
Pkg.add("PyCall")
Pkg.build("PyCall")
