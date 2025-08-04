### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 71eb1272-37f5-48db-b516-8e5bdbac8d7e
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(mktempdir())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	Pkg.add("Conda")
    using Conda
	Conda.add("pybullet", :AglorithmsOfVisionCCN2025)
	pythonpath = joinpath(Conda.ROOTENV, "envs", "AglorithmsOfVisionCCN2025", "bin", "python")
	ENV["PYTHON"] = pythonpath
	Pkg.add("PyCall")
	Pkg.build("PyCall")
	Pkg.add("Gen")
	Pkg.add(url="https://github.com/CNCLgithub/PhySMC.git")
	Pkg.add(url="https://github.com/CNCLgithub/PhyBullet.git")
	using PyCall, PhySMC, PhyBullet, Gen
end

# ╔═╡ 001f5f82-b18f-4a1a-9be6-ac44d66fde21
pybullet = PyCall.pyimport("pybullet")

# ╔═╡ fc18c3c3-6d4b-4d7f-a52f-665e472c9116

function simple_scene(mass::Float64=1.0,
                      restitution::Float64=0.9)
    client = @pycall pb.connect(pb.DIRECT)::Int64
    pb.setGravity(0,0,-10; physicsClientId = client)

    # add a table
    dims = [1.0, 1.0, 0.1] # in meters
    col_id = pb.createCollisionShape(pb.GEOM_BOX,
                                     halfExtents = dims,
                                     physicsClientId = client)
    obj_id = pb.createMultiBody(baseCollisionShapeIndex = col_id,
                                basePosition = [0., 0., -0.1],
                                physicsClientId = client)
    pb.changeDynamics(obj_id,
                      -1;
                      mass = 0., # 0 mass are stationary
                      restitution = 0.9, # some is necessary
                      physicsClientId=client)


    # add a ball
    bcol_id = pb.createCollisionShape(pb.GEOM_SPHERE,
                                      radius = 0.1,
                                      physicsClientId = client)
    bobj_id = pb.createMultiBody(baseCollisionShapeIndex = bcol_id,
                                 basePosition = [0., 0., 1.0],
                                 physicsClientId = client)
    pb.changeDynamics(bobj_id,
                      -1;
                      mass = mass,
                      restitution = restitution,
                      physicsClientId=client)

    (client, bobj_id)
end

# ╔═╡ 1d5eec5b-6eef-489b-a08c-a54346e60d5d
client, ball = simple_scene()

# ╔═╡ 4624cb2b-5767-4899-8991-560b74d10177


# ╔═╡ Cell order:
# ╠═71eb1272-37f5-48db-b516-8e5bdbac8d7e
# ╠═001f5f82-b18f-4a1a-9be6-ac44d66fde21
# ╠═fc18c3c3-6d4b-4d7f-a52f-665e472c9116
# ╠═1d5eec5b-6eef-489b-a08c-a54346e60d5d
# ╠═4624cb2b-5767-4899-8991-560b74d10177
