### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 71eb1272-37f5-48db-b516-8e5bdbac8d7e
begin
    import Pkg
    # activate the project environment
    Pkg.activate(mktempdir())
    Pkg.instantiate()

	# Install pybullet via Conda.jl
	Pkg.add("Conda")
    using Conda
	Conda.add("pybullet", :AglorithmsOfVisionCCN2025)

	# Build PyCall.jl with Conda env. (so it can import pybullet)
	pythonpath = joinpath(Conda.ROOTENV, "envs", "AglorithmsOfVisionCCN2025", "bin", "python")
	ENV["PYTHON"] = pythonpath
	Pkg.add("PyCall")
	Pkg.build("PyCall")
	
	# Add other dependencies
	Pkg.add("Gen")
	Pkg.add("Plots")
	Pkg.add("Accessors");
	Pkg.add("Distributions");
	Pkg.add(url="https://github.com/CNCLgithub/PhySMC.git")
	Pkg.add(url="https://github.com/CNCLgithub/PhyBullet.git")
	using PyCall, PhySMC, PhyBullet, Gen, Accessors, Distributions, Plots
end

# ╔═╡ d1678c83-d8cd-4b2c-8257-ff72300dcf00
md"""
> Preamble
"""

# ╔═╡ 4624cb2b-5767-4899-8991-560b74d10177
md"""
# Inference by reversing physical simulation
"""

# ╔═╡ 7ebb39fe-451c-4f10-811f-3cdde5f71a55
md"""
Introduced, in large part, by [Battaglia et al. (2013)](https://www.pnas.org/doi/pdf/10.1073/pnas.1306572110), the idea of a "physics engine in the mind" refers to a general schema where the mind might use runnable mental models, akin to a physical simulator in a video game engine, to infer "physical hypotheses" that can explain the contents of the kinds of dynamic scenes we encounter in our visual environments. This idea can be made concrete in certain domains to explore its predictions and to empirically evaluate it against behavioral or neural measurmenets.

Here, we explore an implementation a model that uses the pybullet physics engine to simulate the trajectory of a ball given different hypotheses about it's [coefficient of restitution](https://en.wikipedia.org/wiki/Coefficient_of_restitution) and mass. We will also explore inference procedures, such as [particle filters](https://en.wikipedia.org/wiki/Particle_filter) that leverage the sequential nature of the underlying world model to efficiently update physical hypotheses.

Specifically, we will observe a sequence of the positions of a falling and bouncing ball, and based on that, infer its mass and bounciness.
"""

# ╔═╡ 0ba6a7ef-fd6c-4f80-a611-c4774a1767b4
md"""
## Part 1: The Physical Generative Model

The generative model (`model`) consists of a hypothesis over objects in the scene and uses the `bullet` physics engine to produce a sequence of predictions about the object state across time.

This part will codify the following diagram in a generative model -- we will remain at an abstract level not yet thinking about exactly what we are simulating. We will make things concrete in Part 2.

![physical markov chain](https://raw.githubusercontent.com/CNCLgithub/Algorithms-of-the-Mind/ec1ea73cc9cdd8dd198321221826fa6802bd7c51/labs/lab-06/media/phys_gm.png)

The generative model is split up into several small generative functions:

1. `prior` samples new latents (mass and restitution) defining the initial world state
2. `kernel` simulates T steps into the future, generating world states for each step
3. `observe` generates noisy observations over object positions (in $\mathcal{R}^3$) for a given state
"""

# ╔═╡ 7999589d-531f-4552-a490-7445657d3d2c
md"""
### The prior over object latents

The `prior` defines a distribution over what the physical latent's of an object *should* be. For the purposes of this tutorial, the Newtonian objects will only have two physical properties: mass and restitution (bounciness).

Below is a docstring describing `RigidBodyLatents` more thoroughly. 

> Note: at any point during this tutorial you can find more info about an element by either typing `?` before the name, using Pluto's "Live Docs" feature, or evaluating `@doc NAME`. 
"""

# ╔═╡ 79e2b0e8-9099-46a8-a4a3-29e62363d9e2
@doc RigidBodyLatents

# ╔═╡ 007878d3-bdec-4545-9f85-4f42b0556d00
"""
	update_latents(latents, mass, restitution) -> RigidBodyLatents

A helper function to update the state contained in a struct called RigidBodyLatents.
Notice that RigidBodyLatents contains all of the latent variables that we wish to make inferences about
"""
function update_latents(ls::RigidBodyLatents, mass::Float64, res::Float64)
    RigidBodyLatents(setproperties(ls.data;
                           mass=mass,
                           restitution=res))
end

# ╔═╡ 29a855f9-9756-4208-a5d0-b4bb27b7ff3b
md"""
The prior is defined over the mass and restitution of an object (that will fall from a height and bounce). Notice that these latents are stored in the `RigidBodyLatents` struct.
"""

# ╔═╡ cdb31120-5f91-4443-847c-f731c87b17a5
"""
	prior(template) -> RigidBodyLatents

The prior. Samples mass and restitution given a template set of other latents.
"""
@gen function prior(ls::RigidBodyLatents)
    mass ~ gamma(1.2, 10.)
    restitution ~ uniform(0, 1)
    new_latents = update_latents(ls, mass, restitution)
    return new_latents
end

# ╔═╡ 2768f6b2-d3c4-4f88-bd9c-22966014a32b
md"""
Below is an example of sampling from the `prior`.

Notice how each time you evaluate this function, you get a difference set of values.
"""

# ╔═╡ 1b474329-36bd-410d-adf4-230e30f53605
prior(RigidBodyLatents((mass = 0.5, restitution=0.5)))

# ╔═╡ 40ea2b25-12cd-4f31-8340-0cb25ae3da16
md"""
### The likelihood: How well does a state explain an observation?

Let's define both the state space, the observation space, and how they relate.
"""

# ╔═╡ bf0f6ac0-82f1-4927-826a-7beabd0f81dd
md"""
In this tutorial, the state space is define below. Essentially, for a given moment in time, an object has a 3D position, orientation, and velocity.
"""

# ╔═╡ 8d264def-de29-4fce-9925-f632cecae061
@doc RigidBodyState

# ╔═╡ bbdf7d87-336b-48c5-a8bd-33d8423daf8e
md"""
In this tutorial, the observation space is simple a noisy sample over the "ground-truth" 3D position of the ball.
"""

# ╔═╡ 9222f7bf-da91-4e19-8f45-ae81c6599e4a
@doc raw"""
	observe(state) -> XYZ

Defines a conditional distribution over the object's 3D position with ``\sigma=0.1``
"""
@gen function observe(state::RigidBodyState)
    position ~ broadcasted_normal(state.position, 0.1)
    return position
end

# ╔═╡ e6fbc85c-8c0b-4c20-99d2-07796d12db9f
md"""
It can be involved to setup an object state in isolation, so we will get to that in a moment after we define the `kernel`
"""

# ╔═╡ 08cfd3bb-704b-44c5-8d3d-b7eb4285695c
md"""
### The kernel
"""

# ╔═╡ c1e24a30-449f-4bff-84ab-a46e7441e271
md"""
Now let's move to the kernel below, it's only a couple of lines. 
"""

# ╔═╡ e06ed9ba-2620-4401-ab98-f505aad5251c
@doc raw"""

	kernel(t, prev_state, simulator) -> next_state

Advances the physical scene by one timestep and samples observations for that new state.
"""
@gen function kernel(t::Int, prev_state::BulletState, sim::BulletSim)
    # use of PhySMC.step to step the simulation forward one time step
    next_state::BulletState = PhySMC.step(sim, prev_state)
    # `next_state.kinematics = [RigidBodyState]` G
    positions ~ Gen.Map(observe)(next_state.kinematics)
    return next_state
end


# ╔═╡ af8aa73b-52b2-4f37-904a-e2bd2d4b2f44
md"""
Notice that the kernel makes a call to the physics engine to step the simulation one time step forward, using PhySMC.step(...). This implements temporal dynamics in the generative model, the horizontal arrow from the figure above.

(In brief, the package PhySMC provides an appropriate abstraction for interfacing probabilistic programming and physics engine states. Read the [PhySMC ReadMe](https://github.com/CNCLgithub/PhySMC/blob/master/README.md) for more information.)

The return value of this call, next_state is a kinematic state (position, velocity, orientation) of a rigid body (i.e., the object we are simulating).
"""

# ╔═╡ 24c0edff-adad-4661-b218-91998350ecf9
md"""

### The full model

Now we have all three pieces needed to build the full model. Let's take a look below
"""

# ╔═╡ d4391785-11be-4fd4-8d8c-ff6877be964f
@doc raw"""
	model(t, simulator, template_state) -> [RigidBodyStates]

Samples an initial set of latents from the `prior` and then simulates for `t` steps, using `kernel`.
"""
@gen function model(T::Int, sim::BulletSim, template::BulletState)
    # distribution over mass and restitution for objects from the prior
    latents ~ Gen.Map(prior)(template.latents)
    init_state = Accessors.setproperties(template; latents=latents)
    # simulate `T` timesteps; kind of like a cool for-loop
    states ~ Gen.Unfold(kernel)(T, init_state, sim)
    return states
end

# ╔═╡ 946e610e-b34f-4adc-aed3-738776e3246d
md"""
Ok, with the model ready, I bet you are itching to run some samples! First, we will need to prepare the simulation environment and a template scene.
"""

# ╔═╡ deeb7dd7-89fa-4b7c-ba99-a01dad2b970d
md"""
## The Physical Domain

To make the model concrete in a physical scenario, all we need to do is to initialize a scene configuration, which can then be simulated forward.

Here, we will use PhySMC and PhyBullet to initialize a simple scene: a table and a ball above its center.

Below is the implementation of `simple_scene`, a helper function that will do just that.


> NOTE: You do not need to familiarize yourself with the expressions in `simple_scene`, but please feel free to look if you are curious. Most of the commands are invoking the `pybullet` API.
"""

# ╔═╡ 3831ae87-8114-416f-aaae-8ae5dde70b62
"""
	simple_scene(mass=1.0, restitution=0.9) -> Tuple(client, ball)

Initializes a simple scene with a ball over a table.
"""
function simple_scene(mass::Float64=1.0, restitution::Float64=0.9)
    # `pb` is the `pybullet` python package
    # initialize a physics server in pybullet
    client = @pycall pb.connect(pb.DIRECT)::Int64
    # gotta set the gravity
    pb.setGravity(0,0,-10; physicsClientId=client)

    # add a table
    dims = [1.0, 1.0, 0.1] # in meters
    col_id = pb.createCollisionShape(pb.GEOM_BOX,
                                     halfExtents=dims,
                                     physicsClientId=client)
    obj_id = pb.createMultiBody(baseCollisionShapeIndex=col_id,
                                basePosition=[0., 0., -0.1],
                                physicsClientId=client)
    pb.changeDynamics(obj_id,
                      -1;
                      mass=0., # 0 mass means this object does not move (it's stationary)
                      restitution=0.9, # some is necessary; we could learn or infer this parameter
                      physicsClientId=client)


    # add a ball
    bcol_id = pb.createCollisionShape(pb.GEOM_SPHERE,
                                      radius=0.1,
                                      physicsClientId=client)
    bobj_id = pb.createMultiBody(baseCollisionShapeIndex=bcol_id,
                                 basePosition=[0., 0., 1.0],
                                 physicsClientId=client)
    pb.changeDynamics(bobj_id,
                      -1;
                      mass=mass,
                      restitution=restitution,
                      physicsClientId=client)

    (client, bobj_id)
end

# start with a ball above a table

# ╔═╡ 33f63f48-3327-41ec-af6e-9ea442d19a18
md"""
With that, we can now generate an initial scene, and configure all of the arguments to run `model`.

You will see three new data structures

- `BulletSim` : Parameters for using the Bullet physics engine
- `BulletState` : State for BulletSim
- `RigidBody` : A rigid body in BulletSim

Please feel free to inspect their docstrings for more info.
"""

# ╔═╡ fea43bf9-83eb-4747-b0de-847aec44ccc6
client, ball_id = simple_scene()


# ╔═╡ 4e61f9c5-0de6-4eb0-995c-36b5f7ad8c9c
# configure simulator with the provided
# client id
sim = BulletSim(; client=client)

# ╔═╡ d9e48ad2-0f5e-4319-9711-8edc7cb2a40f
# This is the object of interest in the scene
# (the table is static)
ball = RigidBody(ball_id)

# ╔═╡ ad6373b5-07d5-454f-ad50-3203924cfcf5
# Retrieve the default latents for the ball
# as well as its initial positions
# Note: alternative latents will be suggested by the `prior`
init_state = BulletState(sim, [ball])

# ╔═╡ b86a02b2-eaeb-46cc-854d-979419817ee9
# arguments for `model`
gargs = (60, # number of steps (total duration 1s)
         sim,
         init_state);

# ╔═╡ bda887d9-d24c-47c7-83e7-ab51991ef573
md"""
Ok, we are now ready to run the model forward for 60 steps!
"""

# ╔═╡ dc605663-fc83-404a-93b0-ca3d48abbf85
trace, _ = generate(model, gargs);

# ╔═╡ 8493e9f1-e248-4672-8386-b0cc0de1ef40
md"""
Well that was somewhat anti-climactic. Let's use this nifty plotting function to visualize our simulation!
"""

# ╔═╡ 6289ed15-e159-4760-b748-1228cf919bfd
begin
	@userplot SimPlot
	@recipe function f(cp::SimPlot)
	    z, t = cp.args
	    cs = size(z, 1)
	    k = 10
	    inds = (max(1, t-k):t)
	    n = length(inds)
	    linewidth --> range(0, 10, length = n)
	    seriesalpha --> range(0, 1, length = n)
	    xguide --> "time"
	    yguide --> "height of ball (z)"
	    ylims --> (0, 1.0)
	    xlims --> (1, 60)
	    label --> false
	    inds, z[inds, :]
	end
	
	function get_zs(trace::Gen.Trace)
	    t, _... = get_args(trace)
	    states = get_retval(trace)
	    zs = Vector{Float64}(undef, t)
	    for i = 1:t
	        zs[i] = states[i].kinematics[1].position[3]
	    end
	    return zs
	end

	function animate_trace(trace::Gen.Trace; label = "trace")
	    t = first(get_args(trace))
	    zs = reshape(get_zs(trace), (t, 1))
	    anim = @animate for i = 2:t
	        simplot(zs, i, label = label)
	    end
	end;
end

# ╔═╡ bdfef339-82cc-40a2-951e-0d835003c13a
begin
	anim = animate_trace(trace)
	gif(anim, fps = 24)
end

# ╔═╡ e642e7a7-b1ff-4136-84fe-c914dd74b911
md"""
Wow! that's neet. We can clearly see the ball bouncing around - and how much it bounces should depend on the restitution you sampled. 

By repeatedly running `generate` on `model`, we draw a variety of samples from the prior and thus traces. Here, we visualize how traces that sample a range of physical properties for the ball lead to different timecourses.
"""

# ╔═╡ 407b3a7c-4b70-46cb-ab9b-da5f297dbb4a
traces = [first(generate(model, gargs)) for _=1:7];

# ╔═╡ 4588488f-83c8-4880-9290-463c7c7b0b9f
function animate_traces(traces::Vector{<:Gen.Trace})
    n = length(traces)
    zzs = reduce(hcat, map(get_zs, traces))
    t = size(zzs, 1)
    anim = @animate for i=2:t
        simplot(zzs, i)
    end
end

# ╔═╡ bd481c82-99c7-44aa-b9c7-11dacb231070
gif(animate_traces(traces), fps=24)

# ╔═╡ 5df4fbbb-f483-428b-8e2c-f09a7b69a0ca
md"""
## Inference Over Dynamic Scenes

Now that we have implemented a generative model over the table scene, we can perform inferences in it given a set of observed positions

Let's generate a trajectory, and extract its noisy positions. 
"""

# ╔═╡ e21377fc-d136-4cac-9914-299acac72109
begin
	# First, let's generate a scene with a specific mass and restitution 
	# Our goal is to generate a simulated observation that we can work with
	gt_latents = choicemap(
	    (:latents => 1 => :restitution, 0.8), 
	    (:latents => 1 => :mass, 1.0)
	)
	gt = first(generate(model, gargs, gt_latents));
	gt_choices = get_choices(gt)
	
	t = gargs[1]
	
	# one set of observations per time step
	# (notice that these do not contain gt latents)
	observations = Vector{Gen.ChoiceMap}(undef, t)
	for i = 1:t
	    cm = choicemap()
	    addr = :states => i => :positions
	    set_submap!(cm, addr, get_submap(gt_choices, addr))
	    observations[i] = cm
	end
	
	gif(animate_trace(gt), fps=24)
end

# ╔═╡ 8a605295-3865-4fca-bef0-ca9172d3882e
begin
	"""A truncated normal distribution"""
	struct TruncNorm <: Gen.Distribution{Float64} end
	
	const trunc_norm = TruncNorm()
	
	function Gen.random(::TruncNorm, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
	    d = Distributions.Truncated(Distributions.Normal(mu, noise),
	                                low, high)
	    
	    return Distributions.rand(d)
	end;
	
	function Gen.logpdf(::TruncNorm, x::Float64, mu::U, noise::T, low::T, high::T) where {U<:Real,T<:Real}
	    d = Distributions.Truncated(Distributions.Normal(mu, noise),
	                                low, high)
	
	    return Distributions.logpdf(d, x)
	end;
end

# ╔═╡ 3658e062-b85e-4692-83c4-5846da10b624
"""
This proposal function implements a truncated random walk for mass and restitution
"""
@gen function proposal(tr::Gen.Trace)
    # HINT: https://www.gen.dev/tutorials/iterative-inference/tutorial#mcmc-2
    #
    # get previous values from `tr`
    choices = get_choices(tr)
    prev_mass = choices[:latents => 1 => :mass]
    prev_res  = choices[:latents => 1 => :restitution]
    
    # sample new values conditioned on the old ones
    # (Note: values are truncated to avoid issues with simulation)
    mass = {:latents => 1 => :mass} ~ trunc_norm(prev_mass, .1, 0., Inf)
    restitution = {:latents => 1 => :restitution} ~ trunc_norm(prev_res, .1, 0., 1.)
    
    # the return of this function is not
    # neccessary but could be useful
    # for debugging.
    return (mass, restitution)
end

# ╔═╡ d85a0ed5-341c-43d5-bef1-092d3daadd2e
"""
    inference_procedure

Performs particle filter inference with rejuvenation.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Vector{Gen.ChoiceMap},
                             particles::Int=20)
    get_args(t) = (t, gm_args[2:3]...)

    # initialize particle filter
    state = Gen.initialize_particle_filter(model, get_args(0), EmptyChoiceMap(), particles)
    argdiffs = (UnknownChange(), NoChange(), NoChange()) # only the first argument will change
    
    # Then increment through each observation step
    for (t, o) = enumerate(obs)
        # apply a rejuvenation move to each particle
        step_time = @elapsed begin
            for i=1:particles
                state.traces[i], _ = mh(state.traces[i], proposal, ())
            end
        
            Gen.maybe_resample!(state, ess_threshold=particles/2) 
            Gen.particle_filter_step!(state, get_args(t), argdiffs, o)
        end
    end

    return state.traces
    # return the "unweighted" set of traces after t steps
    # return Gen.sample_unweighted_traces(state, particles)
end

# ╔═╡ ba80f8f7-afb7-4ecb-94d0-904c2a777512
result = inference_procedure(gargs, observations); #should take a few seconds

# ╔═╡ c410e103-76ee-4b15-8172-a091c509fe42
gif(animate_traces(result), fps=24)

# ╔═╡ dfc169c5-c89d-487f-af59-3e2b2c9a7277
begin
	
	function get_latents(traces::Vector{<:Gen.Trace})
	    n = length(traces)
	    mass = Vector{Float64}(undef, n)
	    restitution = Vector{Float64}(undef, n)
	    for i = 1:n
	        mass[i] = traces[i][:latents => 1 => :mass]
	        restitution[i] = traces[i][:latents => 1 => :restitution]
	    end
	    (mass, restitution)
	end
	
	function plot_latents(traces::Vector{<:Gen.Trace})
	    mass, restitution = get_latents(traces)
	    res_plt = histogram(
	        restitution, title="Pr(restitution | Xs)", 
	        xlabel="restitution", label="traces",
			xlims = (0., 1.0),
	    )
	    vline!(res_plt, [0.8], label = "gt", linewidth=3) 
	    mass_plt = histogram(
	        mass, title="Pr(mass | Xs)",
	        xlabel="mass", bins=10, label="traces"
	    )
	    vline!(mass_plt, [1.0], label = "gt", linewidth=3) 
	    return plot(res_plt, mass_plt)
	end
	
	plot_latents(result)
	     
end

# ╔═╡ Cell order:
# ╟─d1678c83-d8cd-4b2c-8257-ff72300dcf00
# ╠═71eb1272-37f5-48db-b516-8e5bdbac8d7e
# ╟─4624cb2b-5767-4899-8991-560b74d10177
# ╟─7ebb39fe-451c-4f10-811f-3cdde5f71a55
# ╟─0ba6a7ef-fd6c-4f80-a611-c4774a1767b4
# ╟─7999589d-531f-4552-a490-7445657d3d2c
# ╠═79e2b0e8-9099-46a8-a4a3-29e62363d9e2
# ╟─007878d3-bdec-4545-9f85-4f42b0556d00
# ╟─29a855f9-9756-4208-a5d0-b4bb27b7ff3b
# ╠═cdb31120-5f91-4443-847c-f731c87b17a5
# ╟─2768f6b2-d3c4-4f88-bd9c-22966014a32b
# ╠═1b474329-36bd-410d-adf4-230e30f53605
# ╟─40ea2b25-12cd-4f31-8340-0cb25ae3da16
# ╟─bf0f6ac0-82f1-4927-826a-7beabd0f81dd
# ╠═8d264def-de29-4fce-9925-f632cecae061
# ╟─bbdf7d87-336b-48c5-a8bd-33d8423daf8e
# ╠═9222f7bf-da91-4e19-8f45-ae81c6599e4a
# ╟─e6fbc85c-8c0b-4c20-99d2-07796d12db9f
# ╟─08cfd3bb-704b-44c5-8d3d-b7eb4285695c
# ╟─c1e24a30-449f-4bff-84ab-a46e7441e271
# ╠═e06ed9ba-2620-4401-ab98-f505aad5251c
# ╟─af8aa73b-52b2-4f37-904a-e2bd2d4b2f44
# ╟─24c0edff-adad-4661-b218-91998350ecf9
# ╠═d4391785-11be-4fd4-8d8c-ff6877be964f
# ╟─946e610e-b34f-4adc-aed3-738776e3246d
# ╟─deeb7dd7-89fa-4b7c-ba99-a01dad2b970d
# ╟─3831ae87-8114-416f-aaae-8ae5dde70b62
# ╟─33f63f48-3327-41ec-af6e-9ea442d19a18
# ╠═fea43bf9-83eb-4747-b0de-847aec44ccc6
# ╠═4e61f9c5-0de6-4eb0-995c-36b5f7ad8c9c
# ╠═d9e48ad2-0f5e-4319-9711-8edc7cb2a40f
# ╠═ad6373b5-07d5-454f-ad50-3203924cfcf5
# ╠═b86a02b2-eaeb-46cc-854d-979419817ee9
# ╟─bda887d9-d24c-47c7-83e7-ab51991ef573
# ╠═dc605663-fc83-404a-93b0-ca3d48abbf85
# ╟─8493e9f1-e248-4672-8386-b0cc0de1ef40
# ╠═6289ed15-e159-4760-b748-1228cf919bfd
# ╠═bdfef339-82cc-40a2-951e-0d835003c13a
# ╟─e642e7a7-b1ff-4136-84fe-c914dd74b911
# ╠═407b3a7c-4b70-46cb-ab9b-da5f297dbb4a
# ╠═4588488f-83c8-4880-9290-463c7c7b0b9f
# ╠═bd481c82-99c7-44aa-b9c7-11dacb231070
# ╟─5df4fbbb-f483-428b-8e2c-f09a7b69a0ca
# ╠═e21377fc-d136-4cac-9914-299acac72109
# ╠═8a605295-3865-4fca-bef0-ca9172d3882e
# ╠═3658e062-b85e-4692-83c4-5846da10b624
# ╠═d85a0ed5-341c-43d5-bef1-092d3daadd2e
# ╠═ba80f8f7-afb7-4ecb-94d0-904c2a777512
# ╠═c410e103-76ee-4b15-8172-a091c509fe42
# ╠═dfc169c5-c89d-487f-af59-3e2b2c9a7277
