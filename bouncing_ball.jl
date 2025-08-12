### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 71eb1272-37f5-48db-b516-8e5bdbac8d7e
# Loads required packages
begin
    import Pkg
    # activate the project environment
    Pkg.activate(mktempdir())
    Pkg.instantiate()

	# Install pybullet via Conda.jl
	Pkg.add("Conda")
    using Conda
	Conda.add("pybullet", :AlgorithmsOfVisionCCN2025)

	# Build PyCall.jl with Conda env. (so it can import pybullet)
	pythonpath = joinpath(Conda.ROOTENV, "envs", "AlgorithmsOfVisionCCN2025", "bin", "python")
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

# ╔═╡ 7d05e447-467d-4e10-97a4-6d189aebdf8d
md"""
## Start with this: Tutorial environment setup.
The following code block loads the required packages (e.g., `pybullet` for physics simulation, `Gen` for probabilistic programming, `Plots` for visualizations) and sets up a Conda environment for the rest of this tutorial.
"""

# ╔═╡ 4624cb2b-5767-4899-8991-560b74d10177
md"""
# Visual inference using a physics-based generative model
"""

# ╔═╡ 7ebb39fe-451c-4f10-811f-3cdde5f71a55
md"""
This tutorial introduces a computational model that infers "physics-based representations" of scenes; in other words, the model explains visual inputs in terms of their underlying physical causes. To implement these structure-preserving representations of physical scenes, we build on the idea of a "physics engine in the mind" (introduced, in large part, by [Battaglia et al. (2013)](https://www.pnas.org/doi/pdf/10.1073/pnas.1306572110)), which refers to a general schema where the mind might use runnable mental models, akin to a physics simulator in a video game engine. One way to make this idea concrete (i.e., implement in empirically testable computational models) is to embed off-the-shelf physics engines in probabilistic models. 

Here, we explore an implementation that uses the probabilistic programming package `Gen` and the physics engine `pybullet`. We will build a generative model that defines a probability distribution over the trajectories of a ball, conditioned on the ball's [coefficient of restitution (or, bounciness)](https://en.wikipedia.org/wiki/Coefficient_of_restitution) and mass. We will also explore approximate Bayesian inference procedures, such as [particle filters](https://en.wikipedia.org/wiki/Particle_filter) that leverage the sequential nature of the underlying generative model to efficiently update physics-based representations. 

Specifically, we will observe (i.e., take as input) a sequence of the positions of a falling and bouncing ball, and based on that, infer its bounciness and mass.
"""

# ╔═╡ a2e740d7-01cd-4c00-963b-f8d5f7b396c6
md"""
## The software tools

Before we start, here is an overview of the software tools that we will be using to build our model.

- [Gen](https://www.gen.dev/): An open-source stack for generative modeling and probabilistic inference. (Think of Gen as what pytorch is for deep learning.)
- [bullet3 physics engine](https://github.com/bulletphysics/bullet3) via [pybullet](https://pypi.org/project/pybullet/): real-time collision detection and multi-physics simulation for VR, games, visual effects, robotics, machine learning etc.
- [PhySMC](https://github.com/CNCLgithub/PhySMC) and [PhyBullet](https://github.com/CNCLgithub/PhyBullet): two companion packages that provide appropriate abstractions to integrate off-the-shelf physics simulation (such as pybullet) into probablistic programming. Feel free to read the ReadMe of these packages.
"""

# ╔═╡ deeb7dd7-89fa-4b7c-ba99-a01dad2b970d
md"""
## The Physical scenario: Bouncing balls

To make the model concrete in a physical scenario, all we need to do is to initialize a scene configuration, which can then be simulated forward.

Here, we will work with a simple initial scene configuration: a table and a ball above its center.

Below is the implementation of `simple_scene`, a helper function that will do just that. You do not need to familiarize yourself with or understand the expressions in `simple_scene`, but please feel free to look if you are curious (by toggling the visibility of the next code block). Most of the commands in it invoke the `pybullet` API.
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

# ╔═╡ ce4056a4-f3b8-4bb3-a27b-01498f9a80de
md"""
Now let's initialize a simple scene!:
"""

# ╔═╡ fea43bf9-83eb-4747-b0de-847aec44ccc6
client, ball_id = simple_scene()


# ╔═╡ 33f63f48-3327-41ec-af6e-9ea442d19a18
md"""
Once initialized, a scene configuration can be simulated forward in time using the physics engine.

The following three data structures, which are part of the PhySMC and PhyBullet packages, will help us keep track of the simulation for probabilistic inference: 

- `BulletSim` : Parameters for using the Bullet physics engine (implemented in PhySMC)
- `RigidBody` : A rigid body in BulletSim (implemented in PhyBullet)
- `BulletState`: State for BulletSim, including that of the RigidBody (implemented in PhySMC)

Feel free to inspect their docstrings for more info.

> Note: at any point during this tutorial you can find more info about an element by either typing `?` before the name, using Pluto's "Live Docs" feature, or evaluating `@doc NAME`. 
"""

# ╔═╡ 2a748536-3762-4bfd-a015-5666fc98d9e3
md"""
We first instantiate the simulator data structure (`BulletSim`) with the client id provided by pybullet.
"""

# ╔═╡ 4e61f9c5-0de6-4eb0-995c-36b5f7ad8c9c
# configure simulator with the provided
# client id
sim = BulletSim(; client=client)

# ╔═╡ 74562148-4b9a-4606-a767-637de765c131
md"""
We then instantiate the `RigidBody` data structure using the `ball_id` provided by pybullet. 
"""

# ╔═╡ d9e48ad2-0f5e-4319-9711-8edc7cb2a40f
# This is the object of interest in the scene
# (the table is static)
ball = RigidBody(ball_id)

# ╔═╡ bef8a0b5-a720-4758-ab09-0e210bd4b8cc
md"""
Finally, we instantiate the `BulletState` data structure, which consists of both the physical properties (e.g., mass, bounciness) and scene configuration (e.g., position, velocity).
"""

# ╔═╡ ad6373b5-07d5-454f-ad50-3203924cfcf5
# Retrieve the default latents for the ball
# as well as its initial positions
# Note: we will override these latents using the `prior`
init_state = BulletState(sim, [ball])

# ╔═╡ d67790d2-f5ca-44e7-a3ec-68bb0223c149
md"""
We will use the helper function `simulate_scene` to advance the simulation forward. 
"""

# ╔═╡ d5617124-9d25-4402-b44e-525071c44662
"""
	simulate_scene(init_state, t) -> [BulletState]

Runs physics for `t` steps returning a `Vector` of `t+1` states.
"""
function simulate_scene(init_state::BulletState, steps::Int=60)
	states = Vector{BulletState}(undef, steps+1)
	states[1] = init_state
	for i = 2:(steps+1)
		# Use PhySMC and PhyBullet to simulate a step
		states[i] = PhySMC.step(sim, states[i-1])
	end
	return states
end

# ╔═╡ e3e52efb-c292-4eb1-9391-be2fd214bdec
md"""
Let's run the simulation for 60 time steps.
"""

# ╔═╡ 7d8d2ef3-6053-49cf-8a2a-dcacbdb704de
# Simulating 60 steps.
states = simulate_scene(init_state, 60)

# ╔═╡ 143ccaa8-fe81-429b-aa51-c34b65e827f0
md"""
With some animation functions, let's take a look at the timecourse of the bouncing ball. Here, we are visualizing the height of the ball (y-axis) as a function of time (x-axis)
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
	function get_zs(states::Vector)
		t = length(states)
	    zs = Vector{Float64}(undef, t)
	    for i = 1:t
	        zs[i] = states[i].kinematics[1].position[3]
	    end
	    return zs
	end
	function animate_trace(trace::Vector; label = "trace")
	    t = length(trace)
	    zs = reshape(get_zs(trace), (t, 1))
	    anim = @animate for i = 2:t
	        simplot(zs, i, label = label)
	    end
	end
end

# ╔═╡ 7391bc8c-0af6-4e71-8634-7c987a1cb13f
md"""
Now compare that scene to the one below: What *jumps* out? 
"""

# ╔═╡ 2126a158-d8e6-4035-8115-6a8516bf0d12
md"""
We clearly see a difference in their trajectories --- the different heights each ball reaches. But, it is also possible that we just as well see a much higher property --- what has caused these two different trajectories: The different physical properties, i.e., bounciness, of each ball. You can almost *hear* the distinct sound each collision would make on the table. 

Our goal in this notebook is to implement a psychologically plausible algorithm that can model such human perception of physical properties. We'll do so using probabilistic modeling.

We will accomplish this in two parts:

1. Specify a generative model: the conditional distributions over how objects move and appear
2. Implementing a psychologically plausible inference algorithm that updates the states and latents of objects given a sequence of observations
"""

# ╔═╡ 0ba6a7ef-fd6c-4f80-a611-c4774a1767b4
md"""
## Part 1: The Physical Generative Model

The generative model (`model`) takes as input the initial scene configuration (a ball positioned above a stable table), and defines conditional distributions over how objects move (uses the [bullet3 physics engine](https://github.com/bulletphysics/bullet3)) and appear, producing a sequence of predictions about the object state across time. 

The following diagram illustrates these conditional relationships.

![physical markov chain](https://raw.githubusercontent.com/CNCLgithub/AlgorithmsOfVisionCCN2025/refs/heads/bouncing_ball/physics_chain.svg)

More formally, this generative model [i.e., a joint distribution over latent variables (i.e., objects' physical properties) and observations (i.e., sensory measurements)] can be defined in the following mathematical expression: 

```math
Pr(\vec{S}, \vec{X}) = Pr(S_0) \prod\limits_{t=1}^{T} Pr(X_t \mid S_t) \cdot Pr(S_t \mid S_{t-1})
```

This generative model is factorized into three conditional probability distributions, each of which will be implemented using a `Gen` generative function (more on that below):

1. `prior`: $Pr(S_0)$ samples new latents from a prior distribution. In our example, this consists of prior distributions over the latent variables mass and bounciness of the ball.
2. `observe`: $Pr(X_t \mid S_t)$ generates noisy observations over object positions (in $\mathcal{R}^3$) for a given scene configuration.
3. `kernel`: $Pr(S_t \mid S_{t-1})$ simulates T steps into the future, generating scene configurations for each step.

Let's see how this generative model can be implemented using the general-purpose probabilistic programming package Gen.

"""

# ╔═╡ 7999589d-531f-4552-a490-7445657d3d2c
md"""
### The prior over latent variables (i.e., an object's physical properties), $Pr(S_0)$

The `prior` defines a distribution over what the physical properties of an object *should* be. For the purposes of this tutorial, the prior will implement only two object properties: mass and bounciness (assigning the rest deterministically).

The value of this prior will be stored in a new data structure called `RigidBodyLatents`. See below for a docstring that provides a more thorough description of `RigidBodyLatents`, which consists of both the latents we define a prior distribution over (mass and bounciness) and several others.

"""

# ╔═╡ 79e2b0e8-9099-46a8-a4a3-29e62363d9e2
@doc RigidBodyLatents

# ╔═╡ 8fe3f63e-6108-49f9-a32c-a6b91c16ccf1
md"""
We manipulate the values of these latent variables using the helper function `update_latents`.
"""

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
Now with this new data structure, we are ready to define this prior distribution over the mass and bounciness of a ball (that'll, in a bit, fall from a height and bounce on a stable table) using `Gen`. 
"""

# ╔═╡ cdb31120-5f91-4443-847c-f731c87b17a5
"""
	prior(template) -> RigidBodyLatents

The prior. Samples mass and restitution given a template set of other latent variables.
"""
@gen function prior(ls::RigidBodyLatents)
    mass ~ gamma(1.2, 10.)
    restitution ~ uniform(0, 1)
    new_latents = update_latents(ls, mass, restitution)
    return new_latents
end

# ╔═╡ 1b38a766-fa23-4b8b-9561-f46b2fa30d98
md"""
Notice that the `prior` is a special kind of function: it's a generative function (indicated by the macro `@gen` that precedes its definition) whose evaluations are stochastic, resulting in different values each time it's run. We'll experience this in a moment.

Specifically, this generative function defines mass as distributed with a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) (example below) and restitution as coming from a uniform distribution (between the values 0 and 1).

(Note: Here is a helpful [tool](https://distribution-explorer.github.io/) for exploring probability distributions.)
"""

# ╔═╡ de7092e7-5a74-4826-ab1a-1cd5ccb2420f
html"""
<div style="background-color: white;">
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Gammapdf252.svg?download" alt="Example of gamma distribution (credit Wikipedia)">
</div>
"""


# ╔═╡ 2768f6b2-d3c4-4f88-bd9c-22966014a32b
md"""
Now let's call the function `prior` to draw samples from it.

As mentioned above, notice how each time you evaluate this function, you get a different set of values. Under the hood, each evaluation of such generative functions creates a Gen "trace", which tracks every stochastic decision made during execution. (In this tutorial, we will not go into the details of the "trace" data structure, but it's the key data structure that enables performant, general-purpose probabilistic inference in Gen.)  

"""

# ╔═╡ 1b474329-36bd-410d-adf4-230e30f53605
prior(RigidBodyLatents((mass = 0.5, restitution=0.5)))

# ╔═╡ 1dc9c341-0933-490d-b1dd-b8d9267aa46b
md"""
(As a technical note, when calling the `prior` function, we pass in a template `RigidBodyLatents` data structure to define any latents we do not want changed. Only the mass and restitution are overwritten by the `prior` function.)
"""

# ╔═╡ 40ea2b25-12cd-4f31-8340-0cb25ae3da16
md"""
### The likelihood: How well does a state explain an observation, $Pr(X_t \mid S_t)$

Recall that the generative model takes as input an initial scene configuration. At each time step, the physical properties of an object cause changes to this scene configuration. For example, the ball moves under the influence of gravity at each time step. 

Let's define both the scene configuration, the observation space, and how they relate.
"""

# ╔═╡ bf0f6ac0-82f1-4927-826a-7beabd0f81dd
md"""
For each time step, the scene configuration consists of the 3D positions, orientations, and velocities of objects.
"""

# ╔═╡ 8d264def-de29-4fce-9925-f632cecae061
@doc RigidBodyState

# ╔═╡ bbdf7d87-336b-48c5-a8bd-33d8423daf8e
md"""
In this tutorial, we assume that the observation space is simply a noisy version of the "ground-truth" 3D position of the ball.
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

# ╔═╡ 7361d121-e0aa-416c-9df7-c420978d75c7
md"""
Like `prior`, notice that the `observe` function is also a generative function (indicated by the preceding `@gen` macro), meaning that its executions are stochastic and Gen keeps track of all the stochastic decisions under the hood. 
"""

# ╔═╡ ccfb34c0-af6d-4294-8166-c1d4adbc9b16
md"""

#### A note about the observation space

You might be thinking that this is a contrived observation space - and we agree! The `observe` generative function can readily incorporate pixel-based (or other sensor computable) observation spaces. However, using them in this demo would incur unnecessary technical complications (e.g., need for gpus).  

Instead, below, we provide some code from real projects that use such sensor-computable observation spaces. 

---


From [Woven](https://github.com/CNCLgithub/Woven/blob/82cfcfd7b9f51899df4f0cdcf2ae18dae466d38a/src/model/generative_model.jl#L158-L161) as published in: Computational models reveal that intuitive physics underlies visual processing of soft objects, Bi et al., 2025, Nature Communications

---

From [GranularScenes](https://github.com/CNCLgithub/GranularScenes/blob/7bb9dd9809c1c72d1be320909746df0aa8581c9f/src/gm/qt_model_gen.jl#L35-L47), Goal-conditioned world models: adaptive computation over multi-granular generative models explains human scene perception, Belledonne & Yildirim, manuscript in prep.

```julia
@gen function qt_model(t::Int, params::QuadTreeModel)
    # sample quad tree
    root::QTAggNode = {:trackers} ~ quad_tree_prior(params.start_node, 1)
    qt::QuadTree = QuadTree(root)

    # first image (X_0)
    {:img_a} ~ observe_pixels(params.renderer, qt, params.pixel_var)

    # second image (X_1) (could be the same image)
    changes ~ Gen.Unfold(qt_change_kernel)(t, qt, params)

    return qt
end
```

---

From [ThreeDP3](https://github.com/probcomp/ThreeDP3/blob/main/src/model/sg_model.jl#L188-L201), as published in 3DP3: 3D scene perception via probabilistic programming, Gothoskar et al., 2021, NeurIPS
```julia
@gen (static) function scene(model_params::SceneModelParameters)
    scene_graph ~ scene_graph_prior(model_params)
    rendered_clouds = render_clouds(model_params, scene_graph)
    p_outlier ~ exponential(1/model_params.hyperparams.p_outlier)
    noise ~ exponential(1/model_params.hyperparams.noise)
    obs_cloud = {:obs} ~ uniform_mixture_from_template_multi_cloud(
                                                rendered_clouds,
                                                p_outlier,
                                                noise,
                                                (-100.0,100.0,-100.0,100.0,-100.0,300.0))
    return (scene_graph=scene_graph,
            rendered_clouds=rendered_clouds,
            obs_cloud=obs_cloud)
end
```

---
"""

# ╔═╡ 08cfd3bb-704b-44c5-8d3d-b7eb4285695c
md"""
### The kernel: how objects move, $Pr(S_t \mid S_{t-1})$
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

The return value of this call, next_state is a scene configuration (position, velocity, orientation) of a rigid body (i.e., the object we are simulating). (Such non-dynamical scene configuration information is also called the kinematics state.)
"""

# ╔═╡ 24c0edff-adad-4661-b218-91998350ecf9
md"""

### The full generative model: $Pr(S_0) \prod\limits_{t=1}^{T} Pr(X_t \mid S_t) \cdot Pr(S_t \mid S_{t-1})$

Now we have all three pieces needed to build the full generative model. Let's take a look below.
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

# ╔═╡ b86a02b2-eaeb-46cc-854d-979419817ee9
# arguments for `model`
gargs = (60, # number of steps (total duration 1s)
         sim,
         init_state);

# ╔═╡ bda887d9-d24c-47c7-83e7-ab51991ef573
md"""
Ok, we are now ready to run the model forward for 60 steps!

We will use the `Gen.generate()` function to do so. This function takes as input a generative function (here: `model`) and its arguments (here: `gargs`), and returns a trace of the execution of this generative function (a record of all the stochastic decisions made during the call of the generative function).
"""

# ╔═╡ dc605663-fc83-404a-93b0-ca3d48abbf85
trace, _ = Gen.generate(model, gargs);

# ╔═╡ 8493e9f1-e248-4672-8386-b0cc0de1ef40
md"""
Well that was somewhat anti-climactic. Let's use this nifty plotting function to visualize our simulation!
"""

# ╔═╡ 965ed681-fda8-41c7-b704-c8351f1888fd
begin
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
	end
end

# ╔═╡ fd43db42-e095-449a-a41b-80b5656ca2ed
begin
	anim = animate_trace(states; label="Simulation 1")
	gif(anim, fps = 24)
end

# ╔═╡ 87b0cfec-94a5-4ab7-97dd-bf8de0bf52b1
begin
	client2, ball_id2 = simple_scene(1.0, 0.5)
	ball2 = RigidBody(ball_id2)
	sim2 = BulletSim(; client=client2)
	init_state2 = BulletState(sim2, [ball2])
	states2 = simulate_scene(init_state2, 60)
	anim2 = animate_trace(states2; label="Simulation 2")
	gif(anim2, fps = 24)
end

# ╔═╡ 68f82c0e-ea49-43df-95df-5eeec790bf8a
gif(animate_trace(trace), fps = 24)

# ╔═╡ e642e7a7-b1ff-4136-84fe-c914dd74b911
md"""
That's neat! We can see the ball bouncing around - and how much it bounces should depend on the restitution you sampled. 

By repeatedly running `generate` on `model`, we draw different samples from our generative model, which are stored in `trace`. Here, we visualize the traces of a range of physical properties for the ball and see how each leads to a different timecourse.
"""

# ╔═╡ 407b3a7c-4b70-46cb-ab9b-da5f297dbb4a
traces = [first(Gen.generate(model, gargs)) for _=1:7];

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
## Part 2: Inference Over Dynamic Scenes, $Pr(\vec{S} \mid \vec{X})$

Let's return to the main question: How can we infer the physical latents of the scene given a sequence of observations?

So far, we implemented a generative model, which is the joint distribution $Pr(\vec{S}, \vec{X})$.

Now by the [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes'_theorem), we can condition this generative model on observations to get an approximation of the posterior $Pr(\vec{S} \mid \vec{X})$.

Let's look at our first simulation and extract noisy positions. These will serve as our observations, $\vec{X}$. Take note of the ground truth latents for mass and restitution (1.0 and 0.8, respectively)- we will ultimately want to compare the inferences of our model (the posterior over physical properties, $Pr(\vec{S}|\vec{X})$) to these values.
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

# ╔═╡ db29e242-5029-48be-a76c-b0868060cc1c
md"""
## The inference procedure - A particle filter

Now there are many inference procedures we could use to arrive at this posterior, $Pr(\vec{S}|\vec{X})$. A standard work-horse in Bayesian inference is the [Metropolis-Hastings Markov Chain Monte-Carlo algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm). However, this algorithm is not particularly desirable in this case, as it requires all observations to be received simultaneously, instead of receiving them sequentially. 

A far more natural (but by no means perfect) algorithm is the [particle filter](https://www.stats.ox.ac.uk/~doucet/smc_resources.html), which excels at posteriors that can be factorized sequentially. 

Indeed, our physical setting factorizes along time:

```math
Pr(\vec{S} \mid \vec{X}) \propto Pr(S_0) \cdot \prod\limits_{t=1}^{T} Pr(S_t \mid X_t) \cdot Pr(S_t \mid S_{t-1})
```


Each particle in a particle filtering algorithm is an independent trace of the generative model, conditioned on the observations received so far. Together, these particles form a non-parametric approximation of the posterior over the object's physical properties.

For each incoming observation (i.e., at each time step), the particle filter has three steps:

![Particle filter diagram](https://raw.githubusercontent.com/CNCLgithub/AlgorithmsOfVisionCCN2025/refs/heads/bouncing_ball/particle_filter.svg)

1. **Update**: each particle samples (via `kernel`) the next state of the scene, $S_{t+1}$, weighting both the prior probability of that transition $Pr(S_{t+1} \mid S_t)$ as well as the likelihood $Pr(X_{t+1} \mid S_{t+1})$. 
2. **Resample**: A genetic-like pruning procedure, where particles are drawn, with replacement, from a multinomial distribution based on the normalized log-scores from step 1.
3. **Rejuvenation**: Each surviving particle receives a series of probabilistic adjustments to object latents using a `proposal` function. This procedure keeps each adjustment with a probability proportional to how likely it's under the `proposal` function and how much it improves the current likelihood of the observation (aka the [Metropolis-Hastings acceptance function](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)). 
"""

# ╔═╡ 733276b9-7f00-432c-bf7e-fb9e8058892d
md"""
### The proposal function

Let's start with the `proposal` function. This gets used during the rejuvenation phase of the particle filter.

The function takes a trace of the model and draws a sample for mass and resitution around the current guess in the trace. 

The `proposal` generative function uses a truncated normal distribution to prevent certain values (e.g., negative values for the mass) that would not make sense in the current context.
"""

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
This proposal function implements a random walk for mass and restitution.

The internal distribution is truncated to prevent physically impossible values (e.g., negative mass).
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
    mass = {:latents => 1 => :mass} ~ trunc_norm(prev_mass, 1.0, 0., Inf)
    restitution = {:latents => 1 => :restitution} ~ trunc_norm(prev_res, .1, 0., 1.)
    
    # the return of this function is not
    # neccessary but could be useful
    # for debugging.
    return (mass, restitution)
end

# ╔═╡ f303dbb1-1b59-4712-b197-eb2f6c7c9ad0
md"""
### The particle filter "for loop"

With the proposal defined, we can now implement the particle filter. 

Gen already provides an implementation for steps 1 and 2, we just need to put them all together.
"""

# ╔═╡ d85a0ed5-341c-43d5-bef1-092d3daadd2e
"""
    inference_procedure

Performs particle filter inference with rejuvenation.
"""
function inference_procedure(gm_args::Tuple,
                             obs::Vector{Gen.ChoiceMap},
                             particles::Int=20,
							 rejuv_moves::Int=2)
    get_args(t) = (t, gm_args[2:3]...)

    # initialize particle filter
    state = Gen.initialize_particle_filter(model, get_args(0), EmptyChoiceMap(), particles)
    argdiffs = (UnknownChange(), NoChange(), NoChange()) # only the first argument will change
    
    # Then increment through each observation step
    for (t, o) = enumerate(obs)
		# STEP 1: update
		Gen.particle_filter_step!(state, get_args(t), argdiffs, o)
		# STEP 2: resample
		Gen.maybe_resample!(state, ess_threshold=particles/2) 
        # STEP 3: rejuvenation
        for i=1:particles, s=1:rejuv_moves
            state.traces[i], _ = mh(state.traces[i], proposal, ())
		end
    end

    # return state.traces
    # return the "unweighted" set of traces after t steps
    return Gen.sample_unweighted_traces(state, particles)
end

# ╔═╡ f7ef47b3-1c52-4e8f-a4b4-7ef2a4bf83f1
md"""
### Inference Results

Now let's run the particle fitler with 20 particles (and 2 rejuvination moves per particle / step) across each of the 60 observations.
"""

# ╔═╡ ba80f8f7-afb7-4ecb-94d0-904c2a777512
result = inference_procedure(gargs, observations); #should take a 5-15 seconds

# ╔═╡ 87aa3c1c-4ce4-4b6f-ab6a-7b889c526e96
md"""
To visualize inference results, lets animate each particle after conditioning on all observations.
"""

# ╔═╡ c410e103-76ee-4b15-8172-a091c509fe42
gif(animate_traces(result), fps=24)

# ╔═╡ fa120fc0-9193-4c88-a80f-e4fea8a5827a
md"""
Recall the inference task, we wanted to infer the mass and restitution of the object given the series of noisy position observations. 

Let's look at the marginal of each latent - that is the distribution of restitution considering any value of mass, and vice-versa. 
"""

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
	
	function plot_latents(traces::Vector{<:Gen.Trace}; gt_mass = 1.0,
						 gt_res = 0.8)
	    mass, restitution = get_latents(traces)
	    res_plt = histogram(
	        restitution, title="Pr(restitution | Xs)", 
	        xlabel="restitution", label="traces",
			xlims = (0., 1.0),
			bins=3
	    )
	    vline!(res_plt, [gt_res], label = "gt", linewidth=3) 
	    mass_plt = histogram(
	        mass, title="Pr(mass | Xs)",
	        xlabel="mass", bins=10, label="traces"
	    )
	    vline!(mass_plt, [gt_mass], label = "gt", linewidth=3) 
	    return plot(res_plt, mass_plt)
	end
	
	plot_latents(result)
	     
end

# ╔═╡ 72319312-100b-4053-aafa-4ba20acf4998
md"""
Note how resitution is almost dead on the ground truth (0.8), whereas mass is all over the place (the ground truth was 1.0). 

Why is this the case? (Discuss amongst yourselves)
"""

# ╔═╡ c4854f0d-57b4-4d30-884f-e6273c416d9c
md"""
## Comparing two scenes

Ok, we inferred a posterior distribution over physical latents for the first scene - let's now do the same for the second. As a reminder, here is the trajecotory for the second scene.
"""

# ╔═╡ b2d3f1ba-d1f6-4de8-bbb0-9e9f6d65e7e2
begin
	# First, let's generate a scene with a specific mass and restitution 
	# Our goal is to generate a simulated observation that we can work with
	gt_latents2 = choicemap(
	    (:latents => 1 => :restitution, 0.5), 
	    (:latents => 1 => :mass, 1.0)
	)
	gt2 = first(generate(model, gargs, gt_latents2));
	gt_choices2 = get_choices(gt2)
		
	# one set of observations per time step
	# (notice that these do not contain gt latents)
	observations2 = Vector{Gen.ChoiceMap}(undef, t)
	for i = 1:t
	    cm = choicemap()
	    addr = :states => i => :positions
	    set_submap!(cm, addr, get_submap(gt_choices2, addr))
	    observations2[i] = cm
	end
	
	gif(animate_trace(gt2), fps=24)
end

# ╔═╡ a300228c-df39-4f13-b47c-28a797dcb43a
result2 = inference_procedure(gargs, observations2); #should take a 5-15 seconds

# ╔═╡ 64e76a4f-b9ac-4899-b1d9-9e14ce9ae540
gif(animate_traces(result2), fps=24)

# ╔═╡ 4062274b-f208-460e-9efa-15ef770ee13b
plot_latents(result2; gt_res = 0.5)

# ╔═╡ 0ce787c6-3e8e-428d-9964-9d9ebb7262fd
md"""
And to remind ourselves, here is scene 1
"""

# ╔═╡ 34a2d209-5280-45e0-ba43-b5d6eb01a48b
plot_latents(result)

# ╔═╡ f064467b-b9ff-4938-84f0-8df7ef7d42dd
md"""
Look at that! Notice how the model explains the difference in trajectory do to differences in restitution, with the second scene having lower bounciness.
"""

# ╔═╡ 46d392ce-dea6-4a99-9a02-6c7e97548b9f
md"""
## Summary and Future Reading

You've coded a basic model that infers structure-preserving representations using probabilistic programming. You can continue your learning through the following resources.

* We encourage you to read the [Woven paper (Bi et al., 2025)](https://www.nature.com/articles/s41467-025-61458-x) and explore its [codebase](https://github.com/CNCLgithub/Woven).
* Explore the official [Gen tutorials](https://www.gen.dev/tutorials/).
* Explore lab sections of the [Algorithms of the Mind](https://github.com/CNCLgithub/Algorithms-of-the-Mind/tree/main/labs) course.

We also appreciate your feedback to make this tutorial better for its future editions: XXX
"""

# ╔═╡ Cell order:
# ╟─7d05e447-467d-4e10-97a4-6d189aebdf8d
# ╠═71eb1272-37f5-48db-b516-8e5bdbac8d7e
# ╟─4624cb2b-5767-4899-8991-560b74d10177
# ╟─7ebb39fe-451c-4f10-811f-3cdde5f71a55
# ╟─a2e740d7-01cd-4c00-963b-f8d5f7b396c6
# ╟─deeb7dd7-89fa-4b7c-ba99-a01dad2b970d
# ╟─3831ae87-8114-416f-aaae-8ae5dde70b62
# ╟─ce4056a4-f3b8-4bb3-a27b-01498f9a80de
# ╠═fea43bf9-83eb-4747-b0de-847aec44ccc6
# ╟─33f63f48-3327-41ec-af6e-9ea442d19a18
# ╟─2a748536-3762-4bfd-a015-5666fc98d9e3
# ╠═4e61f9c5-0de6-4eb0-995c-36b5f7ad8c9c
# ╟─74562148-4b9a-4606-a767-637de765c131
# ╠═d9e48ad2-0f5e-4319-9711-8edc7cb2a40f
# ╟─bef8a0b5-a720-4758-ab09-0e210bd4b8cc
# ╠═ad6373b5-07d5-454f-ad50-3203924cfcf5
# ╟─d67790d2-f5ca-44e7-a3ec-68bb0223c149
# ╠═d5617124-9d25-4402-b44e-525071c44662
# ╟─e3e52efb-c292-4eb1-9391-be2fd214bdec
# ╠═7d8d2ef3-6053-49cf-8a2a-dcacbdb704de
# ╟─143ccaa8-fe81-429b-aa51-c34b65e827f0
# ╟─6289ed15-e159-4760-b748-1228cf919bfd
# ╠═fd43db42-e095-449a-a41b-80b5656ca2ed
# ╟─7391bc8c-0af6-4e71-8634-7c987a1cb13f
# ╠═87b0cfec-94a5-4ab7-97dd-bf8de0bf52b1
# ╟─2126a158-d8e6-4035-8115-6a8516bf0d12
# ╟─0ba6a7ef-fd6c-4f80-a611-c4774a1767b4
# ╟─7999589d-531f-4552-a490-7445657d3d2c
# ╠═79e2b0e8-9099-46a8-a4a3-29e62363d9e2
# ╟─8fe3f63e-6108-49f9-a32c-a6b91c16ccf1
# ╟─007878d3-bdec-4545-9f85-4f42b0556d00
# ╟─29a855f9-9756-4208-a5d0-b4bb27b7ff3b
# ╠═cdb31120-5f91-4443-847c-f731c87b17a5
# ╟─1b38a766-fa23-4b8b-9561-f46b2fa30d98
# ╟─de7092e7-5a74-4826-ab1a-1cd5ccb2420f
# ╟─2768f6b2-d3c4-4f88-bd9c-22966014a32b
# ╠═1b474329-36bd-410d-adf4-230e30f53605
# ╟─1dc9c341-0933-490d-b1dd-b8d9267aa46b
# ╟─40ea2b25-12cd-4f31-8340-0cb25ae3da16
# ╟─bf0f6ac0-82f1-4927-826a-7beabd0f81dd
# ╠═8d264def-de29-4fce-9925-f632cecae061
# ╟─bbdf7d87-336b-48c5-a8bd-33d8423daf8e
# ╠═9222f7bf-da91-4e19-8f45-ae81c6599e4a
# ╟─7361d121-e0aa-416c-9df7-c420978d75c7
# ╟─ccfb34c0-af6d-4294-8166-c1d4adbc9b16
# ╟─08cfd3bb-704b-44c5-8d3d-b7eb4285695c
# ╟─c1e24a30-449f-4bff-84ab-a46e7441e271
# ╠═e06ed9ba-2620-4401-ab98-f505aad5251c
# ╟─af8aa73b-52b2-4f37-904a-e2bd2d4b2f44
# ╟─24c0edff-adad-4661-b218-91998350ecf9
# ╠═d4391785-11be-4fd4-8d8c-ff6877be964f
# ╟─946e610e-b34f-4adc-aed3-738776e3246d
# ╠═b86a02b2-eaeb-46cc-854d-979419817ee9
# ╟─bda887d9-d24c-47c7-83e7-ab51991ef573
# ╠═dc605663-fc83-404a-93b0-ca3d48abbf85
# ╟─8493e9f1-e248-4672-8386-b0cc0de1ef40
# ╟─965ed681-fda8-41c7-b704-c8351f1888fd
# ╠═68f82c0e-ea49-43df-95df-5eeec790bf8a
# ╟─e642e7a7-b1ff-4136-84fe-c914dd74b911
# ╠═407b3a7c-4b70-46cb-ab9b-da5f297dbb4a
# ╟─4588488f-83c8-4880-9290-463c7c7b0b9f
# ╟─bd481c82-99c7-44aa-b9c7-11dacb231070
# ╟─5df4fbbb-f483-428b-8e2c-f09a7b69a0ca
# ╟─e21377fc-d136-4cac-9914-299acac72109
# ╟─db29e242-5029-48be-a76c-b0868060cc1c
# ╟─733276b9-7f00-432c-bf7e-fb9e8058892d
# ╟─8a605295-3865-4fca-bef0-ca9172d3882e
# ╠═3658e062-b85e-4692-83c4-5846da10b624
# ╟─f303dbb1-1b59-4712-b197-eb2f6c7c9ad0
# ╠═d85a0ed5-341c-43d5-bef1-092d3daadd2e
# ╟─f7ef47b3-1c52-4e8f-a4b4-7ef2a4bf83f1
# ╠═ba80f8f7-afb7-4ecb-94d0-904c2a777512
# ╟─87aa3c1c-4ce4-4b6f-ab6a-7b889c526e96
# ╠═c410e103-76ee-4b15-8172-a091c509fe42
# ╟─fa120fc0-9193-4c88-a80f-e4fea8a5827a
# ╟─dfc169c5-c89d-487f-af59-3e2b2c9a7277
# ╟─72319312-100b-4053-aafa-4ba20acf4998
# ╟─c4854f0d-57b4-4d30-884f-e6273c416d9c
# ╟─b2d3f1ba-d1f6-4de8-bbb0-9e9f6d65e7e2
# ╠═a300228c-df39-4f13-b47c-28a797dcb43a
# ╠═64e76a4f-b9ac-4899-b1d9-9e14ce9ae540
# ╠═4062274b-f208-460e-9efa-15ef770ee13b
# ╟─0ce787c6-3e8e-428d-9964-9d9ebb7262fd
# ╠═34a2d209-5280-45e0-ba43-b5d6eb01a48b
# ╟─f064467b-b9ff-4938-84f0-8df7ef7d42dd
# ╟─46d392ce-dea6-4a99-9a02-6c7e97548b9f
