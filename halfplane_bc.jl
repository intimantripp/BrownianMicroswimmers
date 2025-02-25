using Pkg
Pkg.activate(".")
using Random
using LinearAlgebra
using StatsPlots
using LaTeXStrings
using NonlinearSolve

# Define the parameters
# const seed1 = 12887
# Random.seed!(seed1)
const v0 = 1.0      # Velocity magnitude
const D_T = 1e-3     # Diffusivity
const D_R = 0.1    # Diffusivity
const dt = 0.01     # Time step
const brownian_step = sqrt(2 * D_T * dt)
const brownian_r_step = sqrt(2 * D_R * dt)
const T = 200.0     # Total simulation time
const steps = Int(T/dt)
const ell = 0.5
const channell = 1.0

# Initial conditions
X0 = [1.0, 0.5]
θ0 = 0.0

const fbottom(θ,l,cl) = 0.5*l*abs(sin(θ))
const fbottomder(θ,l) = 0.5*l*sin(θ)*cos(θ)/abs(sin(θ))
const ftop(θ,l,cl) = cl-0.5*l*abs(sin(θ))
const ftopder(θ,l) = -0.5*l*sin(θ)*cos(θ)/abs(sin(θ))


function find_intersection(p1::Vector{Float64},p2::Vector{Float64},l::Float64,cl::Float64,f::Function)
    error_function(t,p) = f(p1[1] + t * (p2[1] - p1[1]),l,cl) - p1[2] - t * (p2[2] - p1[2])

    prob = IntervalNonlinearProblem(error_function,(0.0,1.0))
    sol = solve(prob)

    if SciMLBase.successful_retcode(sol.retcode)
        t_sol = sol.u
        return [p1[1] + t_sol * (p2[1] - p1[1]),p1[2] + t_sol * (p2[2] - p1[2])]
    else
        return nothing
    end
end

function give_ref(intp::Vector{Float64},p2::Vector{Float64},l::Float64,fder::Function)::Vector{Float64}
    n = [1,fder(intp[1],l)]
    n = n/norm(n)

    u = intp + dot(p2-intp,n)*n

    return 2*u-p2
end

function apply_boundary(X, dX, θ, dθ, l, cl)
    Xn = X+dX
    θn = θ+dθ
    P1 = [θ,X[2]]
    P2 = [θ+dθ,Xn[2]]
    if P2[2] < 0.5*l
        while find_intersection(P1,P2,l,cl,fbottom) !== nothing
            P2 = give_ref(find_intersection(P1,P2,l,cl,fbottom),P2,l,fbottomder)
            yn = P2
            Xn = [Xn[1],yn[2]]
            θn = yn[1]
        end
    elseif P2[2] > cl - 0.5*l
        while find_intersection(P1,P2,l,cl,ftop) !== nothing
            P2 = give_ref(find_intersection(P1,P2,l,cl,ftop),P2,l,ftopder)
            yn = P2
            Xn = [Xn[1],yn[2]]
            θn = yn[1]
        end
    end
    return Xn, θn
end

function simulate(X0,θ0)
    X, θ = X0, θ0
    trajectory = zeros((steps,3))

    for it in 1:steps

        # Update X based on θ
        dX = v0 * [cos(θ), sin(θ)] * dt + brownian_step * randn(2)

        dθ = brownian_r_step * randn()

        # Handle boundary conditions and transitions
        X, θ = apply_boundary(X, dX, θ, dθ, ell, channell)

        # Store the trajectory
        trajectory[it,1] = X[1]
        trajectory[it,2] = X[2]
        trajectory[it,3] = θ
    end
    return trajectory
end

# traj = simulate()

msize1 = 4.0
skip = 5
lxmod = 2
trailsize = 40
ytrajs = []
θtrajs = []
xtrajs = []

Random.seed!(1000)
seedss = [rand(1000:9999) for j ∈ 1:30]
        
for iseed in 1:size(seedss)[1]
    Random.seed!(seedss[iseed])
    X0 = [rand(),0.25 + rand()*0.5]
    θ0 = rand()*2*pi
    traj = simulate(X0,θ0)
    println(iseed)
    push!(ytrajs,traj[1:skip:end,2])
    push!(θtrajs,mod.(traj[1:skip:end,3],2*pi))
    push!(xtrajs,mod.(traj[1:skip:end,1],lxmod))
end

Lmax = lxmod
Lmin = 0.0
Hmax = channell

Xsprojt = zeros((size(xtrajs[1])[1],2))
θsprojt = zeros(size(θtrajs[1])[1])

iseed1 = 1
for j ∈ 1:size(xtrajs[1])[1]
    Xsprojt[j,1] = xtrajs[iseed1][j]
    Xsprojt[j,2] = ytrajs[iseed1][j]
    θsprojt[j] = θtrajs[iseed1][j]
end

anim = @animate for t ∈ trailsize+1:size(Xsprojt)[1]
    plot([Lmin,Lmax],[0.0,0.0],c=:black,label="",
    xlims=(Lmin,Lmax),xticks=Lmin:0.5:Lmax,ylims=(-0.5,channell+0.5),yticks=-0.5:0.5:channell+0.5,aspect_ratio=1)
    plot!([Lmin,Lmax],[channell,channell],c=:black,label="")
    scatter!([Xsprojt[t,1]],[Xsprojt[t,2]],
    m=:circle,markersize=msize1,msw=0.0,color=:tomato,lab="")
    for it ∈ 1:trailsize
        scatter!([Xsprojt[t-it,1]],[Xsprojt[t-it,2]],alpha=1-0.05*it,
        m=:circle,markersize=msize1,msw=0.0,color=:tomato,lab="")
    end
    plot!([Xsprojt[t,1]-0.5*ell*cos(θsprojt[t]),Xsprojt[t,1]+0.5*ell*cos(θsprojt[t])],
    [Xsprojt[t,2]-0.5*ell*sin(θsprojt[t]),Xsprojt[t,2]+0.5*ell*sin(θsprojt[t])], c=:black,label="")
end
mp4(anim, "images/new/half_p_l=$(ell)_cl=$(channell)_anim_v0=$(v0)_DR=$(D_R)_DT=$(D_T)_seed=$(seedss[iseed1])_T=$(T).mp4", fps = 30)

# findmin(θsprojt)
# findmax(Xsprojt[:,2])

ytrajs0 = zeros(size(ytrajs)[1]*size(ytrajs[1])[1])
θtrajs0 = zeros(size(θtrajs)[1]*size(θtrajs[1])[1])
xtrajs0 = zeros(size(xtrajs)[1]*size(xtrajs[1])[1])

for i ∈ 1:size(ytrajs)[1]
    for j ∈ 1:size(ytrajs[1])[1]
        ytrajs0[(i-1)*size(ytrajs[1])[1] + j] = ytrajs[i][j]
        θtrajs0[(i-1)*size(θtrajs[1])[1] + j] = θtrajs[i][j]
        xtrajs0[(i-1)*size(xtrajs[1])[1] + j] = xtrajs[i][j]
    end
end

histogram2d(θtrajs0,ytrajs0,bins=(60,50),xticks=(0:pi/2:2*pi,[L"%$(i)\pi" for i in 0:1/2:2]),xlabel=L"\theta",ylabel=L"y")
savefig("images/histo2d/histoyth_D_T=$(D_T)_D_R=$(D_R)_v0=$(v0)_delta_t=$(dt)_T=$(T)_seeds=$(size(seedss)[1]).png")

histogram2d(xtrajs0,ytrajs0,bins=(40,20),xlabel=L"x",ylabel=L"y")
savefig("images/histo2d/histoxy_D_T=$(D_T)_D_R=$(D_R)_v0=$(v0)_delta_t=$(dt)_T=$(T)_seeds=$(size(seedss)[1]).png")
