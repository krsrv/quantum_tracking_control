using DifferentialEquations;
using LinearAlgebra;
using NaNMath;
using DiffEqCallbacks;

# Required setup for experiment is:
# * Dephasing along Y-axis
# * initial vector on YZ plane
# * coherence on XZ plane
# This corresponds to a Z-axis dephasing, initial vector on XZ plane, and coherence on XY plane
# and a permutation of (X->Z, Y->X, Z->Y) to get the initial setup.

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin

# t_1, t_2 times are 100 microsec each
# t_2 is 1.4 microsec
# 1/t_2 = 1/(2t_1) + 1/t_phi.
# t_phi - dephasing
# t_phi is 1/2gamma
dephasing_gamma = 1 / (1.4 * 1e-6);
thermal_gamma = 1 * 1e4;

# Sampling rate is 2.4 giga samples per sec
sampling_rate = 2.4 * 1e9;
qbit_freq = 3877496000;
bath_temp = 50 * 10e-3;

# Simulation code
F4 = Matrix{ComplexF64}[];
push!(F4, [1 0;0 1]);
push!(F4, [0 1;1 0]);
push!(F4, [0 -1im;1im 0]);
push!(F4, [1 0;0 -1]);

matrix_to_coeff(matrix) = [real(tr(x' * matrix)) for x in F4];
coeff_to_matrix(coeff) = 0.5 * reduce(+, [x*y for (x,y) in zip(cat([1],coeff,dims=1), F4)])

commutator(x, y) = x*y - y*x;

function get_thermal_coeff(temp, qbit_freq)
	βH = (h * qbit_freq) / (kb * temp);
	return 1/(1+exp(-βH))
end

function dissipator(v)
	dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
	thermal_coefficient = get_thermal_coeff(bath_temp, qbit_freq);
	thermal_dissipator = [0,0,0]#thermal_gamma * (-thermal_coefficient * [v[1]/2, v[2]/2, v[3]-1] - (1-thermal_coefficient) * [v[1]/2, v[2]/2, v[3]+1]);
	return dephasing_dissipator + thermal_dissipator;
end

function target(v)
	return v[1]^2 + v[2]^2;
end

function grad(v)
	return 2 * [v[1], v[2], 0];
end

function get_hamiltonian(v)
	dephasing_hamiltonian = -dephasing_gamma / v[3] * [v[2], -v[1], 0];
	thermal_hamiltonian = [0,0,0]#-thermal_gamma / (4 * v[3]) * [v[2], -v[1], 0];
	return dephasing_hamiltonian + thermal_hamiltonian
end

function lindblad(v, p, t)
	hamiltonian = get_hamiltonian(v);
	if any(isnan, hamiltonian)
		return [Inf, Inf, Inf]
	end
	return 2 * cross(hamiltonian, v) + dissipator(v)
end

function save_hamiltonian(u, t, int)
	h_vec = get_hamiltonian(u)
	return h_vec[1] * F4[2] + h_vec[2] * F4[3] + h_vec[3] * F4[4]
end

function save_target(u, t, int)
	return target(u)
end

function outdomain(r, p, t)
	return any(isnan, get_hamiltonian(r))
end


v = [1/sqrt(2),0,1/sqrt(2)]
tend = 1e-3;
problem = ODEProblem(lindblad, v, (0.0, tend));

saved_hamiltonian = SavedValues(eltype(1/sampling_rate), Matrix{ComplexF64});
saved_target = SavedValues(eltype(1/sampling_rate), Float64);
callback = CallbackSet(
	SavingCallback(save_hamiltonian, saved_hamiltonian, saveat=0:1/sampling_rate:tend),
	SavingCallback(save_target, saved_target, saveat=0:1/sampling_rate:tend),
)

@time sol = solve(problem, alg_hints=[:stiff], saveat=1/sampling_rate, isoutofdomain=outdomain);

using DataFrames, CSV;
df = DataFrame(mapreduce(permutedims, vcat, [append!([t],get_hamiltonian(u)) for (u,t) in tuples(sol)]),
	["t","hz","hx","hy"])
CSV.write("h.csv",df)
# using JLD2;
# JLD2.save("hamiltonian.jld2", "hamiltonian", [save_hamiltonian(x,0,0) for x in sol.u]);
# JLD2.save("target.jld2", "target", [target(x) for x in sol.u]);

# using Plots;
# plotly();
# plot(sol, show=true, label=["vx" "vy" "vz"]);
# plot(sol.t, [target(u) for u in sol.u], show=true, ylim=(0,1), label="target")
# plot(sol.t, [norm(save_hamiltonian(u,0,0)) for u in sol.u], show=true, label="hamiltonian norm")


