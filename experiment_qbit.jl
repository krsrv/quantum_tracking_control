using DifferentialEquations;
using LinearAlgebra;
using NaNMath;
using DiffEqCallbacks;

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin

# t_1, t_2 times are 100 microsec each
# 1/t_2 = 1/(2t_1) + 1/t_phi
dephasing_gamma = 0.5 * 1e4;
thermal_gamma = 1e4;

# Sampling rate is 2.4 giga samples per sec
sampling_rate = 2.4 * 1e9;

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
	βH = (h * qbit_freq) / (2 * pi * kb * temp);
	return 1/(1+exp(-βH))
end

function dissipator(v)
	dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
	thermal_coefficient = get_thermal_coeff(50 * 10e-3, 5 * 1e9);
	thermal_dissipator = thermal_gamma * (-thermal_coefficient * [v[1]/2, v[2]/2, v[3]-1] - (1-thermal_coefficient) * [v[1]/2, v[2]/2, v[3]+1]);
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
	thermal_hamiltonian = -thermal_gamma / (4 * v[3]) * [v[2], -v[1], 0];
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


v = [0.2,0.2,0.9]
# rho = 0.995 * rho + 0.005 * F16[1]/4
tend = 1.0;
problem = ODEProblem(lindblad, v, (0.0, tend));

saved_hamiltonian = SavedValues(eltype(1/sampling_rate), Matrix{ComplexF64});
saved_target = SavedValues(eltype(1/sampling_rate), Float64);
callback = CallbackSet(
	SavingCallback(save_hamiltonian, saved_hamiltonian, saveat=0:1/sampling_rate:tend),
	SavingCallback(save_target, saved_target, saveat=0:1/sampling_rate:tend),
)

@time sol = solve(problem, alg_hints=[:stiff], callback=callback, isoutofdomain = outdomain);

using JLD;
save("hamiltonian.jld", "h", saved_hamiltonian.saveval)
save("target.jld", "t", saved_target.saveval)

# target_vals = load("target.jld", "t")
#16094243



