using DifferentialEquations;
using LinearAlgebra;
using NaNMath;
using DiffEqCallbacks;

# The setup for the experiment is:
# 1. Z-axis dephasing
# 2. Pure relaxation, i.e. relaxation to ground state
# The initial state is the bloch vector (1/sqrt(2),0,1/sqrt(2))
# The output is a Hamiltonian, with units in angular frequency. Rabi frequency
# is linear frequency, so divide by 2pi to get in terms of Rabi frequency.

# Experiment setup
h = 6.62607015 * 1e-34; # Planck's constant - Joule per Hertz
kb = 1.380649 * 1e-23; # Boltzmann's constant - Joule per Kelvin

# 1/t_2 = 1/(2t_1) + 1/t_phi.
# t_phi - corresponds to dephasing. Equal to 1/gamma
# t_1 - corresponds to thermal relaxation.
t_1 = 50.5 * 1e-6;
t_2 = 45.8 * 1e-6;
dephasing_gamma = ((1/t_2)-(1/(2*t_1)));
thermal_gamma = 1 / t_1;

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

function get_relaxation_coeff(temp, qbit_freq)
	# Returns the rate associated with relaxation.
	#
	# Assume qubit energy is -(1/2)ωσ. Then ground state population in Gibbs state is
	# 1/(1+exp(-βH)). The probability p of the excitation channel is simply the population
	# of excited state at equilibrium.
	βH = (h * qbit_freq) / (kb * temp);
	return 1/(1+exp(-βH))
end

function dissipator(v)
	dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
	relaxation_coefficient = 1 # get_relaxation_coeff(bath_temp, qbit_freq);
	thermal_dissipator = thermal_gamma * (-relaxation_coefficient * [v[1]/2, v[2]/2, v[3]-1] - (1-relaxation_coefficient) * [v[1]/2, v[2]/2, v[3]+1]);
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
	return dephasing_hamiltonian + thermal_hamiltonian + [0,0,100000/2 * 2 * pi]
end

function lindblad(v, p, t)
	hamiltonian = get_hamiltonian(v);
	if any(isnan, hamiltonian)
		return [Inf, Inf, Inf]
	end
	return 2 * cross(hamiltonian, v) + dissipator(v)
end

function get_hamiltonian_matrix(u, t, int)
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
v = [0.41,0,-0.4]
tend = v[3]^2/(2*((2*dephasing_gamma + thermal_gamma*0.5)*(v[1]^2+v[2]^2)-v[3]*(1-v[3])*thermal_gamma));
problem = ODEProblem(lindblad, v, (0.0, tend));

# saved_hamiltonian = SavedValues(eltype(1/sampling_rate), Matrix{ComplexF64});
# saved_target = SavedValues(eltype(1/sampling_rate), Float64);
# callback = CallbackSet(
# 	SavingCallback(get_hamiltonian_matrix, saved_hamiltonian, saveat=0:1/sampling_rate:tend),
# 	SavingCallback(save_target, saved_target, saveat=0:1/sampling_rate:tend),
# )

@time sol = solve(problem, alg_hints=[:stiff], saveat=1/sampling_rate);

print("Conservative bound: ", v[3]^2/(2*((2*dephasing_gamma + thermal_gamma*0.5)*(v[1]^2+v[2]^2)-v[3]*(1-v[3])*thermal_gamma)))
print("\n")
print(sol.t[end])

# z = sol.u[end][3]
# fz = (v[1]^2+v[2]^2) / z
# print("\nZ: ",z, "\ntarget value: ", v[1]^2+v[2]^2, "\nratio: ", fz, "\nEstimated time: ", z^2/(2*((2*dephasing_gamma + thermal_gamma*0.5)*(v[1]^2+v[2]^2)-z*(1-z)*thermal_gamma)));
using Plots;
plotly();
plot(sol, show=true, label=["vx" "vy" "vz"]);
plot(sol.t, [target(u) for u in sol.u], show=true, ylim=(0,1), label="target")
plot(sol.t, [[get_hamiltonian(u)[1] for u in sol.u], [get_hamiltonian(u)[2] for u in sol.u], [get_hamiltonian(u)[3] for u in sol.u]], show=true, label=["hx", "hy", "hz"])

# using Polynomials, IterTools

# t = sol.t;
# hy = [get_hamiltonian(u)[2] for u in sol.u];
# crude_fit = truncate(fit(t, hy, 20), atol=0.01);
# plot(t, (@.f(t)), label="Poly fit", show=true);
# plot!(t, hy, label="Actual");
# @show maximum([abs(hy[x])-crude_fit(t[x]) for x in range(1,length(t))])

# partitions = 5;
# total_length = length(t);
# partition_length = floor(Int, total_length/partitions);

# crude_fit_parts = [];
# t_parts = [];
# for i in range(1, partitions)
# 	start, nd = partition_length*(i-1)+1, partition_length*i
# 	push!(t_parts, t[start:nd])
# 	push!(crude_fit_parts, truncate(fit(t[start:nd], hy[start:nd], 20), atol=0.01))
# end

# plot(t_parts[1], (@.crude_fit_parts[1](t_parts[1])), color="red", linewidth=2, thickness_scaling=1, label="Fit 1")
# for i in range(2, partitions)
# 	plot!(t_parts[i], (@.crude_fit_parts[i](t_parts[i])), color="red", linewidth=2, thickness_scaling=1, label="Fit $i");
# end
# plot!(t, hy, label="Actual", show=true);
# @show maximum([abs(hy[x])-crude_fit_parts[ceil(Int, x/partition_length)](t[x] % partition_length) for x in range(1,length(t)-1)])

# using DataFrames, CSV;
# hx_coeffs = [0 for x in range(1, partitions*11)];
# hz_coeffs = [0 for x in range(1, partitions*11)];
# hy_coeffs = [];
# for i in range(1, partitions)
# 	append!(hy_coeffs, [getindex(crude_fit_parts[i], j) for j in range(0,10)])
# end
# df = DataFrame(permutedims(append!([t_1*1e6,t_2*1e6], hx_coeffs, hy_coeffs, hz_coeffs)), ["t1","t2",

# "hx_1_0","hx_1_1","hx_1_2","hx_1_3","hx_1_4","hx_1_5","hx_1_6","hx_1_7","hx_1_8","hx_1_9","hx_1_10",
# "hx_2_0","hx_2_1","hx_2_2","hx_2_3","hx_2_4","hx_2_5","hx_2_6","hx_2_7","hx_2_8","hx_2_9","hx_2_10",
# "hx_3_0","hx_3_1","hx_3_2","hx_3_3","hx_3_4","hx_3_5","hx_3_6","hx_3_7","hx_3_8","hx_3_9","hx_3_10",
# "hx_4_0","hx_4_1","hx_4_2","hx_4_3","hx_4_4","hx_4_5","hx_4_6","hx_4_7","hx_4_8","hx_4_9","hx_4_10",
# "hx_5_0","hx_5_1","hx_5_2","hx_5_3","hx_5_4","hx_5_5","hx_5_6","hx_5_7","hx_5_8","hx_5_9","hx_5_10",

# "hy_1_0","hy_1_1","hy_1_2","hy_1_3","hy_1_4","hy_1_5","hy_1_6","hy_1_7","hy_1_8","hy_1_9","hy_1_10",
# "hy_2_0","hy_2_1","hy_2_2","hy_2_3","hy_2_4","hy_2_5","hy_2_6","hy_2_7","hy_2_8","hy_2_9","hy_2_10",
# "hy_3_0","hy_3_1","hy_3_2","hy_3_3","hy_3_4","hy_3_5","hy_3_6","hy_3_7","hy_3_8","hy_3_9","hy_3_10",
# "hy_4_0","hy_4_1","hy_4_2","hy_4_3","hy_4_4","hy_4_5","hy_4_6","hy_4_7","hy_4_8","hy_4_9","hy_4_10",
# "hy_5_0","hy_5_1","hy_5_2","hy_5_3","hy_5_4","hy_5_5","hy_5_6","hy_5_7","hy_5_8","hy_5_9","hy_5_10",

# "hz_1_0","hz_1_1","hz_1_2","hz_1_3","hz_1_4","hz_1_5","hz_1_6","hz_1_7","hz_1_8","hz_1_9","hz_1_10",
# "hz_2_0","hz_2_1","hz_2_2","hz_2_3","hz_2_4","hz_2_5","hz_2_6","hz_2_7","hz_2_8","hz_2_9","hz_2_10",
# "hz_3_0","hz_3_1","hz_3_2","hz_3_3","hz_3_4","hz_3_5","hz_3_6","hz_3_7","hz_3_8","hz_3_9","hz_3_10",
# "hz_4_0","hz_4_1","hz_4_2","hz_4_3","hz_4_4","hz_4_5","hz_4_6","hz_4_7","hz_4_8","hz_4_9","hz_4_10",
# "hz_5_0","hz_5_1","hz_5_2","hz_5_3","hz_5_4","hz_5_5","hz_5_6","hz_5_7","hz_5_8","hz_5_9","hz_5_10",
# ])

# CSV.write("poly.csv", df)










