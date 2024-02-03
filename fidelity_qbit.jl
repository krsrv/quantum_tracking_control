using DifferentialEquations;
using LinearAlgebra;
using Plots;
using NaNMath;
using DiffEqCallbacks;

F4 = Matrix{ComplexF64}[];
push!(F4, [1 0;0 1]);
push!(F4, [0 1;1 0]);
push!(F4, [0 -1im;1im 0]);
push!(F4, [1 0;0 -1]);

matrix_to_coeff(matrix) = [real(tr(x' * matrix)) for x in F4];
coeff_to_matrix(coeff) = 0.5 * reduce(+, [x*y for (x,y) in zip(cat([1],coeff,dims=1), F4)])

commutator(x, y) = x*y - y*x;
w = [1/sqrt(2), 1/sqrt(3), 1/sqrt(6)];

function dissipator(v)
	dephasing_gamma = .01
	dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
	return dephasing_dissipator;
end

function target(v)
	return 0.5 * (1+dot(w,v) + sqrt((1-norm(v)^2)*(1-norm(w)^2)));
end

function get_hamiltonian(v)
	dephasing_gamma = 0.01;
	k = sqrt((1-norm(w)^2)/(1-norm(v)^2));
	dephasing_hamiltonian = -dephasing_gamma * dot(w-k*v, [v[1],v[2],0]) * cross(w,v)/norm(cross(w,v))^2;
	return dephasing_hamiltonian
end

function lindblad(v, p, t)
	hamiltonian = get_hamiltonian(v);
	if any(isnan, hamiltonian)
		return [Inf, Inf, Inf]
	end
	return 2 * cross(hamiltonian, v) + dissipator(v)
end

v = [1/sqrt(6),1/sqrt(3),1/sqrt(3)];
# rho = 0.995 * rho + 0.005 * F16[1]/4
tend = 1000.0;
problem = ODEProblem(lindblad, v, (0.0, tend));
sol = solve(problem, alg_hints=[:stiff], tstops = 0.0:0.1:tend,);

if sol.t[end] < tend
	print("Ended early at ", sol.t[end], "\n")
end
times = [t for (u, t) in tuples(sol)];
targets = [target(u) for (u, t) in tuples(sol)];
ham = [norm(get_hamiltonian(u)) for (u, t) in tuples(sol)];
# perp = [norm(u)^2 * norm(grad(u))^2 - (grad(u)' * u)^2 for (u,t) in tuples(sol)];
# normie = [norm(u)^2 for (u,t) in tuples(sol)];

plotly();
plot(times, [targets], show=true, label="Target");
plot(times, ham, show=true, label="Hamiltonian norm");
# plot(times, perp, show=true, label="Perpendicular component norm");
# plot(times, normie, show=true, label="Purity", ylim=(0,1), ylabel="Purity");
# plot(sol, show=true, label=["vx" "vz" "vy"], ylabel="Bloch vector components");


# Just hamiltonian - 253.12564570409663










