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

function dissipator(v)
	return [0,0,0];
	# operator = F4[3];
	# dephasing_gamma = .01 - 0.001
	# dephasing_dissipator = -2 * dephasing_gamma * [v[1], v[2], 0];
	# relaxation_gamma = .03 - 0.003
	# relaxation_dissipator = -relaxation_gamma * [v[1]/2, v[2]/2, v[3]-1];
	# return dephasing_dissipator + relaxation_dissipator;
end

function target(v)
	return v[1]^2 + v[2]^2;
end

function grad(v)
	return 2 * [v[1], v[2], 0];
end

function get_hamiltonian(v)
	return [v[2]*v[3],v[3]*v[1],v[1]*v[2]];
	# dephasing_gamma = 0.01;
	# dephasing_hamiltonian = -dephasing_gamma / v[3] * [v[2], -v[1], 0];
	# relaxation_gamma = .03
	# relaxation_hamiltonian = -relaxation_gamma / (4 * v[3]) * [v[2], -v[1], 0];
	# return dephasing_hamiltonian + relaxation_hamiltonian
end

function lindblad(v, p, t)
	hamiltonian = get_hamiltonian(v);
	if any(isnan, hamiltonian)
		return [Inf, Inf, Inf]
	end
	return 2 * cross(hamiltonian, v) + dissipator(v)
end

function check_positivity(m::AbstractMatrix)
    minimum(m) >= 0
end

function PositivityCheck!(ρ, t, integrator)
    if !check_positivity(integrator.u) || !check_positivity(ρ)
        @warn "The density matrix becomes negative at time $t."
        terminate!(integrator)
    end
    u_modified!(integrator, false)
end
callback = FunctionCallingCallback(PositivityCheck!, func_everystep=true, func_start=false);

function outdomain(r, p, t)
	return any(isnan, get_hamiltonian(r))
end


v = [0.2,0.2,0.9]
# rho = 0.995 * rho + 0.005 * F16[1]/4
tend = 1000.0;
problem = ODEProblem(lindblad, v, (0.0, tend));
sol = solve(problem, alg_hints=[:stiff], tstops = 0.0:0.1:tend, isoutofdomain = outdomain);

if sol.t[end] < tend
	print("Ended early at ", sol.t[end], "\n")
end
times = [t for (u, t) in tuples(sol)];
targets = [target(u) for (u, t) in tuples(sol)];
model_target = [target(sol.u[1]) * exp( 4 * 0.001 * t + 0.003 * t) for (u, t) in tuples(sol)];
ham = [norm(get_hamiltonian(u)) for (u, t) in tuples(sol)];
perp = [norm(u)^2 * norm(grad(u))^2 - (grad(u)' * u)^2 for (u,t) in tuples(sol)];
normie = [norm(u)^2 for (u,t) in tuples(sol)];

plotly();
access(i) = [x[i] for x in sol.u];
plot(times, normie, show=true, label="norm");
# plot(times, ham, show=true, label="Hamiltonian norm");
# plot(times, perp, show=true, label="Perpendicular component norm");
# plot(times, normie, show=true, label="Norm");
path3d(access(1),access(2),access(3), show=true)
x = y = range(0.1,stop=1,length=20)
surface!(x,y,(x,y)->0.2*0.2*0.9/x/y,alpha=0.4)
u = range(0,stop=2*π,length=20);
w = range(0,stop=π,length=20);
surface!(norm(v)*cos.(u) * sin.(w)', norm(v)*sin.(u) * sin.(w)', norm(v)*ones(20)*cos.(w)',alpha=0.4)
# Just hamiltonian - 253.12564570409663










