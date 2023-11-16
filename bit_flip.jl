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

# Global variables
dephasing_gamma = 0.01;

function dissipator(v)
	operator = F4[2];
	dephasing_dissipator = -2 * dephasing_gamma * [0, v[2], v[3]];
	return dephasing_dissipator;
end

function target(v)
	return v[1]^2 + v[2]^2;
end

function grad(v)
	return 2 * [v[1], v[2], 0];
end

function get_hamiltonian(v, dissipation)
	f_preserving_gamma = 0.01;
	f_preserving_hamiltonian = dephasing_gamma * v[2]^2 / (v[3] * target(v)) * [-v[2], v[1], 0];
	state_preserving_hamiltonian = begin
	    # action = dissipation + f_preserving_hamiltonian;
	    # cross = 2*v[3]*[v[2],-v[1],0];
	    # coeff = 10 / (cross[2]-cross[1])
	    # 0.001 * tanh(10*v[2]) * grad(v)
	    # dephasing_gamma * (v[2]-(v[2]^3+v[1]*v[2]^2)/(v[1]^2+v[2]^2))/((v[2]-v[1])*v[3]) * grad(v);
	    [0,0,0]
	    # dephasing_gamma/(4*v[3]^2) * (pi * (v[2]^2+v[3]^2) - 2* v[1]*v[2]*v[3]/(v[1]^2+v[2]^2)) * [2 * v[1], 2*v[2], 0]
	end
	return f_preserving_hamiltonian + state_preserving_hamiltonian
end

function get_hamiltonian(v)
	return get_hamiltonian(v, dissipator(v))
end

function lindblad(v, p, t)
	dissipation = dissipator(v);
	hamiltonian = get_hamiltonian(v, dissipation);
	if any(isnan, hamiltonian)
		return [Inf, Inf, Inf]
	end
	return 2 * cross(hamiltonian, v) + dissipation
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


# 1/0.04 * ln((0.5*exp(1/2)-0.25)/0.25)
v = [0.5,0.5,0.5]
# rho = 0.995 * rho + 0.005 * F16[1]/4
tend = 500.0;
problem = ODEProblem(lindblad, v, (0.0, tend));
sol = solve(problem, alg=Rosenbrock23(), tstops = 0.0:10:tend, isoutofdomain = outdomain); #alg_hints=[:stiff], 

if sol.t[end] < tend
	print("Ended early at ", sol.t[end], "\n")
end
times = [t for (u, t) in tuples(sol)];
targets = [target(u) for u in sol.u];
ham = [norm(get_hamiltonian(u)) for u in sol.u];
perp = [norm(u)^2 * norm(grad(u))^2 - (grad(u)' * u)^2 for u in sol.u];
normie = [norm(u)^2 for u in sol.u];
preserving_rates = [[u[2]*u[3] for u in sol.u] [u[1]*u[3] for u in sol.u]];
lindblad_rates = [lindblad(u,0,0) for u in sol.u]

access(i,arr) = [x[i] for x in arr];
plotly();
# plot(times, targets, show=true, label="target");
plot(times, ham, show=true, label="Hamiltonian norm");
plot(times, normie, show=true, label="Purity");
# plot(times, preserving_rates, show=true, label=["dx" "dy"]);
# plot(times[1:30], [access(1,lindblad_rates), access(2,lindblad_rates), access(3,lindblad_rates)], show=true, label=["del x" "del y" "del z"]);
scatter(access(1,sol.u),access(2,sol.u),access(3,sol.u), show=true);
r = 0.5*sqrt(2)
h = 1
m, n =20, 20
u = range(0, 2pi, length=n)
v = range(0, h, length=m)

us = ones(m)*u'
vs = v*ones(n)'
#Surface parameterization
X = r*cos.(us)
Y = r*sin.(us)
Z = vs
surface!(X, Y, Z, size=(600,600), cbar=:none, legend=false, alpha=0.5)

# Just hamiltonian - 253.12564570409663










