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

F16 = [kron(x,y) for x in F4 for y in F4];
normalised_F16 = 1/2 * F16;

matrix_to_coeff(matrix) = [real(tr(x' * matrix)) for x in normalised_F16];
function coeff_to_matrix(coeff)
	if length(coeff) == 15 # normalised Hermitian matrix
		return reduce(+, [x*y for (x,y) in zip(cat([0.5],coeff,dims=1), normalised_F16)])
	elseif length(coeff) == 16
		return reduce(+, [x*y for (x,y) in zip(coeff, normalised_F16)])
	elseif length(coeff) == 3 # bloch vector
		return 0.5 * reduce(+, [x*y for (x,y) in zip(cat([1],coeff,dims=1), F4)])
	else
		throw(DomainError("Coeff length should be 3, 4, 15 or 16"))
	end
end

hsinner(x, y) = tr(x' * y);
commutator(x, y) = x*y - y*x;

function dissipator(rho)
	operator = F16[4] + F16[13];
	gamma = .01
	return gamma * Hermitian(operator * rho * operator' - 0.5 * (operator' * operator * rho + rho * operator' * operator));
end

function target(rho)
	coeff = matrix_to_coeff(rho);
	bloch_norm = norm(coeff[2:4]) * 2;
	entropy_emt = -(1+bloch_norm)/2 * log((1+bloch_norm)/2) - (1-bloch_norm)/2 * log((1-bloch_norm)/2);
	entropy = reduce(+, [real(lambda) == 0.0 ? 0 : -real(lambda) * NaNMath.log(real(lambda)) for lambda in eigvals(rho)]);
	return entropy - entropy_emt;
end

function grad(rho)
	coeff = matrix_to_coeff(rho);
	vec_norm = norm(coeff[2:4]);
	# @show rho, eigvals(rho);
	grad1 = 1/vec_norm * NaNMath.log((1-2*vec_norm)/(1+2*vec_norm)) * reduce(+, [x*y for (x,y) in zip(coeff[2:4], normalised_F16[2:4])]);
	grad2 = 0 * F16[1];
	r = Hermitian(rho);
	evals = eigvals(r);
	evecs = eigvecs(r);
	for i in range(1,4)
		val = evals[i];
		vc = evecs[:,i];
		grad2 += -(1 + NaNMath.log(val)) / (vc' * vc) * reduce(+, [(vc' * basis * vc) * basis for basis in normalised_F16[3:16]])
	end
	return grad2 - grad1;
end

function get_hamiltonian(rho)
	dissipation = dissipator(rho);
	gradient = grad(rho);
	return get_hamiltonian(rho, dissipation, gradient);
end

function get_hamiltonian(rho, dissipation, gradient)
	alpha = -1.0im * hsinner(gradient, dissipation)/hsinner(commutator(rho, gradient), commutator(rho, gradient));
	return Matrix(Hermitian(alpha * commutator(rho, gradient)));
end

function lindblad(rho, p, t)
	if any(isnan, rho)
		return NaN * F16[1];
	end
	dissipation = dissipator(rho);
	gradient = grad(rho);
	hamiltonian = get_hamiltonian(rho, dissipation, gradient);
	return (-1.0im * commutator(hamiltonian, rho) + dissipation);
end

function check_positivity(m::AbstractMatrix)
    if !ishermitian(m)
        @warn "Input fails the numerical test for Hermitian matrix. Use the upper triangle to construct a new Hermitian matrix."
        d = Hermitian(m)
    else
        d = m
    end
    eigmin(d) >= 0
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
	return any(isnan, r) || eigmin(Hermitian(r))<0
end

### Tests begin
r = 0.5 * [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]; # Maximally entangled state
@assert target(r) == -log(2);
@assert matrix_to_coeff(r) == 0.5 * [1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1];

r = coeff_to_matrix(zeros(15)); # Maximally mixed state
@assert target(r) == log(2);
### Tests end

# x = rand(4,4) + 1.0im * rand(4,4);
# rho = x' * x;
# rho = rho / tr(rho);
# Initial rho
rho = kron([1 0;0 0], [1 0;0 0]);
# rho = 0.995 * rho + 0.005 * F16[1]/4
print(rho, "\n");
print(eigvals(rho), "\n");

tend = 10.0;
problem = ODEProblem(lindblad, Matrix(Hermitian(rho)), (0.0, tend));
sol = solve(problem, tstops = 0.0:0.1:tend, isoutofdomain = outdomain);

if sol.t[end] < tend
	print("Ended early at ", sol.t[end], "\n")
end
times = [t for (u, t) in tuples(sol)];
targets = [target(u) for (u, t) in tuples(sol)];

plot(times, targets, show=true);













