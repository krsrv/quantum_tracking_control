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
	function basic_dissipator(operator, gamma)
		return gamma * Hermitian(operator * rho * operator' - 0.5 * (operator' * operator * rho + rho * operator' * operator));
	end
	return basic_dissipator(F16[4], 0.01) + basic_dissipator(+F16[13], 0.03)
	# I x Z + Z x I
	# operator = F16[4] + F16[13];
	# gamma = .01
	# return gamma * Hermitian(operator * rho * operator' - 0.5 * (operator' * operator * rho + rho * operator' * operator));
end

# Target property is coherent information
function target(rho)
	coeff = matrix_to_coeff(rho);
	bloch_norm = min(1,norm(coeff[2:4]) * 2);
	entropy_emt = bloch_norm == 1 ? 0 : -(1+bloch_norm)/2 * log((1+bloch_norm)/2) - (1-bloch_norm)/2 * log((1-bloch_norm)/2);
	lambdas = [max(0.0, l) for l in eigvals(Hermitian(rho))];
	entropy = reduce(+, [lambda == 0.0 ? 0 : -lambda * log(lambda) for lambda in lambdas]);
	return entropy - entropy_emt;
end

function grad(rho)
	coeff = matrix_to_coeff(rho);
	# First calculate gradient of S(B)
	reduced_matrix = Hermitian(reduce(+, [x*y for (x,y) in zip(coeff[1:4], F4[1:4])]))
	eigvecs_reduced_matrix = eigvecs(reduced_matrix)
	eigvals_reduced_matrix = eigvals(reduced_matrix)
	gradient_reduced_matrix = 0im * F16[1]
	for basis in range(2,4)
		for ev in range(1,2)
			if eigvals_reduced_matrix[ev] == 0.0
				# Derivative is 0
				continue
			else
				derivative_eval = adjoint(eigvecs_reduced_matrix[:,ev]) * F4[basis] * eigvecs_reduced_matrix[:,ev]
				# Derivative of (- λ log λ)
				term = -derivative_eval * (1+NaNMath.log(eigvals_reduced_matrix[ev]))
				gradient_reduced_matrix += term * normalised_F16[basis]
			end
		end
	end
	# Now calculate gradient of S(AB)
	eigvecs_system_matrix = eigvecs(Hermitian(rho))
	eigvals_system_matrix = eigvals(Hermitian(rho))
	gradient_system_matrix = 0 * F16[1];
	for basis in range(2,16)
		for ev in range(1,4)
			derivative_eval = adjoint(eigvecs_system_matrix[:,ev]) * normalised_F16[basis] * eigvecs_system_matrix[:,ev]
			# Derivative of (- λ log λ)
			term = -derivative_eval * (1+NaNMath.log(eigvals_system_matrix[ev]))
			gradient_system_matrix += term * normalised_F16[basis]
		end
	end
	return gradient_system_matrix - gradient_reduced_matrix;
end

function get_hamiltonian(rho)
	dissipation = dissipator(rho);
	gradient = grad(rho);
	return get_hamiltonian(rho, dissipation, gradient);
end

function get_hamiltonian(rho, dissipation, gradient)
	alpha = -hsinner(gradient, dissipation)/hsinner(commutator(rho, gradient), commutator(rho, gradient));
	return Matrix(Hermitian(-1.0im * alpha * commutator(rho, gradient)));
end

function lindblad(rho, p, t)
	# return -1.0im * commutator(F16[2], rho)
	if any(isnan, rho)
		return Inf * F16[1];
	end
	dissipation = dissipator(rho);
	gradient = grad(rho);
	hamiltonian = get_hamiltonian(rho, dissipation, gradient);
	if any(isnan, hamiltonian)
		return Inf * F16[1]
	end
	return (-1.0im * commutator(hamiltonian, rho) + dissipation);
end

###################
### Tests begin ###
###################
r = 0.5 * [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]; # Maximally entangled state
@assert target(r) == -log(2);
@assert matrix_to_coeff(r) == 0.5 * [1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1];

r = coeff_to_matrix(zeros(15)); # Maximally mixed state
@assert target(r) == log(2);
###################
###  Tests end  ###
###################

function get_random_density_matrix()
	x = rand(ComplexF64, (4,4))
	x = x * adjoint(x)
	return x/tr(x)
end
# Initial rho
# rho = (1/2+0.0im) * kron([1;0;0;1], [1 0 0 1]);
# rho = 0.6 * rho + 0.4 * get_random_density_matrix();
rho = get_random_density_matrix();
print(rho, "\n");
print(eigvals(rho), "\n");

tend = 10.0;
problem = ODEProblem(lindblad, rho, (0.0, tend));
sol = solve(problem, tstops = 0.0:0.1:tend);

if sol.t[end] < tend
	print("Ended early at ", sol.t[end], "\n")
end

times = sol.t;
targets = [target(u) for u in sol.u];
purity = [tr(Hermitian(u*u)) for u in sol.u];

plotly()
plot(times, purity, show=true, label="Purity");
plot(times, targets, show=true, label="Coherent info");
