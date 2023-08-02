using DifferentialEquations;
using LinearAlgebra;
using Plots;
using NaNMath;
using DiffEqCallbacks;
using Debugger;

F4 = Matrix{ComplexF64}[];
push!(F4, [1 0;0 1]);
push!(F4, [0 1;1 0]);
push!(F4, [0 -1im;1im 0]);
push!(F4, [1 0;0 -1]);

matrix_to_coeff(matrix) = [real(tr(x' * matrix)) for x in F4[2:4]];
function coeff_to_matrix(coeff)
	if length(coeff) == 3 # bloch vector
		return 0.5 * reduce(+, [x*y for (x,y) in zip(cat([1],coeff), F4)])
	else
		throw(DomainError("Coeff length should be 3"))
	end
end

hsinner(x, y) = tr(x' * y);
commutator(x, y) = x*y - y*x;

function dissipator(rho)
	operator = F4[3];
	gamma = 1
	return gamma * Hermitian(operator * rho * operator' - 0.5 * (operator' * operator * rho + rho * operator' * operator));
end

function lindblad(rho, p, t)
	return dissipator(rho);
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

function PositivityCheckCallback()
    affect! = function (ρ, t, integrator)
    	@show ρ, t
        if !check_positivity(ρ)
            @warn "The density matrix becomes negative at time $t."
            @show ρ
            terminate!(integrator)
        end
        u_modified!(integrator, false)
    end
    FunctionCallingCallback(affect!, func_everystep=true, func_start=false)
end

x = rand(2,2) + 1.0im * rand(2,2);
rho = x' * x;
rho = Hermitian(rho / tr(rho));
print(rho, "\n");
# print(eigvals(rho), "\n");

tend = 100.0;
problem = ODEProblem(lindblad, rho, (0.0, tend));
sol = solve(problem, alg_hints=[:nonstiff], tstops = 0.0:0.1:tend, callback = PositivityCheckCallback());
sol.t[end]











