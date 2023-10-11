using .CUDA
using .LinearMaps
import LinearAlgebra.mul!
import Base.size
export CGIndirectKKTSolverGPU

mutable struct IndirectReducedKKTSolverGPU{T} <: AbstractKKTSolver 
    m::Integer
    n::Integer
    P::CUSPARSE.CuSparseMatrixCSC{T, Int32}
    A::CUSPARSE.CuSparseMatrixCSC{T, Int32}
    σ::T
    ρ::CuVector{T}
    tol_constant::T
    tol_exponent::T
    # Memory and statistics for cg
    solver_type::Symbol
    previous_solution::CuVector{T}
    iteration_counter::Int
    multiplications::Vector{Int}
    tmp_n::CuVector{T} # n-dimensional used to avoid memory allocation
    tmp_m::CuVector{T} # m-dimensional used to avoid memory allocation
	y1gpu::CuVector{T}
	y1cpu::Vector{T}
	y2gpu::CuVector{T}
	y2cpu::Vector{T}
	x1gpu::CuVector{T}
	x1cpu::Vector{T}
	x2gpu::CuVector{T}
	x2cpu::Vector{T}
    function IndirectReducedKKTSolverGPU(P, A, σ::T, ρ;
        solver_type::Symbol=:CG, tol_constant::T=T(1.0), tol_exponent::T=T(3.0)
        ) where {T}
        m, n = size(A)
        if isa(ρ, T)
            ρ = ρ*CUDA.ones(T, m)
        end
        @assert eltype(P) == eltype(A) == T "Inconsistent element types."
        @assert solver_type == :CG || solver_type == :MINRES "Solver symbol must be either :CG or :MINRES"
        new{T}(m, n, CUSPARSE.CuSparseMatrixCSC(P), CUSPARSE.CuSparseMatrixCSC(A), σ, ρ,
            tol_constant, tol_exponent,
            solver_type,
            CUDA.zeros(T, n), 1, zeros(Int, 0), CUDA.zeros(T, n), CUDA.zeros(T, m), CUDA.zeros(T,n), zeros(T,n), CUDA.zeros(T,m), zeros(T,m), CUDA.zeros(T,n), zeros(T,n), CUDA.zeros(T,m), zeros(T,m))
    end
end

function size(S::IndirectReducedKKTSolverGPU, n::Int)
    return S.m + S.n
end

function mul!(y::CuVector{T}, S::IndirectReducedKKTSolverGPU, x::CuVector{T}) where {T}
    mul!(S.tmp_m, S.A, x)
    @. S.tmp_m .*= S.ρ
    mul!(S.tmp_n, S.A', S.tmp_m)
    axpy!(S.σ, x, S.tmp_n)
    mul!(y, S.P, x)
    axpy!(one(T), S.tmp_n, y)
    S.multiplications[end] += 1
    return y
end


function solve!(S::IndirectReducedKKTSolverGPU, y::AbstractVector{T}, x::AbstractVector{T}) where {T}
    # Solves the (KKT) linear system
    # | P + σI     A'  |  |y1|  =  |x1|
    # | A        -I/ρ  |  |y2|  =  |x2|
    # x1,y1: R^n, x2/y2: R^m
    # where [y1; y2] := y, [x1; x2] := x

    # In particular we perform cg/minres on the reduced system
    # (P + σΙ + Α'ρΑ)y1 = x1 + A'ρx2
    # And then recover y2 as
    # y2 = ρ(Ay1 - x2)
	
    copyto!(S.x1cpu, view(x, 1:S.n))
    copyto!(S.x2cpu, view(x, S.n+1:S.n+S.m))
	copyto!(S.x1gpu, S.x1cpu)
    copyto!(S.x2gpu, S.x2cpu)
	
   # x1 = view(S.x, 1:S.n); y1 = view(S.y, 1:S.n)
    #x2 = view(S.x, S.n+1:S.n+S.m); y2 = view(S.y, S.n+1:S.n+S.m)


    # Form right-hand side for cg/minres: y1 = x1 + A'ρx2
	S.y2gpu .= S.x2gpu
	S.y2gpu .*= S.ρ
    mul!(S.y1gpu, S.A', S.y2gpu)
    S.y1gpu .+= S.x1gpu
	
    push!(S.multiplications, 0)

    if S.solver_type == :CG
        cg!(S.previous_solution, S, S.y1gpu, abstol=get_tolerance(S)/norm(S.y1gpu))
    elseif S.solver_type == :MINRES
        init_residual = norm(L*S.previous_solution - S.y1gpu)
        minres!(S.previous_solution, S, S.y1gpu, tol=get_tolerance(S)/init_residual)
    end
    # Sanity check for tolerance
    # might not always hold for MINRES, as its termination criterion is approximate, (see its documentation)
    # @assert get_tolerance(S) > norm(L*S.previous_solution - y1)

    # y2 = Ay1 - x2
    mul!(S.y2gpu, S.A, S.y1gpu)
    axpy!(-one(T), S.x2gpu, S.y2gpu)
    S.y2gpu .*= S.ρ

    S.iteration_counter += 1
    copyto!(S.y1cpu, S.y1gpu)
	copyto!(S.y2cpu, S.y2gpu)
	copyto!(view(y, 1:S.n), S.y1cpu)
	copyto!(view(y, S.n+1:S.n+S.m), S.y2cpu)

    return y
end

function update_rho!(S::IndirectReducedKKTSolverGPU, ρ)
    if(isbits(ρ)) # scalar
        S.ρ .= ρ
    else          # vector
        copyto!(S.ρ, ρ)
    end
end

function get_tolerance(S::IndirectReducedKKTSolverGPU)
    return S.tol_constant/S.iteration_counter^S.tol_exponent
end


struct CGIndirectKKTSolverGPU{T} <: AbstractKKTSolver
    indirect_kktsolver::IndirectReducedKKTSolverGPU{T}
    function CGIndirectKKTSolverGPU(P, A, σ::T, ρ; tol_constant::T=T(1.0), tol_exponent::T=T(3.0),) where {T}
        new{T}(IndirectReducedKKTSolverGPU(P, A, σ, ρ; solver_type = :CG, tol_constant = tol_constant, tol_exponent = tol_exponent))
    end
end


update_rho!(S::CGIndirectKKTSolverGPU, ρ) = update_rho!(S.indirect_kktsolver, ρ)
get_tolerance(S::CGIndirectKKTSolverGPU, ρ) = get_tolerance(S.indirect_kkt_solver, ρ)
solve!(S::CGIndirectKKTSolverGPU, y::AbstractVector{T}, x::AbstractVector{T}) where {T} = solve!(S.indirect_kktsolver, y, x)
