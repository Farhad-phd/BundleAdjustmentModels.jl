using Test
using BundleAdjustmentModels
using CUDA
using BenchmarkTools

# Choose a large problem to see the GPU's advantage
problem_name = "problem-49-7776-pre" 
nls_cpu = BundleAdjustmentModel(problem_name)
x_cpu = copy(nls_cpu.meta.x0)
vals_cpu = zeros(nls_cpu.nls_meta.nnzj)

println("--- CPU Performance ---")
print("Residual calculation: ")
@btime residual!($nls_cpu, $x_cpu, $(similar(nls_cpu.nls_meta.x0, nls_cpu.nls_meta.nequ)))
print("Jacobian calculation: ")
@btime jac_coord_residual!($nls_cpu, $x_cpu, $vals_cpu)

# Check if a GPU is available before running GPU tests
if CUDA.functional()
    println("\n--- GPU Performance ---")
    
    # Create the model by moving the initial data to the GPU
    nls_gpu = BundleAdjustmentModel(problem_name) |> gpu
    x_gpu = CuArray(x_cpu)
    vals_gpu = CUDA.zeros(nls_gpu.nls_meta.nnzj)

    # Warm-up GPU to compile kernels
    residual!(nls_gpu, x_gpu, similar(x_gpu, nls_gpu.nls_meta.nequ))
    jac_coord_residual!(nls_gpu, x_gpu, vals_gpu)

    print("Residual calculation (GPU): ")
    @btime CUDA.@sync residual!($nls_gpu, $x_gpu, $(similar(x_gpu, nls_gpu.nls_meta.nequ)))
    print("Jacobian calculation (GPU): ")
    @btime CUDA.@sync jac_coord_residual!($nls_gpu, $x_gpu, $vals_gpu)
else
    println("\nSkipping GPU performance tests: CUDA is not functional.")
end



@testset "GPU vs. CPU Correctness" begin
    if !CUDA.functional()
        @warn "Skipping GPU correctness tests: CUDA is not functional."
        return # Exit the testset if no GPU is available
    end

    nls_cpu = BundleAdjustmentModel("problem-49-7776-pre")
    x_cpu = rand(nls_cpu.meta.nvar) # Use a random vector for a robust test
    
    # 1. Create a GPU version of the model and variables
    nls_gpu = gpu(nls_cpu) # We'll need to define this helper function
    x_gpu = CuArray(x_cpu)

    # 2. Calculate residuals on both devices
    res_cpu = residual(nls_cpu, x_cpu)
    res_gpu = residual(nls_gpu, x_gpu)
    
    @testset "Residual Correctness" begin
        # Compare results (copy GPU vector back to CPU for comparison)
        @test Array(res_gpu) ≈ res_cpu rtol=1e-5
    end

    # 3. Calculate Jacobian values on both devices
    vals_cpu = jac_coord_residual(nls_cpu, x_cpu)
    vals_gpu = jac_coord_residual(nls_gpu, x_gpu)

    @testset "Jacobian Correctness" begin
        # Compare results
        @test Array(vals_gpu) ≈ vals_cpu rtol=1e-4
    end
end
