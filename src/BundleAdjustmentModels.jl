module BundleAdjustmentModels

using CUDA
using Pkg.Artifacts,
  Pkg.PlatformEngines, NLPModels, .Threads, CodecBzip2, SHA, DataFrames, JLD2, LinearAlgebra

const ba_jld2 = joinpath(@__DIR__, "ba_probs_df.jld2")
const ba_artifacts = joinpath(@__DIR__, "..", "Artifacts.toml") |> normpath

include("BundleAdjustmentProblemsList.jl")
include("BundleAdjustmentNLSFunctions.jl")
include("BundleAdjustmentArtifactFunctions.jl")
include("ReadFiles.jl")
include("JacobianByHand.jl")


"""
    gpu(nls::BundleAdjustmentModel)

Creates a new BundleAdjustmentModel with all its data moved to the GPU.
This is a non-mutating operation.
"""
function gpu(nls::BundleAdjustmentModel)
    # 1. Ensure a functional GPU is available
    if !CUDA.functional()
        error("Cannot move model to GPU: CUDA is not functional.")
    end

    # 2. Create copies of all relevant CPU arrays on the GPU
    x0_gpu = CuArray(nls.meta.x0)
    pt2d_gpu = CuArray(nls.pt2d)
    cams_indices_gpu = CuArray{Int}(nls.cams_indices)
    pnts_indices_gpu = CuArray{Int}(nls.pnts_indices)

    # Temporary storage vectors
    k_gpu = CuArray(nls.k)
    P1_gpu = CuArray(nls.P1)
    P1_vec_gpu = CuArray(nls.P1_vec)
    P1_cross_gpu = CuArray(nls.P1_cross)
    P2_vec_gpu = CuArray(nls.P2_vec)

    # Temporary storage matrices
    JProdP321_gpu = CuArray(nls.JProdP321)
    JProdP32_gpu = CuArray(nls.JProdP32)
    JP1_mat_gpu = CuArray(nls.JP1_mat)
    JP2_mat_gpu = CuArray(nls.JP2_mat)
    JP3_mat_gpu = CuArray(nls.JP3_mat)
    
    T = eltype(x0_gpu)
    S = typeof(x0_gpu)

    # 3. Call our new constructor with all the GPU arrays
    return BundleAdjustmentModel(
        NLPModelMeta(nls.meta.nvar, x0=x0_gpu, name=nls.meta.name),
        NLSMeta(nls.nls_meta.nequ, nls.meta.nvar, x0=x0_gpu, nnzj=nls.nls_meta.nnzj),
        NLSCounters(),
        cams_indices_gpu,
        pnts_indices_gpu,
        pt2d_gpu,
        nls.nobs,
        nls.npnts,
        nls.ncams,
        k_gpu,
        P1_gpu,
        JProdP321_gpu,
        JProdP32_gpu,
        JP1_mat_gpu,
        JP2_mat_gpu,
        JP3_mat_gpu,
        P1_vec_gpu,
        P1_cross_gpu,
        P2_vec_gpu,
    )
end

gpu(nls::BundleAdjustmentModel) = convert(CuArray, deepcopy(nls))


end
