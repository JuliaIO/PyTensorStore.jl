using PyTensorStore
using Test
using PythonCall

@testset "PyTensorStore.jl" begin
    spec_dict = Dict(
        "driver" => "zarr",
        "kvstore" => "memory://test_runtests",
        "dtype" => "int32",
        "metadata" => Dict("shape" => [10, 20]),
        "schema" => Dict("domain" => Dict("labels" => ["x", "y"])),
        "create" => true,
        "delete_existing" => true
    )

    fw = PyTensorStore.open(spec_dict)
    @test fw isa PyTensorStore.FutureWrapper{PyTensorStore.TensorStoreWrapper}
    
    w = fw.result()
    @test w isa PyTensorStore.TensorStoreWrapper
    
    @testset "Array Interface" begin
        @test eltype(w) == Int32
        @test ndims(w) == 2
        @test size(w) == (10, 20)
        @test axes(w) == (1:10, 1:20)
        @test firstindex(w, 1) == 1
        @test lastindex(w, 1) == 10
        @test lastindex(w, 2) == 20

    end

    @testset "Write & Read Operations" begin
        data = Int32.(reshape(1:200, (10, 20)))
        w[1:10, 1:20] = data
        @test all(w.read().result() .== data)
        
        data2 = Int32.(rand(1:100, 10, 20))
        w.write(data2).result()
        @test all(w.read().result() .== data2)
    end

    @testset "Domain Manipulation" begin
        domain = w.domain
        @test domain isa PyTensorStore.IndexDomainWrapper
        @test PyTensorStore.labels(domain) == ["x", "y"]
        
        # Labeled indexing
        sub_w = w[x=1:5, y=11:15]
        @test size(sub_w) == (5, 5)
        @test axes(sub_w) == (1:5, 11:15)
        @test firstindex(sub_w, 1) == 1
        @test lastindex(sub_w, 1) == 5
        @test lastindex(sub_w, 2) == 15
        
        # translate_by
        tw = PyTensorStore.translate_by(w, 10, 20)
        @test axes(tw) == (11:20, 21:40)
        
        # translate_to
        tw2 = PyTensorStore.translate_to(w, 101, 201)
        @test axes(tw2) == (101:110, 201:220)
    end

    @testset "Transactions" begin
        txn = PyTensorStore.transaction()
        w_txn = w.with_transaction(txn)
        w_txn[1, 1] = 999
        @test w_txn[1, 1].read().result()[] == 999
        @test w[1, 1].read().result()[] != 999
        
        PyTensorStore.commit_sync(txn)
        @test w[1, 1].read().result()[] == 999
        
        # Context manager
        PyTensorStore.transaction() do t
            w_t = w.with_transaction(t)
            w_t[2, 2] = 888
        end
        @test w[2, 2].read().result()[] == 888
    end

    @testset "Specs & Schemas" begin
        @test w.spec isa PyTensorStore.SpecWrapper
        @test w.schema isa PyTensorStore.SchemaWrapper
        @test w.schema.chunk_layout isa PyTensorStore.ChunkLayoutWrapper
    end

    @testset "Context" begin
        ctx_dict = Dict("cache_pool" => Dict("total_bytes_limit" => 10^6))
        ctx = PyTensorStore.context(ctx_dict)
        @test ctx isa PyTensorStore.ContextWrapper
        
        # Opening with context should work
        w_ctx = PyTensorStore.open(spec_dict, context=ctx).result()
        @test w_ctx isa PyTensorStore.TensorStoreWrapper
    end
end
