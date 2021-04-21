begin
    NsSet = [5;30];
    G_Set = [1;2;3;4];
    for nsIdx = 1:length(NsSet)
        for gIdx = 1:length(G_Set)

            @everywhere push!(LOAD_PATH, homedir())

#             @everywhere begin
                using DelimitedFiles
                resultsFolder = "/home/gridsan/changd/juliaOutput/nbd/";
                dirSim = "/home/gridsan/changd/NHPPResults/";
                global sce; track = 1;

                N = 36; T = 15; B = 36; verbose = false; eps = 0.1;
                G = G_Set[gIdx]; CB = 2; Ns = NsSet[nsIdx]; sce = 1; jmax = 200; optSP = 1; optMobile = 2;
                opt = [0; 1e-10; 1; 0; 1000; optSP; optMobile];
                setGlobalParameters(optMobile);

                dualConstantCollect = zeros(Float64, Ns, T, jmax);
                dualVarCollect = zeros(Float64, Ns, 7*N+2, T, jmax);
                siteDevCollect = zeros(Float64, Ns, length(SITE) + length(RES) + 1, T, jmax);
                derAllocCollect = zeros(Float64, Ns, length(SITE), length(RES), T, jmax);
                yCollect = zeros(Int64, Ns, N, T, jmax);
                rcCollect = zeros(Float64, Ns, length(LOADS), T, jmax);

                objSlaveCollect = zeros(Float64, T, Ns, jmax);
                costToGo = zeros(Float64, T, Ns, jmax);
                derReallocSoln = zeros(Int64, N, length(RES), Ns, jmax);
                derReallocSolnBD = zeros(Int64, N, length(RES), Ns, jmax);

                doc1 = string(dirSim,"Simulations",track,"_",N,"Node",".csv");
        #         doc1 = string(dirSim,"SimulationsCat1","_",N,"Node",".csv");
                kmvread = CSV.read(doc1);
                global km0 = zeros(Int64,N,Ns);
                for k = 1:Ns
                  km0[:,k] = kmvread[k];
                end
                km0[1,:] = 1*ones(Int64,1,Ns); #km0_1 = km0;
#             end

            start = time();
            nCurrentIter, xsv, xv, objectives, LBs, UBs, sysPerf, idx, objValMean = getBendersMethodCascade(km0, G, opt);
            xvOpt = xv[idx,:,:]; xsvOpt = xsv[idx,:];
            elapsed = time() - start; println("Total time: ", elapsed);

            Y = CB;

            writedlm(string(resultsFolder,"ResultsLBD_noIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), nCurrentIter);
            writedlm(string(resultsFolder,"ResultsLBD_totalTime_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), elapsed);
        #     writedlm(string(resultsFolder,"ResultsLBD_timeSceIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), timeElapsedSce[1:nCurrentIter,:]);
        #     writedlm(string(resultsFolder,"ResultsLBD_timeMasterIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), timeElapsedMaster[1:nCurrentIter]);
            writedlm(string(resultsFolder,"ResultsLBD_objTotal_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), objectives);
            writedlm(string(resultsFolder,"ResultsLBD_LBs_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), LBs);
            writedlm(string(resultsFolder,"ResultsLBD_UBs_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), UBs);
            writedlm(string(resultsFolder,"ResultsLBD_allocations_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), xv);
            writedlm(string(resultsFolder,"ResultsLBD_SP_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), sysPerf);
            writedlm(string(resultsFolder,"ResultsLBD_objValMean_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), objValMean);

        end
    end
end
