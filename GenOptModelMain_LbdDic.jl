@everywhere push!(LOAD_PATH, homedir())

NsSet = 5;
G_Set = [1;2;3;4];
for nsIdx = 1:length(NsSet)
    for gIdx = 1:length(G_Set)
#         @everywhere begin
        #     "/home/gridsan/shelard/julia_DC"
            # Folder with simulated failure scenario sets
        #     resultsFolder = "/home/gridsan/shelard/output/lbdDic/sysPerf/";
            dirSim = "/home/gridsan/changd/NHPPResults/";
            resultsFolder = "/home/gridsan/changd/juliaOutput/lbdDic/"

            optSP = 2; optMobile = 1;
            opt = [optSP; optMobile];

            N = 36; track = 1; Ns = NsSet[nsIdx];

            # global jmax = 200;
            global timeElapsedSce = zeros(Float64, jmax, Ns);

            ## Create matrix of failure scenarios considered
            doc1 = string(dirSim,"Simulations",track,"_",N,"Node",".csv");
        #     doc1 = string(dirSim,"SimulationsCat1","_",N,"Node",".csv"); kmvread = CSV.read(doc1);
            kmvread = CSV.read(doc1);
            global km0 = zeros(Int64,N,Ns); noFailures = zeros(Int64,Ns);
            for k = 1:Ns
                 km0[:,k] = kmvread[k];
                 noFailures[k] = sum(km0[:,k]);
            end
            km0[1,:] = 1*ones(Int64,1,Ns);

            ## Parameters for two-stage stochastic program
            G = G_Set[gIdx]; Y = 2; T = 15;

            ## Set global parameters // initiate model of distribution test feeder
            setGlobalParameters(optMobile);

            println("No. scenarios: ", Ns)
            println("No. generators: ", G)
#         end

        # Benders decomposition
        start = time();
        nCurrentIter, xsv, xv, objectives, LBs, UBs, sysPerf, idx, objValMean = getBendersMethodCascade(km0, G, Y, T, opt);
        elapsed = time() - start; println("Total time: ", elapsed);

        println(sysPerf)

        writedlm(string(resultsFolder,"ResultsLBD_noIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), nCurrentIter);
        writedlm(string(resultsFolder,"ResultsLBD_totalTime_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), elapsed);
        writedlm(string(resultsFolder,"ResultsLBD_timeSceIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), timeElapsedSce[1:nCurrentIter,:]);
        writedlm(string(resultsFolder,"ResultsLBD_timeMasterIter_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), timeElapsedMaster[1:nCurrentIter]);
        writedlm(string(resultsFolder,"ResultsLBD_objTotal_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), objectives);
        writedlm(string(resultsFolder,"ResultsLBD_LBs_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), LBs);
        writedlm(string(resultsFolder,"ResultsLBD_UBs_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), UBs);
        writedlm(string(resultsFolder,"ResultsLBD_allocations_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), xv);
        writedlm(string(resultsFolder,"ResultsLBD_SP_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), sysPerf);
        writedlm(string(resultsFolder,"ResultsLBD_objValMean_",N,"node_Track",track,"_G",G,"_Y",Y,"_Ns",Ns,".txt"), objValMean);
    end
end
