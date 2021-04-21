using JuMP, Distributed

## PACKAGES
@everywhere using JuMP, Gurobi, Combinatorics, DistributedArrays
@everywhere using LinearAlgebra, Random, Statistics, CSV, DelimitedFiles
pwd()

## BENDERS DECOMPOSITION
@everywhere function getBendersMethodCascade(kmv, G, Y, T, opt)

    optSP = opt[1]; optMobile = opt[2];

    # Get initial solution to Master Problem (MP) -- no Benders cuts added
    mm, x, xs, theta, kcm = getMasterModel(G, Ns, kmv);

    println("Entering loop");
    println("-------------------------------")

    nCurrentIter = 0;
    LBs = zeros(Float64,jmax); UBs = zeros(Float64,jmax); obj = zeros(Float64,jmax);
    xsv = zeros(Int64, jmax, N); objLb = zeros(Float64, jmax);
    xv = zeros(Int64, jmax, N, G);
    addGreedy = zeros(Int64, jmax);
    for j = 1:jmax
        nCurrentIter += 1; println("Iteration: ", nCurrentIter)

        # Solution from one iteration of Benders decomposition
        mm, isMasterModelNotSolvable, LBs[j], obj[j], objLb[j], xv[j,:,:], xsv[j,:], sysPerf = getBendersCascadeIteration(mm, x, xs, theta, kcm, kmv, G, verbose, j, optMobile);

#         println(mm)

        if sum(xv[j,:,:]) == G && G == 1
            idxPlace = findall(xv[j,:,:] .== 1);
            idxPlaceLoc = idxPlace[1][1];
            @constraint(mm, x[idxPlaceLoc,1] == 0)
        end
        #
        if sum(xv[j,:,:]) == G #&& sum(xsv) == G
            idxPlace = findall(xv[j,:,:] .== 1);
            idxPlaceLoc = zeros(Int64, G);
            for gIter = 1:G
                idxPlaceLoc[gIter] = idxPlace[gIter][1];
            end
            nCombos = factorial(G);
            for constraintIter = 1:nCombos
                idxPlacePerm = nthperm(idxPlaceLoc,constraintIter);

                if G == 2
                    @constraint(mm, x[idxPlacePerm[1],1] + x[idxPlacePerm[2],2] <= G-1);
                elseif G == 3
                    @constraint(mm, x[idxPlacePerm[1],1] + x[idxPlacePerm[2],2] + x[idxPlacePerm[3],3]  <= G-1);
                end
            end
        elseif j > 1 && sum(xv[j,:,:] .== xv[j-1,:,:]) == N*Nres
            # println("ADD GREEDY CONSTRAINT")
            idxPlace = findall(xv[j,:,:] .== 1);
            @constraint(mm, sum(x[idxPlace]) <= sum(xv[j,:,:])-1);
            addGreedy[j] = 1;
        end

        # print(mm)

        # Update upper bound
        # Lower bound updated from solution to updated Master Problem (MP)
        UBs[j] = minimum(obj[1:nCurrentIter]);
        println("Lower bound: ", LBs[j]); println("Upper bound: ", UBs[j]);

        # Check for algorithm convergence
        if isMasterModelNotSolvable
            LBs[j] = -Inf; UBs[j] = Inf;
            println("-------------------------------")
            println("Exit loop")
            break
        elseif (UBs[j]-LBs[j])/LBs[j] < 0.01
            println("-------------------------------")
            println("Exit loop")
            break
        # elseif j > 1 && sum(xv[j,:,:]) == 0
        #     println("-------------------------------")
        #     println("Exit loop")
        #     break
        end

        println("-------------------------------")
    end

    # Get optimal solution and corresponding system performance
    idx = argmax(-obj[1:nCurrentIter]); #sysPerf = sysPerf[idx,:];

    xsv = xsv[1:nCurrentIter,:]; objectivesLb = objLb[1:nCurrentIter];
    xv = xv[1:nCurrentIter,:,:]; objectives = obj[1:nCurrentIter];
    LBs = LBs[1:nCurrentIter]; UBs = UBs[1:nCurrentIter];

    xvOpt = xv[idx,:,:]; xsvOpt = xsv[idx,:];

    sysPerf = zeros(Float64,Ns,T+2);

    objVal = zeros(Float64,Ns);
    if optSP == 1
        for i = 1:Ns
            xvOpt = xv[idx,:,:]; xsvOpt = xsv[idx,:];
            objVal[i], sysPerf[i,:] = getSolutionRES_MILP(km0[:,i], G, Y, xvOpt, xsvOpt);
        end
    elseif optSP == 2
        for i = 1:Ns
            objVal[i], sysPerf[i,:] = getSolutionRES_MILP_Greedy(km0[:,i], G, Y, xvOpt, xsvOpt);

            # println("System performance: ", sysPerf[i,:]);
        end
    end
    sysPerf = mean(sysPerf, dims = 1);
    objValMean = mean(objVal);

    nCurrentIter, xsv, xv, objectives, LBs, UBs, sysPerf, idx, objValMean
end

@everywhere function getBendersCascadeIteration(mm, x, xs, theta, kcm, kmv, G, verbose, iter, optMobile)

    isMasterModelNotSolvable = false;

    # if verbose
        # println(mm);
    # end

    # Solve master problem (MP)
    start = time();

#     println(mm)

    optimize!(mm); timeElapsedMaster[iter] = time() - start;
    println("Time elapsed for Master Problem (MP): ", timeElapsedMaster[iter])
    mstatusCheck = termination_status(mm);
    println(mstatusCheck)
    if mstatusCheck == MOI.OPTIMAL
        xv, xsv, kcmv, genIdxSim = getResultsMaster(x, xs, kcm);
        objMaster = JuMP.objective_value(mm);
    else
        println(mm)
    end

    println("Resource placement: ", xv[SITE,RES]);
#     println("Site development: ", xsv[SITE]);

    # # Get dual cost vectors
    # genbsVar = addBendersCutGen(x); droopbsVar = addBendersCutDroop(x);

    # Get subproblem solution + Benders cut components for all scenarios
    # Permits distributed computation across multiple cores
    (objvLbSum, objvSum, bdCut, siteDev0_sum, fixedSum, mobileSum, sysPerf, rcSum, rcSdSum, rcDerSum, acSum) = @distributed addObjBDcut for sce =  1:Ns
        objSlaveLb, objSlave, bendersCutLS, siteDev0, fixed, mobile, totalCostNorm, rc_kc, rc_sd, rc_der, ac = getBendersCascadeSceIter(km0, G, Y, sce, iter, x, xv, xsv, optMobile);
        (objSlaveLb, objSlave, bendersCutLS, siteDev0, fixed, mobile, totalCostNorm, rc_kc, rc_sd, rc_der, ac)
    end

    # Add new Benders cuts to Master Problem (MP)
    for sce = 1:Ns
        mm = addBendersCut(mm, sce, iter, bdCut, siteDev0_sum, fixedSum, mobileSum, theta, kcm, rcSum, rcSdSum, rcDerSum, acSum, x, xs, optMobile);
    end

    # Update lower / upper bounds
    objLB = objMaster; objTotal = mean(objvSum) + sum(WSITE[SITE].*xsv[SITE]);
    objTotalLb = mean(objvLbSum) + sum(WSITE[SITE].*xsv[SITE]);

    # Calculate mean system performance
    # sysPerf = mean(sysPerf, dims=1);
    sysPerf = 0;

    mm, isMasterModelNotSolvable, objLB, objTotal, objTotalLb, xv, xsv, sysPerf
end

## ------------------------------------##
## GET SUBPROBLEM SOLUTION + BENDERS CUT COMPONENTS FOR SINGLE SCENARIO
@everywhere function getBendersCascadeSceIter(km0, G, Y, sce, iter, x, xv, xsv, optMobile)

    start = time();

    # Get Stage II subproblem (SP2) model
    xvMP = xv; xsvMP = xsv;
    sm, p, q, pg, qg, beta, P, Q, v, t, kc, v0, y, vpar, xs, x, xr, basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqbs, genIneq, genIneqbs, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum = getSlaveModel(km0[:,sce], G, Y, 1, xvMP, xsvMP, 1);

    if verbose
        println("Updated slave model");
        println(sm);
    end

    # Solve (SP2), obtain objective and values of relevant variables
    optimize!(sm);
    sstatus = termination_status(sm);

    if sstatus != MOI.OPTIMAL
        println("Slave model solve status : ", sstatus);
        return;
    end
    objSlave = JuMP.objective_value(sm);

    println("REC objective: ", objSlave)

    kcv = JuMP.value.(kc); yv = JuMP.value.(y); betav = JuMP.value.(beta);
    xv = JuMP.value.(x); xsv = JuMP.value.(xs); xrv = JuMP.value.(xr);

    # Calculate period-wise system performance
    # totalCostNorm = printResultsSP(1, ov, uv, kcv, betav, sce);
    totalCostNorm = 0;

    # Get Benders cut components
    bendersCutLS, rc_kc, rc_sd, rc_der, ac, objRcModel, siteDev0_bar, reallocFixedBar, reallocMobileBar = getBendersCutComponents(km0[:,sce], xvMP, basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqbs, genIneq, genIneqbs, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs, xsvMP, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum, optMobile);

    timeElapsedSce[iter, sce] = time() - start;

    objSlaveLb = objSlave;
    objSlave += objRcModel;

    if size(siteDev0_bar)[1] != length(SITE)
        println("Wrong dimensions")
    elseif size(reallocFixedBar)[1] != length(SITE) || size(reallocFixedBar)[2] != length(RESf) || size(reallocFixedBar)[3] != T
        println("Wrong dimensions")
    elseif size(reallocMobileBar)[1] != length(SITE) || size(reallocMobileBar)[2] != length(RESm)
        println("Wrong dimensions")
    elseif size(rc_kc)[1] != length(LOADS) || size(rc_kc)[2] != T
        println("Wrong dimensions")
    elseif size(rc_sd)[1] != length(SITE) || size(rc_sd)[2] != T
        println("Wrong dimensions")
    elseif size(rc_der)[1] != length(SITE) || size(rc_der)[2] != length(RES) || size(rc_der)[3] != T
        println("Wrong dimensions")
    elseif size(ac)[1] != length(LOADS) || size(ac)[2] != length(SITE) || size(ac)[3] != T
        println("Wrong dimensions")
    else
#         println("Good")
    end

    # println(size(objSlaveLb))
    # println(size(objSlave))
    # println(bendersCutLS)
    # println(size(siteDev0_bar))
    # println(size(reallocFixedBar))
    # println(size(reallocMobileBar))
    # println(size(totalCostNorm))
    # println(size(rc_kc))
    # println(size(rc_sd))
    # println(size(rc_der))
    # println(size(ac))

    return objSlaveLb, objSlave, bendersCutLS, siteDev0_bar, reallocFixedBar, reallocMobileBar, totalCostNorm, rc_kc, rc_sd, rc_der, ac
end

## Concatenate variables from distributed computation
@everywhere function addObjBDcut(tupleA, tupleB)
    (objSlaveLbA, objSlaveA, bendersCutLSA, siteDevA, fixedA, mobileA, spA, rcA, rcSdA, rcDerA, acA) = tupleA;
    (objSlaveLbB, objSlaveB, bendersCutLSB, siteDevB, fixedB, mobileB, spB, rcB, rcSdB, rcDerB, acB) = tupleB;

    ([objSlaveLbA; objSlaveLbB], [objSlaveA; objSlaveB], [bendersCutLSA; bendersCutLSB], [siteDevA; siteDevB], [fixedA; fixedB], [mobileA; mobileB], [spA ;spB], [rcA; rcB], [rcSdA; rcSdB], [rcDerA; rcDerB], [acA; acB])
end

## ------------------------------------##
## GET SCENARIO-WISE BENDERS CUT AT CURRENT ITERATION
@everywhere function addBendersCut(mm, sce, iter, bdCut, siteDev0_sum, fixedSum, mobileSum, theta, kcm, rcSum, rcSdSum, rcDerSum, acSum, x, xs, optMobile)

    epsilon = 0.1

    siteDev0 = siteDev0_sum[length(SITE)*(sce-1)+1:length(SITE)*sce];
    fixed = fixedSum[length(SITE)*(sce-1)+1:length(SITE)*sce,:,:];
    mobile = mobileSum[length(SITE)*(sce-1)+1:length(SITE)*sce,:];

    rcSd = rcSdSum[length(SITE)*(sce-1)+1:length(SITE)*sce,:];
    rcDer = rcDerSum[length(SITE)*(sce-1)+1:length(SITE)*sce,:,:];

    #1
    bdCutAdd = getUpdatedBendersConstant(bdCut, sce, siteDev0, fixed, mobile, x, xs, rcSd, rcDer, optMobile);

    rc_kc = rcSum[(sce-1)*length(LOADS)+1:sce*length(LOADS), :];
    ac = acSum[(sce-1)*length(LOADS)+1:sce*length(LOADS), :, :];

    # @constraint(mm, theta[sce] >= bdCutAdd + epsilon);
    #2
    @constraint(mm, theta[sce] >= bdCutAdd + sum(kcm[:,sce] .* rc_kc[:,1]) + epsilon);

    #3
    if iter == 1
        @constraint(mm, [i=1:length(LOADS)], kcm[i,sce] >= 1 - sum(xs[SITE] .* ac[i,:,1]));
        # @constraint(mm, [i=1:length(LOADS)], kcm[i,sce] <= 1 - sum(xs[SITE] .* ac[i,:,1])/length(SITE));
    end

    # println(bdCutAdd + sum(kcm[:,sce,iter] .* rc_kc[:,1]))

    return mm
end

@everywhere function getUpdatedBendersConstant(bdCut, sce, siteDev0_sum, fixedSum, mobileSum, x, xs, rcSd, rcDer, optMobile)

    # idxDroop = [(sce-1)*length(SITE)*2 + 1; (sce-1)*length(SITE)*2 + length(SITE)*2];
    # idxGen = [(sce-1)*length(SITE)*4 + 1; (sce-1)*length(SITE)*4 + length(SITE)*4];
    # bdCutAdd = bdCut[sce] + sum(droopSum[idxDroop[1]:idxDroop[2],:,:] .* droopbsVar) + sum(genSum[idxGen[1]:idxGen[2],:,:] .* genbsVar);

    if Ns == 1
        bdCutAdd = bdCut + sum(xs[SITE] .* siteDev0_sum);
    elseif Ns > 1
        bdCutAdd = bdCut[sce] + sum(xs[SITE] .* siteDev0_sum);
    end

    bdCutAdd += sum(x[SITE,RESm] .* mobileSum);
    for i = 1:T
        bdCutAdd += sum(x[SITE,RESf] .* fixedSum[:,:,i]);
        # if optMobile == 1
        #     bdCutAdd += (sum(xs[SITE] .* rcSd[:,i]) + sum(x[SITE,RES] .* rcDer[:,:,i]));
        # end
    end

    bdCutAdd
end

## ------------------------------------##
## SET GLOBAL PARAMETERS
@everywhere begin
    global resourceResponse = false;
    global deltaV0min = 0;
    global printPerformance = true;
    global trial = false;
    global LTCSetting = 1.05;
    global diversification = false;
    global vreg = 0;
    global P1nom = 0;
    global Q1nom = 0;
    global N = 1;
    global Ns;
    global T = 1;
    global Y = 1;
    global G = 1;
    global costGen = 1;
    global M = 10;
    global verbose = false;
    # global sce;
    global jmax = 1000;
    # global timeElapsedSce = zeros(Float64, jmax, Ns);
    global timeElapsedMaster = zeros(Float64, jmax);
end

## ------------------------------------##
## DEFINE PARAMETERS FOR TEST FEEDER
# IEEE distribution test feeders are used as representative networks for research studies.
# Here, we define necessary parameters for 12-, 36-, and 69-node networks.
@everywhere function setGlobalParameters(optMobile)
    global vmin, vmax, vmmin, vmmax, vgmin, vcmin, WAC, WLC, WVR, WSD, ov, uv
    global xrratio, rc, Cloadc, resistance, reactance, srmax
    global infty, tol, epsilon, I, O, fmin, fmax, WLC, WSD, nodes, WMG, WEG
    global path, commonPath, R, X, betamin, vgmax, vcmax, v0nom, v0dis
    global pcmax, qcmax, LLCmax, pgmax, qgmax, betamax, delta, pdmax, qdmax, Smax, pgnom, qgnom, vref, mp, mq, SITE, Nres, Nsite, WSITE, WDER, WSITE_2, WR

    global myinf, leaves, SDI, noSDI, MGL, noMGL, par, SuccMat, RES, RESf, RESm, EG, semax, DG, UDG, LOADS, CHILD
    global LTC, noLTC, N, onev

    ov = zeros(Float64,N); uv = ones(Float64,N); v0dis = 0.001; epsilon = 0.1^5; infty = 10^5+1; tol = 0.1^3; myinf = 0.1^7;
    v0nom = 1; I = zeros(N,N) + UniformScaling(1); O = zeros(Float64,N,N); vmin = 0.95uv; vmax = 1.05uv; WVR = 10; Smax = 2uv;
    vmmmax = 2uv; vref = 1.00;

    WMG = zeros(Float64,N); WEG = zeros(Float64,N); WLC = zeros(Float64,N); WSD = zeros(Float64,N); WSITE = zeros(Float64,N);
    WDER = zeros(Float64,N); WSITE_2 = zeros(Float64,N); WR = zeros(Float64,N,N,G);
    pcmax = zeros(Float64,N); qcmax = zeros(Float64,N); pgmax = zeros(Float64,N); qgmax = zeros(Float64,N);
    costGen = ones(Int64,N);

    nodes = collect(1:N); EG = copy(nodes); LOADS = copy(nodes); LTC = zeros(Int64,0); DG = copy(nodes); SDI = nodes[nodes.%2 .== 1]; noSDI = setdiff(nodes,SDI); MGL = zeros(Int64,0);
    par = collect(0:N-1);
    if N%3 == 0
        MGL = [1; Int(N/3)+1; 2Int(N/3)+1];
    end
    if N%2 == 0
        EG = nodes[nodes.%2 .== 1];
    end


    Random.seed!(716);
    DG = sort((nodes[randperm(N)])[1:round(Int64,N/2)]);

    if N == 12
        Random.seed!(123113);
        SITE = [1, 4, 9, 12];
        WSITE[SITE] = 1000*ones(size(SITE));
        betamax = 0.5uv; betamin = 1 .- betamax;

        if !diversification
            par[9] = 4;
            xrratio = 2; rc = 0.01; srmax = 1/N; Cloadc = 100;
            srmax = 6/N; # for sequential vs online
            LOADS = setdiff(nodes,DG)
            LOADS_CRIT = [3, 6]
            LOADS_NCRIT = setdiff(LOADS, LOADS_CRIT);
            pcmax[LOADS] = 1.25srmax * ones(size(LOADS)); qcmax = pcmax / 3;
            pgmax[DG] = srmax * ones(size(DG)); qgmax = pgmax / 3;
            WLC[LOADS_CRIT] = 500 * ones(size(LOADS_CRIT)); WLC[LOADS_NCRIT] = 100 * ones(size(LOADS_NCRIT));
            WSD[LOADS_CRIT] = 5000  * ones(size(LOADS_CRIT)); WSD[LOADS_NCRIT] = 1000  * ones(size(LOADS_NCRIT));
            WMG[MGL] = 200 * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
            SDI = copy(nodes); noSDI = [];
            vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
            pdmax = 2srmax * uv; qdmax = pdmax/3;
        else
            par[9] = 4;
            xrratio = 2; rc = 0.01; srmax = 1/N; Cloadc = 100;
            srmax = 3/N; # for sequential vs online
            DG = copy(nodes); LOADS = copy(nodes);
            pcmax[LOADS] = 2.2srmax; qcmax = pcmax / 3;
            pgmax[DG] = srmax; qgmax = pgmax / 3;
            WLC[LOADS] = 100; WSD[LOADS] = 1000;
            WMG[MGL] = 200; WEG[EG] = 200;
            SDI = copy(nodes); noSDI = [];
            vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
            pdmax = 2srmax * uv; qdmax = pdmax/3;
        end
    elseif N == 36
        par[5] = 3; par[6] = 2; par[10] = 7; par[13] = 10; par[15] = 13; par[16] = 2; par[20] = 18; par[21] = 16; par[24] = 22; par[25] = 22; par[27] = 25; par[31] = 29; par[32] = 28; par[36] = 34;
        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100;

        SITE = [1,5,7,8,11,12,20,22,24,29,33,35];
        LOADS = setdiff(nodes, DG);
        LOADS_CRIT = [4,10,20,27]; LOADS_NCRIT = setdiff(LOADS, LOADS_CRIT);

        pcmax[LOADS] = 1.25*srmax * ones(size(LOADS)); qcmax = pcmax / 3;
        pgmax[DG] = srmax* ones(size(LOADS)); qgmax = pgmax / 3;
        WLC[LOADS] = 100 * ones(size(LOADS)); MGL = [5,6,10,13,15,25,32];
        WLC[LOADS_CRIT] = 500 * ones(size(LOADS_CRIT)); WLC[LOADS_NCRIT] = 100 * ones(size(LOADS_NCRIT));
        WSD[LOADS_CRIT] = 5000 * ones(size(LOADS_CRIT)); WSD[LOADS_NCRIT] = 1000 * ones(size(LOADS_NCRIT));
        WMG[MGL] = 400  * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
        WSITE[SITE] = 1000*ones(size(SITE));
        betamax = 0.5uv; betamin = ones(size(betamax)) - betamax;
        pdmax = 3srmax * uv; qdmax = pdmax/3;
        vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
    elseif N == 69
        par[28] = 3; par[36] = 3; par[47] = 4; par[45] = 8; par[53] = 9; par[66] = 11; par[68] = 12;
        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100; gammac = 1; lcfmin = 0.8ones(N,1);
        LOADS = setdiff(nodes, DG);
        # SITE = [2,4,6,7,12,13,14,15,16,17,18,20,21,22,27,30,31,33,34,35,36,46,47,48,54,55,56,58,59,60,64,65,67,69];
        SITE = [2,4,7,12,14,17,18,22,30,31,35,46,48,54,56,58,60,64,67,69];

        pcmax[LOADS] = 1.25srmax * ones(size(LOADS)); qcmax = pcmax / 3;
        pgmax[DG] = srmax * ones(size(DG)); qgmax = pgmax / 3;
        WLC[LOADS] = 100 * ones(size(LOADS)); WSD[LOADS] = 1000 * ones(size(LOADS));
        WMG[MGL] = 400 * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
        vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;

        betamax = 0.2uv; betamin = ones(size(betamax)) - betamax;
        pdmax = 2srmax * uv; qdmax = pdmax/3;
    elseif N == 118
        # missing nodes are 28, 39, 57, 92, 104
        par[4] = 2; par[10] = 2; par[18] = 11; par[28] = 4; par[36] = 30; par[38] = 29; par[47] = 35; par[55] = 29;
        par[63] = 1; par[78] = 64; par[89] = 65; par[96] = 91; par[85] = 79; par[100] = 1; par[114] = 100;
        SITE = [2,4,7,12,14,17,18,22,30,31,35,46,48,54,56,58,60,64,67,69,73,77,81,84,89,95,97,102,106,109,114,117];

        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100; gammac = 1; lcfmin = 0.8 * ones(N,1);

        LOADS = setdiff(nodes, DG);
        pcmax[LOADS] = 1.25srmax * ones(size(LOADS)); qcmax = pcmax / 3;
        pgmax[DG] = srmax * ones(size(DG)); qgmax = pgmax / 3;
        WLC[LOADS] = 100 * ones(size(LOADS)); WSD[LOADS] = 1000 * ones(size(LOADS));
        MGL = [28,36,47,45,53,66];
        WMG[MGL] = 400 * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
        vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.1uv; vcmin = 0.90uv; vcmax = 1.1uv;
        betamax = 0.2uv; betamin = ones(size(betamax)) - betamax;
        pdmax = 2srmax * uv; qdmax = pdmax/3;
    end

    Random.seed!(167); ldg = round(Int64,length(DG)/2);

    RES = collect(1:G);
    resIdx = trunc(Int, ceil(G/2));
    # RESf = collect(1:G); RESm = [];
    if optMobile == 1
        RESf = collect(1:G); RESm = [];
    elseif optMobile == 2
        RESf = 1:G-Gm; RESm = G-Gm+1:G;
    end

    onev = ones(Float64,length(RES),1);

    Nres = length(RES); Nsite = length(SITE);

    EG = copy(MGL);

    noMGL = setdiff(nodes,MGL); noLTC = setdiff(nodes,LTC);
    LLCmax = sum(WSD);
    semax = zeros(Float64, N);
    WAC = zeros(Float64,N); WAC[1] = 0.1WLC[1];
    semax[EG] = 0.5(sum(pcmax))/length(EG)  * ones(size(EG));

    pgnom = 0pgmax; qgnom = 0qgmax; mq = 0.1;

    UDG = resourceResponse ? setdiff(DG,RES) : DG;
    UDG = diversification ? DG : UDG;

    noMGL = setdiff(nodes, MGL);
    leaves = getLeaves(par);
    CHILD = getChildrenMatrix(par);
    SuccMat = getSuccessorMatrix();

    r = rc * ones(Float64,N); x = xrratio * copy(r);
    resistance = copy(r); reactance = copy(x);
    path = getPath(par);
    commonPath, R, X = getCommonPathVariables(N,par,r,x);

    pgmax[SITE] = 0.8sum(pcmax) / G * ones(size(SITE));
    qgmax = pgmax/3; mq = 0.1;
end

@everywhere function getLeaves(par)
    N = length(par); isLeaf = ones(Int64,N);
    for i = 1:length(par)
        par[i] != 0 ? isLeaf[par[i]] = 0 : continue;
    end
    leaves = findall(isLeaf .== 1);
    leaves
end

@everywhere function getPath(par)
    N = length(par);
    path = zeros(Int64,N);
    for i=1:N
        j = i;
        while j!=0
            path[i] += 1;
            j = par[j];
        end
    end
    path
end

@everywhere function getPathPair(pair, par)
    dest = maximum(pair);
    origin = minimum(pair);
    subpath = getSubPath(origin, dest, par);
    pathDest = subpath; pathOrigin = origin;
    endpoint = origin;

    if dest == origin
        path = origin;

        return path
    end

    if pathDest[length(pathDest)] != pathOrigin[1]
        pair = [pathDest[length(pathDest)]; pathOrigin[1]];
        dest = maximum(pair);
        origin = minimum(pair);
        subpath = getSubPath(origin, dest, par);

        if subpath[1] == pathDest[length(pathDest)] && subpath[length(subpath)] == pathOrigin[1]
            pathDest = [pathDest; subpath[2:length(subpath)]];
        elseif subpath[1] == pathDest[length(pathDest)] && subpath[length(subpath)] != pathOrigin[1]
            pathDest = [pathDest; subpath[2:length(subpath)]];
        elseif subpath[1] == pathOrigin[1] && subpath[length(subpath)] == pathDest[length(pathDest)]
            pathDest = [pathDest; reverse(subpath[1:length(subpath)-1])];
        elseif subpath[1] == pathOrigin[1] && subpath[length(subpath)] != pathDest[length(pathDest)]
            pathOrigin = [reverse(subpath[2:length(subpath)]); pathOrigin];
        elseif subpath[length(subpath)] == pathDest[length(pathDest)]
            pathDest = [pathDest; reverse(subpath[1:lenth(subpath)-1])];
        elseif subpath[length(subpath)] == pathOrigin[1]
            pathOrigin = [reverse(subpath[1:length(subpath)-1]); pathOrigin];
        end
    end

    if length(pathDest) == 1
        path = [pathDest; pathOrigin];
    else
        path = [pathDest[1:length(pathDest)-1]; pathOrigin];
    end

    path
end

@everywhere function getSubPath(origin, dest, par)
    path = dest; parent = dest;
    while parent > origin
        if parent > origin
            parent = par[parent];
            path = [path; parent];
        end
    end

    path
end

@everywhere function getChildrenMatrix(par)
    N = length(par);
    CHILD = zeros(Int64,N,N);
    for node=1:N
        parent = par[node];
        if parent != 0
            CHILD[parent, node] = 1;
        end
    end

    CHILD
end

@everywhere function getSuccessorMatrix()
    SuccMat = zeros(Int64,N,N);
    for i = 1:N
        cur = i;
        while cur != 0
            SuccMat[cur, i] = 1;
            cur = par[cur];
        end
    end

    SuccMat
end

@everywhere function getCommonPathVariables(N,par,r,x)
    # println("r, x: ", r, "\n",x)
    R = zeros(Float64,N,N); X = zeros(Float64,N,N); commonPath = zeros(Int64,N,N);
    for i=1:N
        for j=1:N
            i1 = i;
            j1 = j;
            lca = 0;
            #       println("going in ",i, " ", j);
            while lca==0 # least common ancestor not found
                #         println(i1, " ", j1);
                while j1!=0 # j1 has not reached the root
                    if j1 == i1 # if a common ancestor found, note and break
                        lca = j1;
                        break;
                    end
                    j1 = par[j1]; # go up path of j
                end

                if lca!=0
                    break;
                end
                i1 = par[i1]; j1 = j;
                if i1 == 0 && j1==0
                    println("Error in the tree. Check variable par.");
                    break;
                end
            end
            if lca!=0
                k = lca;
                #         println("lca ", lca);
                while k!=0 #populate values of R[i,j] and X[i,j]
                    commonPath[i,j] += 1;
                    R[i,j] += r[k]; X[i,j] += x[k];
                    k = par[k];
                end
            end
        end
    end
    commonPath, R, X
end

## ------------------------------------##
## SET UP MASTER PROBLEM (MP)
@everywhere function getMasterModel(G, Ns, kmv)
    mm = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0));

    @variable(mm, xvar[1:length(SITE),1:length(RES)],Bin);
    @variable(mm, xsvar[1:length(SITE)],Bin);
    @variable(mm, theta[1:Ns]);
    @variable(mm, kcm[1:length(LOADS),1:Ns], Bin);

    x  = zeros(AffExpr,N,Nres); x[SITE,RES] = xvar;
    xs  = zeros(AffExpr,N); xs[SITE] = xsvar;

    @constraint(mm, sum(x[i,j] for i in SITE, j in RES) <= G);

    # 1 DG is allocated to at most 1 site
    @constraint(mm, [j in RES], sum(x[i,j] for i in SITE) <= 1);

    # if a site is operated, at least 1 DG is allocated
    @constraint(mm, [i in SITE], xs[i] <= sum(x[i,j] for j in RES));

    # if a DG is operated at a node, a site is operated
    @constraint(mm, [i in SITE], sum(x[i,j] for j in RES) <= Nres * xs[i]);

    @constraint(mm, [i in SITE, j in RES], x[i,j] <= xs[i]);

    # No mobile DGs allocated
    # @constraint(mm, [j in RESm], sum(x[i,j] for i in SITE) == 0);

    @constraint(mm, [i in 1:Ns], theta[i] >= 0);

    costDER = 0;
    for i = 1:Nres
        costDER += sum(WDER[SITE] .* xvar[:,i]);
    end

    @objective(mm, Min, sum(WSITE[SITE].*xsvar) + costDER + sum(theta)/Ns);

    mm, x, xs, theta, kcm
end

## ------------------------------------##
## GET STAGE II SUBPROBLEM (SP2)
@everywhere function getSlaveModel(kmv, G, Y, Ns, xvMP, xsvMP, LP)

    sm = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0));
    # sm = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0, MIPGap = 1e-10));
    sm, p, q, pg, qg, beta, P, Q, v, v0, vpar, t = getSlaveVariables(sm, Ns, T);

    y = zeros(AffExpr,N,Ns,T);
    for i = 1:Ns
        nFailed = sum(kmv[:,i]);
        if LP == 0
            y[kmv[:,i] .== 1, i, 1:T] = @variable(sm, [1:nFailed,1,1:T], Bin);
        elseif LP == 1
            y[kmv[:,i] .== 1, i, 1:T] = @variable(sm, [1:nFailed,1,1:T]);
            @constraint(sm, y[kmv[:,i] .== 1, i, 1:T] .>= 0)
        end
    end

    if LP == 0
        @variable(sm, kcvar[1:length(LOADS),1:Ns,1:T], Bin);
    elseif LP == 1
        @variable(sm, kcvar[1:length(LOADS),1:Ns,1:T]); # kc
        @constraint(sm, kcvar .>= 0);
    end
    kcv = zeros(AffExpr,N,Ns,T); kcv[LOADS,:,:,] = kcvar;

    sm, xs, x, xr = getMobileVariables(sm, Ns, LP, T);

    basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs = addGenPlaceConstraintsSAA(sm, p, q, pg, qg, beta, P, Q, v, v0, t, vpar, kcv, kmv, y, costGen, G, Y, 1, x);

    sm, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum = getMobileConstraints(sm, xs, x, xr, xsvMP, xvMP, Ns, 0);

    addObjectiveSlaveModel(sm, t, beta, kcv, P, Ns, xs, xr);

    sm, p, q, pg, qg, beta, P, Q, v, t, kcv, v0, y, vpar, xs, x, xr, basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum
end

## Get variables for Stage II subproblem
@everywhere function getSlaveVariables(sm, Ns, T)
    @variable(sm, p[1:N,1:Ns,1:T]); @variable(sm, q[1:N,1:Ns,1:T]);
    @variable(sm, pgvar[1:length(SITE),1:length(RES),1:Ns,1:T]);
    @variable(sm, qgvar[1:length(SITE),1:length(RES),1:Ns,1:T]);
    @variable(sm, betavar[1:length(LOADS),1:Ns,1:T]);
    @variable(sm, v[1:N,1:Ns,1:T]); @variable(sm, v0);
    @variable(sm, P[1:N,1:Ns,1:T]); @variable(sm, Q[1:N,1:Ns,1:T]);
    @variable(sm, tvar[1:N,1:Ns,1:T]);

    beta = zeros(AffExpr,N,Ns,T); t = zeros(AffExpr,N,Ns,T);
    pg = zeros(AffExpr,N,Nres,Ns,T); qg = zeros(AffExpr,N,Nres,Ns,T);
    beta[LOADS,1:Ns,1:T] = betavar; vpar = zeros(AffExpr,N,Ns,T);
    t = tvar;
    pg[SITE,:,:,:] = pgvar; qg[SITE,:,:,:] = qgvar;

    for i = 1:N
        for j = 1:Ns
            for k = 1:T
                vpar[i,j,k] = par[i] == 0 ? v0 : v[par[i],j,k];
            end
        end
    end

    sm, p, q, pg, qg, beta, P, Q, v, v0, vpar, t
end

@everywhere function getMobileVariables(sm, Ns, LP, T)
    if LP == 0
        @variable(sm, xsvar[1:length(SITE),1:Ns,1:T], Bin);
        @variable(sm, xvar[1:length(SITE),1:length(RES),1:Ns,1:T], Bin);
        @variable(sm, xrvar[1:length(SITE),1:length(SITE),1:length(RES),1:Ns,1:T], Bin);
    else
        @variable(sm, xsvar[1:length(SITE),1:Ns,1:T]);
        @variable(sm, xvar[1:length(SITE),1:length(RES),1:Ns,1:T]);
        @variable(sm, xrvar[1:length(SITE),1:length(SITE),1:length(RES),1:Ns,1:T]);
    end

    xs  = zeros(AffExpr,N,Ns,T); xs[SITE,1:Ns,1:T] = xsvar;
    x  = zeros(AffExpr,N,Nres,Ns,T); x[SITE,RES,1:Ns,1:T] = xvar;
    xr = zeros(AffExpr,N,N,Nres,Ns,T); xr[SITE,SITE,RES,1:Ns,1:T] = xrvar;

    sm, xs, x, xr
end

@everywhere function getMobileConstraints(sm, xs, x, xr, xsvMP, xvMP, Ns, opt)

    # Ts = length(setInc) + length(setExc);

    @constraint(sm, siteDev0[i = 1:length(SITE), j = 1:Ns], xs[SITE[i], j, 1] == xsvMP[SITE[i]]);
    @constraint(sm, siteDev[i = 1:length(SITE), j = 1:Ns, k = 2:T], xs[SITE[i], j, k-1] <= xs[SITE[i], j, k])

    @constraint(sm, reallocFixed[i = 1:length(SITE), j = 1:length(RESf), k = 1:Ns, l = 1:T], x[SITE[i], RESf[j], k, l] == xvMP[SITE[i], RESf[j]]);
    @constraint(sm, reallocMobile[i = 1:length(SITE), j = 1:length(RESm), k = 1:Ns], x[SITE[i], RESm[j], k, 1] == xvMP[SITE[i], RESm[j]]);
    @constraint(sm, reallocSum[i = 1:length(SITE), j = 1:length(RESm), k = 1:Ns, l = 2:T], sum(xr[SITE[i], :, RESm[j], k, l]) == x[SITE[i], RESm[j], k, l]);

    @constraint(sm, alloc[i = 1:length(SITE), j = 1:length(RES), k = 1:Ns, l = 1:T], x[SITE[i], RES[j], k, l] <= xs[SITE[i], k, l]);
    @constraint(sm, allocMobile[i = 1:length(RESm), j = 1:Ns, k = 2:T], sum(x[:,RESm[i], j, k]) >= sum(x[:, RESm[i], j, k-1]));
    @constraint(sm, derOnce[i = 1:length(RES), j = 1:Ns, k = 1:T], sum(x[:, RES[i], j, k]) <= 1);

    @constraint(sm, derSum[i = 1:Ns, j = 1:T], sum(x[SITE, RES, i, j]) <= G);

    sm, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum
end

## Get objective function
@everywhere function addObjectiveSlaveModel(sm, t, beta, kcval, P, Ns, xs, xr)

    costP = sum(sum(WAC) * P[1,1:Ns,1:T]);
    costT = sum(WVR * sum(t[:,1:Ns,1:T])/N);

    costLC = 0; costLS = 0; costP = 0; costSD = 0; costR = 0;
    for i = 1:T
        costLC += sum(WLC[LOADS] .* (ones(Float64,length(LOADS),1,1)-beta[LOADS,1,i]) );
        costLS += sum((WSD-WLC).*kcval[:,1,i]);
        if i > 1
            costSD += sum(WSITE_2[SITE] .* (xs[SITE,1,i] - xs[SITE,1,i-1]));
            if length(RESm) >= 1
                for j in SITE
                    idx = setdiff(SITE, j);
                    costR += sum(WR[j,idx,RESm] .* xr[j,idx,RESm,1,i]);
                end
            end
        end
    end

    # println(costR)

    @objective(sm, Min, costP + costT + costLC + costLS + costSD + costR);

    # @objective(sm, Min, sum(sum(WAC) * P[1,1,i] + WVR * sum(t[:,1,i])/N + sum(WLC[LOADS].*(ones(Float64,length(LOADS),1,1)-beta[LOADS,1,i])) + sum((WSD-WLC).*kcval[:,1,i]) for i=1:T) );
    # ) for i=1:T);
end

@everywhere function addObjectiveSlaveModelGreedy(smg, t, beta, kcval, P, xs, xr, xsv)

    costP = sum(sum(WAC) * P[1,1,1]);
    costT = sum(WVR * sum(t[:,1,1])/N);
    costLC = sum(WLC[LOADS] .* (ones(Float64,length(LOADS),1,1)-beta[LOADS,1,1]) );
    costLS = sum((WSD-WLC).*kcval[:,1,1]);
    costSD = sum(WSITE_2[SITE] .* (xs[SITE,1,1] - xsv[SITE,1,1]));

    costR = 0;
    if length(RESm) >= 0
        for j in SITE
            idx = setdiff(SITE, j);
            costR += sum(WR[j,idx,RESm] .* xr[j,idx,RESm,1,1]);
        end
    end

    # println(costR)

    @objective(smg, Min, costP + costT + costLC + costLS + costSD + costR);

    smg
end

## Add all constraints to the constructed Stage II subpfoblem (SP2)
@everywhere function addGenPlaceConstraintsSAA(sm, p, q, pg, qg, beta, P, Q, v, v0, t, vpar, kcv, kmv, y, costGen, G, Y, Ns, x)

    repairIneqs = []; repairIneqbs = []; repairEqs = []; repairEqbs = [];
    for i = 1:Ns
        sm, ineq, ineqb, eq, eqb = addRepairConstraints(sm, kmv[:,i], y[:,i,:], Y);
        repairIneqs = [repairIneqs; ineq];
        repairIneqbs = [repairIneqbs; ineqb];
        repairEqs = [repairEqs; eq];
        repairEqbs = [repairEqbs; eqb];
    end

    sm, genIneq, genIneqb = addGenInequalities(sm, pg, qg, x, T);
    sm, droopIneq, droopIneqb = addDroopInequalities(sm, v, qg, 1:1:T, x, T);
    sm, basicIneq, basicIneqb = addBasicInequalities(sm, beta, kcv, v, t, T);
    sm, discIneq, discIneqb = addDisconnectivityConstraints(sm, kcv, v, T);
    sm, failureIneqs, failureIneqbs = addFailureInequalities(sm, P, Q, vpar, v, kmv, y, T);
    sm, basicEq, basicEqb = addBasicEqualities(sm, v, vpar, p, q, pg, qg, beta, P, Q, T);

    Eqs = basicEq;
    Eqbs = basicEqb;

    basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs
end

# Get constraints for the multi-stage repair model
@everywhere function addRepairConstraints(sm, kmv, y, Y)

    nFailed = sum(kmv);
    noPeriods = trunc(Int,ceil((nFailed-1)/Y));

    Yk = zeros(Float64,T); Yk[1] = 0; Yk[T] = 1;
    for i = 1:noPeriods
        Yk[i+1] = minimum([Y; nFailed - 1 - Y*(i-1)]);
    end

    repairIneq = []; repairEq = [];
    repairIneqb = []; repairEqb = [];
    for ts = 1:T
        repairStep = @constraint(sm, sum(y[:,ts]) == Yk[ts]);  #crew budget
        repairIneq = [repairIneq; repairStep];
        repairIneqb = [repairIneqb; Yk[ts]];
    end

    for e = 1:N
        if kmv[e] == 1
            repairEdge = @constraint(sm, sum(y[e,:]) == 1);
            repairIneq = [repairIneq; repairEdge];
            repairIneqb = [repairIneqb; kmv[e,1]];
        end
    end

    # repairLim = @constraint(sm, sum(y[1,:]) <= 1);
    # repairIneq = [repairIneq; repairLim];
    # repairIneqb = [repairIneqb; 1];

    # repairInit = @constraint(sm, y[kmv .== 1, 1] .== 0); # initial timestep = 0
    repairSub = @constraint(sm, y[1,T] == 1); # substation is connected only on the last timestep

    # for e = 2:N
    #     if kmv[e] == 1
    #         repairNotSub = @constraint(sm, y[e,T] == 0);
    #     end
    # end

    # repairEq = [repairInit; repairSub];
    # repairEqb = [ov[kmv .== 1]; 1];
    repairEq = repairSub; repairEqb = 1

    sm, repairIneq, repairIneqb, repairEq, repairEqb
end

# Get constraints for power generation from distributed generators (DGs)
@everywhere function addGenInequalities(sm, pg, qg, x, T)

    @constraint(sm, pglb[i = 1:length(SITE), j = 1:length(RES), k = 1:T], pg[SITE[i],RES[j],1,k] >= 0);
    @constraint(sm, qglb[i = 1:length(SITE), j = 1:length(RES), k = 1:T], qg[SITE[i],RES[j],1,k] >= 0);
    @constraint(sm, pgub[i = 1:length(SITE), j = 1:length(RES), k = 1:T], pg[SITE[i],RES[j],1,k] <= x[SITE[i],RES[j],1,k] * pgmax[SITE[i]]);
    @constraint(sm, qgub[i = 1:length(SITE), j = 1:length(RES), k = 1:T], qg[SITE[i],RES[j],1,k] <= x[SITE[i],RES[j],1,k] * qgmax[SITE[i]]);
    genIneqs=[pglb; qglb; pgub; qgub];

    # ovGen = repeat(transpose(ov[RES]),length(SITE));
    # pgBase = repeat(pgmax[SITE],1,length(RES));
    # qgBase = repeat(qgmax[SITE],1,length(RES));
    # zeroB = zeros(AffExpr,length(SITE)*2,length(RES),T);
    # pgB = zeros(AffExpr,length(SITE),length(RES),T);
    # qgB = zeros(AffExpr,length(SITE),length(RES),T);
    # for i = 1:T
    #     zeroB[:,:,i] = repeat(ovGen,2,1);
    #     pgB[:,:,i] = pgBase;
    #     qgB[:,:,i] = qgBase;
    # end
    # genIneqbs = [zeroB; pgB; qgB];

    genIneqbs = zeros(Float64,length(SITE)*4,length(RES),T);

    sm, genIneqs, genIneqbs
end

# Get constraints for voltage droop control in islanded microgrids
@everywhere function addDroopInequalities(sm, v, qg, ts, x, T)

    @constraint(sm, voltMGUB[i = 1:length(SITE), j = 1:length(RES), k = 1:T-1], v[SITE[i],1,k] + mq * qg[SITE[i],RES[j],1,k] <= vref + 2 * (1 - x[SITE[i],RES[j],1,k]) + mq * qgnom[RES[j]]);
    @constraint(sm, voltMGLB[i = 1:length(SITE), j = 1:length(RES), k = 1:T-1], v[SITE[i],1,k] + mq * qg[SITE[i],RES[j],1,k] >= vref - 2 * (1 - x[SITE[i],RES[j],1,k]) + mq * qgnom[RES[j]]);

    droopIneqs = [voltMGUB; voltMGLB];

    bBaseLB = zeros(AffExpr,length(SITE),length(RES),T-1);
    bBaseUB = zeros(AffExpr,length(SITE),length(RES),T-1);
    for i = 1:T-1
        bBaseLB[:,:,i] = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) + 2 * ones(Float64, length(SITE), length(RES));
        bBaseUB[:,:,i] = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) - 2 * ones(Float64, length(SITE), length(RES));
    end

    droopIneqbs = [bBaseLB; bBaseUB];

    sm, droopIneqs, droopIneqbs
end

## Get constraints for load control
@everywhere function addBasicInequalities(sm, beta, kcval, v, t, T)
    @constraint(sm, betalb[i = 1:length(LOADS), j = 1:T], beta[LOADS[i],1,j] >= (1-kcval[LOADS[i],1,j]) * betamin[LOADS[i]]);
    @constraint(sm, betaub[i = 1:length(LOADS), j = 1:T], beta[LOADS[i],1,j] <= (1-kcval[LOADS[i],1,j]));

    @constraint(sm, lovrMin[i=1:N,j=1:T], t[i,1,j] + v[i,1,j] >= v0nom * uv[i]);
    @constraint(sm, lovrMax[i=1:N,j=1:T], t[i,1,j] - v[i,1,j] >= -v0nom * uv[i]);

    basicIneqs = [betalb; betaub; lovrMin; lovrMax];

    basicIneqbs = repeat(betamin[LOADS], 1, T);
    basicIneqbs = [basicIneqbs; ones(Float64, length(LOADS), T)];
    basicIneqbs = [basicIneqbs; repeat(v0nom * uv, 1, T); repeat(-v0nom * uv, 1, T)];

    sm, basicIneqs, basicIneqbs
end

## Get constraints for load shedding
@everywhere function addDisconnectivityConstraints(sm, kcval, v, T)

    @constraint(sm, loadLow[i=1:length(LOADS),j=1:T], v[LOADS[i],1,j]  >= vcmin[LOADS[i]] - kcval[LOADS[i],1,j]);
    @constraint(sm, loadUp[i=1:length(LOADS),j=1:T], -v[LOADS[i],1,j]  >=  -vcmax[LOADS[i]] - kcval[LOADS[i],1,j]);

    discIneqs = [loadLow; loadUp];

    discIneqbs = [repeat(vcmin[LOADS],1,T); repeat(-vcmax[LOADS],1,T)];

    sm, discIneqs, discIneqbs
end

## Get inequality constraints for the LinDistFlow power flow model
@everywhere function addFailureInequalities(sm, P, Q, vpar, v, kmval, y, T)
    failureIneqs = []; failureIneqbs = [];

    @constraint(sm, pdislb[i = 1:N, j = 1:T], P[i,1,j] >= -M*(1-kmval[i]+sum(y[:,1,1:j],dims=2)[i]));
    @constraint(sm, qdislb[i = 1:N, j = 1:T], Q[i,1,j] >= -M*(1-kmval[i]+sum(y[:,1,1:j],dims=2)[i]));
    @constraint(sm, pdisub[i = 1:N, j = 1:T], P[i,1,j] <= M*(1-kmval[i]+sum(y[:,1,1:j],dims=2)[i]));
    @constraint(sm, qdisub[i = 1:N, j = 1:T], Q[i,1,j] <= M*(1-kmval[i]+sum(y[:,1,1:j],dims=2)[i]));

    failureIneqs = [pdislb; qdislb; pdisub; qdisub];

    ySum = zeros(AffExpr,N,T);
    for i = 1:T
        ySum[:,i] = sum(y[:,1,1:i],dims=2);
    end
    bBase = M*(ones(Float64,N,T) - repeat(kmval,1,T));
    failureIneqbs = [-bBase; -bBase; bBase; bBase];

    @constraint(sm, vlb[i = 1:length(noLTC), j = 1:T], vpar[noLTC[i],1,j] - v[noLTC[i],1,j] - resistance[noLTC[i]]*P[noLTC[i],1,j] - reactance[noLTC[i]]*Q[noLTC[i],1,j] >= -M * (kmval[noLTC[i]]-sum(y[noLTC[i],1,1:j])));
    @constraint(sm, vub[i=1:length(noLTC), j = 1:T], vpar[noLTC[i],1,j] - v[noLTC[i],1,j] - resistance[noLTC[i]]*P[noLTC[i],1,j] - reactance[noLTC[i]]*Q[noLTC[i],1,j] <= M * (kmval[noLTC[i]] - sum(y[i,1,1:j])));

    failureIneqs = [failureIneqs; vlb; vub];
    # bBase = zeros(Float64, N, T);
    bBase = M*repeat(kmval,1,T);
    failureIneqbs = [failureIneqbs; -bBase; bBase];

    sm, failureIneqs, failureIneqbs
end

## Get equality constraints for the LinDistFlow power flow model
@everywhere function addBasicEqualities(sm, v, vpar, p, q, pg, qg, beta, P, Q, T)
    basicEq =[]; basicEqb = [];

    @constraint(sm, realFlow[i in 1:N, j = 1:T], P[i,1,j] == sum(CHILD[i,:] .* P[:,1,j]) + p[i,1,j]);
    @constraint(sm, reacFlow[i in 1:N, j = 1:T], Q[i,1,j] == sum(CHILD[i,:] .* Q[:,1,j]) + q[i,1,j]);
    @constraint(sm, realCons[i in 1:N, j = 1:T], p[i,1,j] - beta[i,1,j]*pcmax[i] + sum(pg[i,:,1,j]) == 0);
    @constraint(sm, reacCons[i in 1:N, j = 1:T], q[i,1,j] - beta[i,1,j]*qcmax[i] + sum(qg[i,:,1,j]) == 0);

    basicEq = [realFlow; reacFlow; realCons; reacCons];
    basicEqb = [repeat(ov,1,T); repeat(ov,1,T); repeat(ov,1,T); repeat(ov,1,T)];

    if length(LTC) > 0
        @constraint(sm, voltDropLTC[i in LTC, j = 1:T], v[i,1,j] - LTCSetting * vpar[i,j] == 0);
        basicEq = [basicEq; voltDropLTC];
        basicEqb = [basicEqb; repeat(ov[LTC],1,T)];
    end

    sm, basicEq, basicEqb
end

## ------------------------------------##
## GET BENDERS CUT

# Benders cut requires: dual solution, dual cost vector, reduced costs, subproblem discrete variable values
@everywhere function getBendersCutComponents(kmv, xvMP, basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs, xsvMP, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum, optMobile)

    noFailures = sum(kmv); noPeriods = trunc(Int, ceil((noFailures-1)/Y));

    # Get dual variable values and dual cost vectors
    ineqbar, Ineqbs, failureIneqbar, basicIneqbar, discIneqbar, droopIneqbar, genIneqbar, eqbar = getDualVariables(basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, genIneq, Eqs, repairIneqs);
    siteDev0_bar, siteDevBar, reallocFixedBar, reallocMobileBar, allocbar, derOncebar, derSumbar = getDualVariablesRealloc(siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum);
    repairIneqbar, repairIneqPeriod, repairIneqLine = getRepairIneqBar(repairIneqs, noFailures);
    repairEqbar, repairEqSubstation = getRepairEqbar(repairEqs, noFailures);

    # Get Benders cut (constant term)
    bendersCutLS = sum(WLC[LOADS])*T + sum(ineqbar .* Ineqbs) + sum(eqbar .* Eqbs) + sum(repairIneqbar .* repairIneqbs) + sum(repairEqbar .* repairEqbs);
    bendersCutLS += sum(droopIneqbar .* droopIneqb) + sum(genIneqbar .* genIneqb);
    bendersCutLS += sum(derOncebar) + G*sum(derSumbar);

    # println(bendersCutLS + sum(xsvMP[SITE] .* siteDev0_bar) + sum(derOncebar) + G*sum(derSumbar) + sum(xvMP[SITE,RESm] .* reallocMobileBar))

    # Get reduced costs
    rc_yl, rc_kc = getReducedCosts(repairIneqPeriod, repairIneqLine, repairEqSubstation, failureIneqbar, basicIneqbar, discIneqbar, kmv);
    rc_sd = getReducedCostsSite(siteDev0_bar, siteDevBar, allocbar);
    rc_der = getReducedCostsDER(reallocFixedBar, allocbar, derOncebar, derSumbar, genIneqbar, droopIneqbar);
    # println(rc_sd)
    # println(rc_der)
    # rc_sd, rc_der = getReducedCostsDER(siteDev0_bar, reallocMobileBar, reallocFixedBar, derOncebar, derSumbar, xsvMP, xvMP);

    # Get candidate feasible repair solution (yInput)
    yInput, objRcModel = getRC_Model(kmv, rc_kc, rc_yl, rc_sd, rc_der, xvMP, xsvMP, optMobile);

    # Get connectivity between loads / DG sites at all periods
    ac = zeros(Int64, length(LOADS), length(SITE), T);
    ac[:,:,1:noPeriods+1] = getConnectivitySet(kmv, yInput, noFailures);
    ac[:,:,noPeriods+2:T] = repeat(ac[:,:,noPeriods+1],1,1,T-noPeriods-1);

    bendersCutLS, rc_kc, rc_sd, rc_der, ac, objRcModel, siteDev0_bar, reallocFixedBar, reallocMobileBar#droopIneqbar, genIneqbar, rc_kc, ac, objRcModel
end

@everywhere function getDualVariablesRealloc(siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum)

    sizeIneqs = size(siteDev0);
    siteDev0_bar = zeros(Float64, length(SITE));
    siteDevBar = zeros(Float64, length(SITE), T-1);
    reallocFixedBar = zeros(Float64, length(SITE), length(RESf), T);
    reallocMobileBar = zeros(Float64, length(SITE), length(RESm));
    allocBar = zeros(Float64, length(SITE), length(RES), T);
    derOncebar = zeros(Float64, length(RES), T);
    derSumbar = zeros(Float64, T);


    for i = 1:length(SITE)
        siteDev0_bar[i] = JuMP.dual(siteDev0[i,1]);
        for j = 2:T
            # println(siteDev)
            # println(JuMP.dual(siteDev[i,1,j-1]))
            siteDevBar[i,j-1] = JuMP.dual(siteDev[i,1,j]);
        end
    end

    for i = 1:length(SITE)
        for j = 1:length(RES)
            for k = 1:T
                allocBar[i,j,k] = JuMP.dual(alloc[i,j,1,k]);
            end
        end

        for j = 1:length(RESf)
            for k = 1:T
                reallocFixedBar[i,j,k] = JuMP.dual(reallocFixed[i,j,1,k]);
            end
        end

        for j = 1:length(RESm)
            reallocMobileBar[i,j] = JuMP.dual(reallocMobile[i,j,1,1]);
        end
    end

    for i = 1:length(RES)
        for j = 1:T
            derOncebar[i,j] = JuMP.dual(derOnce[i,1,j]);
        end
    end

    for i = 1:T
        derSumbar[i] = JuMP.dual(derSum[1,i]);
    end

    siteDev0_bar, siteDevBar, reallocFixedBar, reallocMobileBar, allocBar, derOncebar, derSumbar
end

## ------------------------------------##
## Get dual cost vectors for droop control and DG generation constraints.
# Unlike other constraints, these constraints contain Master Problem variables.
# Remaining dual cost vectors are obtained from construction of subproblem constraints.

## Dual cost vector for voltage droop control constraints
@everywhere function addBendersCutDroop(x)
    bBaseLB = zeros(AffExpr,length(SITE),length(RES),T-1);
    bBaseUB = zeros(AffExpr,length(SITE),length(RES),T-1);
    for i = 1:T-1
        bBaseLB[:,:,i] = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) + 2 * (ones(length(SITE),length(RES)) - x[SITE,RES]);
        bBaseUB[:,:,i] = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) - 2 * (ones(length(SITE),length(RES)) - x[SITE,RES]);
    end

    droopIneqbs = [bBaseLB; bBaseUB];

    droopIneqbs
end

## Dual cost vector for DG generation constraints
@everywhere function addBendersCutGen(x)
    ovGen = repeat(transpose(ov[RES]),length(SITE));
    pgBase = repeat(pgmax[SITE],1,length(RES));
    qgBase = repeat(qgmax[SITE],1,length(RES));
    zeroB = zeros(AffExpr,length(SITE)*2,length(RES),T);
    pgB = zeros(AffExpr,length(SITE),length(RES),T);
    qgB = zeros(AffExpr,length(SITE),length(RES),T);
    for i = 1:T
        zeroB[:,:,i] = repeat(ovGen,2,1);
        pgB[:,:,i] = x[SITE,RES] .* pgBase;
        qgB[:,:,i] = x[SITE,RES] .* qgBase;
    end
    genIneqbs = [zeroB; pgB; qgB];

    genIneqbs
end

## ------------------------------------##
## Get values of dual variables corresponding to Stage II subproblem (SP2)

## Get all dual variables corresponding to Stage II subproblem (SP2)
@everywhere function getDualVariables(basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, genIneq, Eqs, repairIneqs)

    ineqbar, Ineqbs = getIneqBar(basicIneq, discIneq, failureIneqs, basicIneqb, discIneqb, failureIneqbs);
    failureIneqbar = getFailureIneqBar(failureIneqs);
    basicIneqbar, discIneqbar = getBasicDiscIneqBar(basicIneq, discIneq);
    droopIneqbar = getDroopIneqBar(droopIneq);
    genIneqbar = getGenineqBar(genIneq);
    eqbar = getEqBar(Eqs);

    ineqbar, Ineqbs, failureIneqbar, basicIneqbar, discIneqbar, droopIneqbar, genIneqbar, eqbar
end

## Dual variables: load control, load shedding, and power flow inequalities
@everywhere function getIneqBar(basicIneq, discIneq, failureIneqs, basicIneqb, discIneqb, failureIneqbs);

    Ineqs = [basicIneq; discIneq; failureIneqs];
    Ineqbs = [basicIneqb; discIneqb; failureIneqbs];
    sizeIneqs = size(Ineqs);
    ineqbar = zeros(Float64, sizeIneqs);
    for i = 1:sizeIneqs[1]
        for j = 1:sizeIneqs[2]
            ineqbar[i,j] = JuMP.dual(Ineqs[i,j]);
        end
    end

    ineqbar, Ineqbs
end

## Dual variables: power flow inequalities
@everywhere function getFailureIneqBar(failureIneqs)

    sizeIneqs = size(failureIneqs);
    failureIneqbar = zeros(Float64, sizeIneqs);
    for i = 1:sizeIneqs[1]
        for j = 1:sizeIneqs[2]
            failureIneqbar[i,j] = JuMP.dual(failureIneqs[i,j]);
        end
    end

    failureIneqbar
end

## Dual variables: load control, load shedding
@everywhere function getBasicDiscIneqBar(basicIneq, discIneq)

    basicIneqbar = zeros(Float64, 6*length(LOADS), T);
    discIneqbar = zeros(Float64, 2*length(LOADS), T);
    for i = 1:2*length(LOADS)
        for j = 1:T
            basicIneqbar[i,j] = JuMP.dual(basicIneq[i,j]);
            discIneqbar[i,j] = JuMP.dual(discIneq[i,j]);
        end
    end

    basicIneqbar, discIneqbar
end

## Dual variables: voltage droop control
@everywhere function getDroopIneqBar(droopIneq)

    droopIneqbar = zeros(Float64, size(droopIneq));
    sizeDroopIneq = size(droopIneq);
    for i = 1:sizeDroopIneq[1]
        for j = 1:sizeDroopIneq[2]
            for k = 1:sizeDroopIneq[3]
                droopIneqbar[i,j,k] = JuMP.dual(droopIneq[i,j,k]);
            end
        end
    end

    droopIneqbar
end

## Dual variables: DG power generation
@everywhere function getGenineqBar(genIneq)

    genIneqbar = zeros(Float64, size(genIneq));
    sizeGenIneq = size(genIneq);
    for i = 1:sizeGenIneq[1]
        for j = 1:sizeGenIneq[2]
            for k = 1:sizeGenIneq[3]
                genIneqbar[i,j,k] = JuMP.dual(genIneq[i,j,k]);
            end
        end
    end

    genIneqbar
end

## Dual variables: power flow equalities
@everywhere function getEqBar(Eqs)

    eqbar = zeros(Float64, size(Eqs));
    sizeEqs = size(Eqs);
    for i = 1:sizeEqs[1]
        for j = 1:sizeEqs[2]
            eqbar[i,j] = JuMP.dual(Eqs[i,j]);
        end
    end

    eqbar
end

## Dual variables: repair model inequalities
@everywhere function getRepairIneqBar(repairIneqs, noFailures)

    repairIneqbar = zeros(Float64, size(repairIneqs));
    sizeRepairIneqs = size(repairIneqs);
    for i = 1:sizeRepairIneqs[1]
        repairIneqbar[i] = JuMP.dual(repairIneqs[i]);
    end

    repairIneqPeriod = repairIneqbar[1:T];
    repairIneqLine = repairIneqbar[T+1:T+noFailures];
    # repairIneqSubstation = repairIneqbar[length(repairIneqbar)];

    repairIneqbar, repairIneqPeriod, repairIneqLine
end

## Dual variables: repair model equalities
@everywhere function getRepairEqbar(repairEqs, noFailures)

    repairEqbar = zeros(Float64, size(repairEqs));
    sizeRepairEqs = size(repairEqs);
    for i = 1:sizeRepairEqs[1]
        repairEqbar[i] = JuMP.dual(repairEqs[i]);
    end
    # repairEqFirstPeriod = repairEqbar[1:noFailures];
    repairEqSubstation = repairEqbar[length(repairEqbar)];

    repairEqbar, repairEqSubstation
end

## ------------------------------------##
## Get reduced costs for solution to Stage II subproblem (SP2)
@everywhere function getReducedCostsSite(siteDev0_bar, siteDevBar, allocBar)
    rc_sd = zeros(Float64, length(SITE), T);

    for i = 1:T
        rc_sd[:,i] = WSITE_2[SITE];
    end

    for i = 1:length(SITE)
        rc_sd[i,1] -= (siteDev0_bar[i] - sum(allocBar[i,:,1]) - siteDevBar[i,1]);
        rc_sd[i,T] -= (-sum(allocBar[i,:,T]) + siteDevBar[i,T-1]);

        for j = 2:T-1
            rc_sd[i,j] -= (-sum(allocBar[i,:,j]) + siteDevBar[i,j-1] - siteDevBar[i,j]);
        end
    end

    rc_sd
end


@everywhere function getReducedCostsDER(reallocFixedBar, allocbar, derOncebar, derSumbar, genIneqbar, droopIneqbar)

    rc_der = zeros(Float64, length(SITE), length(RES), T);

    # println(reallocMobileBar)

    for i = 1:length(SITE)
        for j = 1:length(RES)
            for k = 1:T
                rc_der[i,j,k] -= (reallocFixedBar[i,j,k] + allocbar[i,j,k] + derOncebar[j,k] + derSumbar[k]);
                rc_der[i,j,k] += (pgmax[SITE[i]]*genIneqbar[2*length(SITE)+i,j,k] + qgmax[SITE[i]]*genIneqbar[3*length(SITE)+i,j,k]);
                if k < T
                    rc_der[i,j,k] -= 2*(droopIneqbar[i,j,k] + droopIneqbar[length(SITE)+i,j,k]);
                end
            end
        end
    end

    # println(rc_der)

    rc_der
end

@everywhere function getReducedCosts(repairIneqPeriod, repairIneqLine, repairEqSubstation, failureIneqbar, basicIneqbar, discIneqbar, km0)

    rc_yl = getReducedCostsRepairs(repairIneqPeriod, repairIneqLine, repairEqSubstation, failureIneqbar, km0);
    rc_kc = getReducedCostsLoads(basicIneqbar, discIneqbar);

    rc_yl, rc_kc
end

## Reduced costs: repair variables
@everywhere function getReducedCostsRepairs(repairIneqPeriod, repairIneqLine, repairEqSubstation, failureIneqbar, km0)

    noFailures = sum(km0);
    rc_yl = zeros(Float64, noFailures, T);

    for i = 1:noFailures
        for j = 1:T
            if i == 1 && j == 1
                rc_yl[i,j] = repairIneqLine[i] + repairIneqPeriod[j];# + repairEqFirstPeriod[i] + repairIneqSubstation;
                # rc_yl[i,j] = repairIneqLine[i] + repairEqFirstPeriod[i] + repairIneqSubstation;
            elseif i == 1 && j > 1
                rc_yl[i,j] = repairIneqPeriod[j] + repairIneqLine[i];# + repairIneqSubstation;
            elseif i > 1 && j == 1
                rc_yl[i,j] = repairIneqLine[i] + repairIneqPeriod[j];# + repairEqFirstPeriod[i];
            elseif i > 1 && j > 1
                rc_yl[i,j] = repairIneqPeriod[j] + repairIneqLine[i];
            end
        end
    end
    rc_yl[1,T] = rc_yl[1,T] + repairEqSubstation;

    idx = findall(km0 .== 1);

    failureIneq_pLb = failureIneqbar[1:N,:];
    failureIneq_qLb = failureIneqbar[N+1:2*N,:];
    failureIneq_pUb = failureIneqbar[2*N+1:3*N,:];
    failureIneq_qUb = failureIneqbar[3*N+1:4*N,:];
    failureIneq_vLb = failureIneqbar[4*N+1:5*N,:];
    failureIneq_vUb = failureIneqbar[5*N+1:6*N,:];
    for i = 1:noFailures
        for j = 1:T
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_pLb[idx[i][1],j:T]) + M*sum(failureIneq_pUb[idx[i][1],j:T]);
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_qLb[idx[i][1],j:T]) + M*sum(failureIneq_qUb[idx[i][1],j:T]);
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_vLb[idx[i][1],j:T]) + M*sum(failureIneq_vUb[idx[i][1],j:T]);
        end
    end
    rc_yl = -rc_yl;

    rc_yl
end

## Reduced costs: load shedding
@everywhere function getReducedCostsLoads(basicIneqbar, discIneqbar)

    rc_kc = repeat(WSD[LOADS]-WLC[LOADS], 1, T);
    basicIneqbar_lb = basicIneqbar[1:length(LOADS),:];
    basicIneqbar_ub = basicIneqbar[length(LOADS)+1:2*length(LOADS),:];
    discIneqbar_lb = discIneqbar[1:length(LOADS),:];
    discIneqbar_ub = discIneqbar[length(LOADS)+1:2*length(LOADS),:];

    rc_kc = rc_kc - (repeat(betamin[LOADS], 1, T).*basicIneqbar_lb + basicIneqbar_ub - discIneqbar_lb + discIneqbar_ub);
    rc_kc += 0.1*ones(Float64, length(LOADS), T);

    rc_kc
end

### ------------------------------------##
# Get candidate repair solution for subproblem
@everywhere function getRC_Model(km0, rc_kc, rc_yl, rc_sd, rc_der, xvMP, xsvMP, optMobile)

    # println("kc: ", rc_kc)
    # println("yl: ", rc_yl)
    # println("sd: ", rc_sd)
    # println("der: ", rc_der)

    # Get parameters, number of periods, and list of failed lines
    noFailures = sum(km0); noPeriods = trunc(Int,ceil((noFailures-1)/Y));

    idxFailPre = findall(km0 .== 1); idxFail = zeros(Int64, noFailures);
    for i = 1:noFailures
        idxFail[i] = idxFailPre[i][1];
    end

    # Calculate Yk
    Yk = zeros(Float64,T); Yk[1] = 0; Yk[T] = 1;
    for i = 1:noPeriods
        Yk[i+1] = minimum([Y; noFailures - 1 - Y*(i-1)]);
    end

    # Initialize model + variables
    rcModel = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0));

    # Get variables
    y = zeros(AffExpr,N,T); kc = zeros(AffExpr,N,T); kcc = zeros(AffExpr,N,N,T);
    xs  = zeros(AffExpr,N,T); x = zeros(AffExpr,N,Nres,T); xr = zeros(AffExpr,N,N,Nres,T);

    y[idxFail, 1:noPeriods+1] = @variable(rcModel, yvar[1:noFailures,1:noPeriods+1], Bin);
    kc[LOADS, 1:noPeriods+1] = @variable(rcModel, kcvar[1:length(LOADS),1:noPeriods+1]);
    kcc[LOADS, SITE, 1:noPeriods+1] = @variable(rcModel, kccvar[1:length(LOADS), 1:length(SITE), 1:noPeriods+1]);

    @constraint(rcModel, [i = 1:length(LOADS), j = 1:noPeriods+1], kc[LOADS[i], j] >= 0)
    @constraint(rcModel, [i = 1:length(LOADS), j = 1:length(SITE), k = 1:noPeriods+1], kcc[LOADS[i], SITE[j], k] >= 0)
    @constraint(rcModel, [i = 1:length(LOADS), j = 1:noPeriods+1], kc[LOADS[i], j] <= 1)
    @constraint(rcModel, [i = 1:length(LOADS), j = 1:length(SITE), k = 1:noPeriods+1], kcc[LOADS[i], SITE[j], k] <= 1)

    xs[SITE,1:noPeriods+1] = @variable(rcModel, xsvar[1:length(SITE),1:noPeriods+1], Bin);
    x[SITE,RES,1:noPeriods+1] = @variable(rcModel, xvar[1:length(SITE),1:length(RES),1:noPeriods+1], Bin);
    xr[SITE,SITE,RES,1:noPeriods+1] = @variable(rcModel, xrvar[1:length(SITE),1:length(SITE),1:length(RES),1:noPeriods+1], Bin);

    # Get objective
    if optMobile == 1
        @objective(rcModel, Min, sum(kcvar .* rc_kc[:,1:noPeriods+1]) + sum(yvar .* rc_yl[:,1:noPeriods+1]));# + sum(xs[SITE,:] .* rc_sd) + sum(x[SITE,RES,:] .* rc_der));
    elseif optMobile == 2
        @objective(rcModel, Min, sum(kcvar .* rc_kc[:,1:noPeriods+1]) + sum(yvar .* rc_yl[:,1:noPeriods+1]));# + sum(xs[SITE,1] .* rc_sd) + sum(x[SITE,RES,1:noPeriods+1] .* rc_der[:,:,1:noPeriods+1]));
    end

    # Get repair constraints
    for ts = 1:noPeriods+1
        periodBudget = @constraint(rcModel, sum(y[:,ts]) == Yk[ts]); #crew budget
    end
    for e = 2:length(idxFail)
        repairEdge = @constraint(rcModel, sum(yvar[e,:]) == 1);
    end

    # Get reallocation constraints
    @constraint(rcModel, siteDev0[i = 1:length(SITE)], xs[SITE[i], 1] == xsvMP[SITE[i]]);
    @constraint(rcModel, siteDev[i = 1:length(SITE), k = 2:noPeriods+1], xs[SITE[i], k-1] <= xs[SITE[i], k])

    @constraint(rcModel, reallocFixed[i = 1:length(SITE), j = 1:length(RESf), l = 1:noPeriods+1], x[SITE[i], RESf[j], l] == xvMP[SITE[i], RESf[j]]);
    @constraint(rcModel, reallocMobile[i = 1:length(SITE), j = 1:length(RESm)], x[SITE[i], RESm[j], 1] == xvMP[SITE[i], RESm[j]]);
    @constraint(rcModel, reallocSum[i = 1:length(SITE), j = 1:length(RESm), l = 2:noPeriods+1], sum(xr[SITE[i], :, RESm[j], l]) == x[SITE[i], RESm[j], l]);

    @constraint(rcModel, alloc[i = 1:length(SITE), j = 1:length(RES), l = 1:noPeriods+1], x[SITE[i], RES[j], l] <= xs[SITE[i], l]);
    @constraint(rcModel, allocMobile[i = 1:length(RESm), k = 2:noPeriods+1], sum(x[:,RESm[i], k]) >= sum(x[:, RESm[i], k-1]));
    @constraint(rcModel, derOnce[i = 1:length(RES), k = 1:noPeriods+1], sum(x[:, RES[i], k]) <= 1);

    @constraint(rcModel, derSum[j = 1:noPeriods+1], sum(x[SITE, RES, j]) <= G);

    # Connectivity constraints
    noFailures = sum(km0);

    @constraint(rcModel, loadShed[i = 1:length(LOADS), j = 1:noPeriods+1], kc[LOADS[i], j] >= 1 - sum(ones(Float64,length(SITE))-kcc[LOADS[i], SITE, j]));

    kl = 0
    for i = 1:noPeriods+1
        for j = 1:length(LOADS)
            for k = 1:length(SITE)
                path = getPathPair([LOADS[j]; SITE[k]], par);
                kl = getPathConnectivity(path, km0, y, i, 1);

                @constraint(rcModel, kcc[LOADS[j], SITE[k], i] >= 1-sum(x[SITE[k], :, i]));

                for l = 1:length(kl)
                    @constraint(rcModel, kcc[LOADS[j], SITE[k], i] >= kl[l]);
                end
            end
        end
    end

    # Solve mini-subproblem
    start = time();
    optimize!(rcModel); elapsed = time() - start;
    println("Time elapsed for mini-subproblem: ", elapsed)
    rcModelStatus = termination_status(rcModel);
    objRcModel = JuMP.objective_value(rcModel);
#     println("RC Objective: ", objRcModel)

    yv = JuMP.value.(y);
    kcv = JuMP.value.(kc);
    kccv = JuMP.value.(kcc);
    xsv = JuMP.value.(xs);
    xv = JuMP.value.(x);
    xrv = JuMP.value.(xr);

    # Get candidate repair solution
    yvPre = JuMP.value.(y); yv = zeros(Int64, N, T);
    for i = 1:noFailures
        for j = 1:noPeriods+1
            yv[idxFail[i],j] = trunc(Int, round(yvPre[i,j]));
        end
    end

    yv, objRcModel
end

## Get constraints for 'reduced cost' model (used to obtain discrete variable values)
@everywhere function getConstraintsSubsubproblem(rcModel, kcc, km0, yv, xv, noFailures)

    noFailures = sum(km0);
    noPeriods = trunc(Int,ceil((noFailures-1)/Y));

    kl = 0
    for i = 1:noPeriods+1
        for j = 1:length(LOADS)
            for k = 1:length(SITE)
                path = getPathPair([LOADS[j]; SITE[k]], par);
                kl = getPathConnectivity(path, km0, yv, i, 1);
                prodDers
                # prodDers = 0;
                # if G == 1
                #     prodDers = 1-xv[SITE[k]];
                # elseif G > 1
                #     prodDers = prod(ones(Float64, size(xv[SITE[k],:]))-xv[SITE[k],:]);
                # end
                # @constraint(rcModel, kcc[LOADS[j],SITE[k],i] >= 0.5*prodDers + 0.5/noFailures*sum(kl));
                # @constraint(rcModel, kcc[LOADS[j],SITE[k],i] <= prodDers + sum(kl));
            end
        end
    end

    return rcModel
end

## ------------------------------------##
## Get connectivity for edges on a path (1 if edge failed, 0 otherwise)
@everywhere function getPathConnectivity(path, km0, yv, period, opt)
    edges = zeros(Int64, length(path)-1);

    if opt == 1
        kl = zeros(AffExpr, length(path)-1);
    elseif opt == 2
        kl = zeros(Int64, length(path)-1);
    end

    for iter = 1:length(path)-1
        edges[iter] = maximum([path[iter]; path[iter+1]]);
        kl[iter] = km0[edges[iter]]-sum(yv[edges[iter],1:period]);
    end

    return kl
end

# Get connectivity for every relevant node pair (1 if connected, 0 otherwise)
@everywhere function getConnectivitySet(km0, yv, noFailures)

    noFailures = sum(km0);
    noPeriods = trunc(Int,ceil((noFailures-1)/Y));

    ac = zeros(Float64, length(LOADS), length(SITE), noPeriods + 1)
    for i = 1:noPeriods+1
        for j = 1:length(LOADS)
            for k = 1:length(SITE)
                path = getPathPair([LOADS[j]; SITE[k]], par);
                kl = getPathConnectivity(path, km0, yv, i, 2);
                sumKl = trunc(Int, sum(kl));
                ac[j,k,i] = 1 - minimum([1; sumKl]);
            end
        end
    end

    return ac
end

## ------------------------------------##
## FUNCTIONS TO GET/PRINT SOLUTIONS TO MASTER PROBLEM (MP) AND SUBPROBLEM (SP2)

@everywhere function getResultsMaster(x, xs, kcm)
    xv = JuMP.value.(x);
    xsv = JuMP.value.(xs);
    kcmv = JuMP.value.(kcm);
    genIdxSim = sum(xv, dims = 2);
    idx = findall(genIdxSim -> genIdxSim .> 0,genIdxSim);

    sizeXv = size(xv);
    for i = 1:length(xsv)
        xsv[i] = trunc(Int, xsv[i]);
        for j = 1:sizeXv[2]
            xv[i,j] = trunc(Int, xv[i,j]);
        end
    end

    xv, xsv, kcmv, genIdxSim
end

@everywhere function printResultsMaster(xv, xsv,genIdxSim)
    println("x: ", xv)
    println("xs: ", xsv)
    println("genIdxSim: ", genIdxSim)
    println("----------------------")
end

@everywhere function printResultsSP(ov, uv, kcv, betav)

    kc0 = ov; beta0 = uv; initialCost = (WSD-WLC)'*kc0 + WLC'*(ones(Float64,size(beta0))-beta0);
    totalCost = [initialCost; initialCost];

    for ts = 1:T
        costAtTs = (WSD-WLC)' * kcv[:,1,ts] + WLC' * (ones(Float64,size(betav[:,1,ts]))-betav[:,1,ts]);
        totalCost = [totalCost; costAtTs];

    end
    sysPerf = 100*(ones(Float64,size(totalCost)) - totalCost/sum(WSD));

    sysPerf
end

@everywhere function getSolutionRES_MILP(kmv, G, Y, xv, xsv)

    start = time();

    # Get Stage II subproblem (SP2) model
    xvMP = xv; xsvMP = xsv;
    sm, p, q, pg, qg, beta, P, Q, v, t, kc, v0, y, vpar, xs, x, xr, basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqbs, genIneq, genIneqbs, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs, siteDev0, siteDev, reallocFixed, reallocMobile, reallocSum, alloc, allocMobile, derOnce, derSum = getSlaveModel(kmv, G, Y, 1, xvMP, xsvMP, 0);

    # Solve (SP2), obtain objective and values of relevant variables
    optimize!(sm);
    sstatus = termination_status(sm);

    timeElapsed = time() - start;
    println("REC MILP time elapsed: ", timeElapsed)

    if sstatus != MOI.OPTIMAL
        println("Slave model solve status : ", sstatus);
        return;
    end
    objVal = JuMP.objective_value(sm);

    println("REC objective: ", objVal)

    kcv = JuMP.value.(kc); yv = JuMP.value.(y); betav = JuMP.value.(beta);
    xv = JuMP.value.(x); xsv = JuMP.value.(xs); xrv = JuMP.value.(xr);

    # println("SD: ", xsv[SITE, 1, :])
    # println("DER: ", xv[SITE, RES, 1, :])
    # println("DER realloc: ", xrv[SITE, SITE, RES, 1, :])
    # println("Repairs: ", yv[kmv .== 1, 1, :])

    sysPerf = printResultsSP(ov, uv, kcv, betav);
    # println("System performance: ", sysPerf)

    objVal, sysPerf, yv, kcv, betav
end

@everywhere function getSolutionRES_MILP_Greedy(kmv, G, Y, xvOpt, xsvOpt)

    # if Ns > 1
    #     kmv = km0[i, :];
    # else
    #     kmv = km0;
    # end
    kmv0 = kmv;
    objGreedy = zeros(Float64, T);
    kcv = zeros(Float64, N, 1, T); betav = zeros(Float64, N, 1, T);
    yv = zeros(Float64, N, T);
    xv = zeros(Float64, N, Nres, T); xsv = zeros(Float64, N, T);

    xvIter = xvOpt; xsvIter = xsvOpt;
    # println("-------------------------")
    for i = 1:T
        objGreedy[i], kmv0, yv[:,i], xvOut, xsvOut, kcv[:, 1, i], betav[:, 1, i] = getSystemPerformanceGreedy(kmv0, G, Y, i, xvIter, xsvIter);

        # println("Period: ", i)
        # println("Objective: ", objGreedy[i]);
        # println("Network state: ", kmv0);
        # println("Repairs: ", yv[:,i]);
        # println("Site development: ", xsvOut);
        # println("DER allocation: ", xvOut);
        # println("Load shedding: ", kcv);
        # println("Load control: ", betav);
        # println("-------------------------")

        xv[:,:,i] = xvOut; xsv[:,i] = xsvOut;

        xvIter = xvOut; xsvIter = xsvOut;
    end

    sysPerf = printResultsSP(ov, uv, kcv, betav);

    objVal = sum(objGreedy);

    objVal, sysPerf, yv, kcv, betav
end

@everywhere function getSystemPerformanceGreedy(kmv, G, Y, tstep, xv, xsv)

    smg = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0));

    # Variables
    smg, p, q, pg, qg, beta, P, Q, v, v0, vpar, t = getSlaveVariables(smg, 1, 1);

    y = zeros(AffExpr,N,1,1); nFailed = sum(kmv);
    idxRepair = findall(kmv .== 1);
    y[idxRepair,1,1] = @variable(smg, [1:nFailed,1,1], Bin);

    @variable(smg, kcvar[1:length(LOADS),1,1], Bin);
    kc = zeros(AffExpr,N,1,1); kc[LOADS,:,:] = kcvar;

    smg, xs, x, xr = getMobileVariables(smg, 1, 0, 1);

    # Constraints
    smg, genIneq, genIneqb = addGenInequalities(smg, pg, qg, x, 1);
    smg, droopIneq, droopIneqb = addDroopInequalities(smg, v, qg, 1:1:T, x, 1);
    smg, basicIneq, basicIneqb = addBasicInequalities(smg, beta, kc, v, t, 1);
    smg, discIneq, discIneqb = addDisconnectivityConstraints(smg, kc, v, 1);
    smg, failureIneqs, failureIneqbs = addFailureInequalities(smg, P, Q, vpar, v, kmv, y, 1);
    smg, basicEq, basicEqb = addBasicEqualities(smg, v, vpar, p, q, pg, qg, beta, P, Q, 1);

    if tstep == 1
        @constraint(smg, sum(y) == 0);
    elseif tstep < T
        @constraint(smg, sum(y) == min(Y, nFailed-1));
        @constraint(smg, y[1] == 0);
    elseif tstep == T
        @constraint(smg, sum(y) == 1);
        @constraint(smg, y[1] == 1);
    end

    if tstep == 1
        @constraint(smg, siteDev0[i = 1:length(SITE)], xs[SITE[i], 1, 1] == xsv[SITE[i]]);
    else
        @constraint(smg, siteDev[i = 1:length(SITE)], xs[SITE[i], 1, 1] >= xsv[SITE[i], 1, 1]);
    end

    @constraint(smg, reallocFixed[i = 1:length(SITE), j = 1:length(RESf)], x[SITE[i], RESf[j], 1, 1] == xv[SITE[i], RESf[j]]);

    if length(RESm) > 0
        if tstep == 1
            @constraint(smg, reallocMobile[i = 1:length(SITE), j = 1:length(RESm)], x[SITE[i], RESm[j], 1, 1] == xv[SITE[i], RESm[j]]);
        else
            @constraint(smg, allocMobile[i = 1:length(RESm)], sum(x[:, RESm[i], 1, 1]) >= sum(xv[:, RESm[i], 1, 1]));
        end
        @constraint(smg, reallocSum[i = 1:length(SITE), j = 1:length(RESm)], sum(xr[SITE[i], :, RESm[j], 1, 1]) == x[SITE[i], RESm[j], 1, 1]);
    end

    @constraint(smg, alloc[i = 1:length(SITE), j = 1:length(RES)], x[SITE[i], RES[j], 1, 1] <= xs[SITE[i], 1, 1]);
    @constraint(smg, derOnce[i = 1:length(RES)], sum(x[:, RES[i], 1, 1]) <= 1);

    @constraint(smg, derSum, sum(x[SITE, RES, 1, 1]) <= G);

    # Objective
    smg = addObjectiveSlaveModelGreedy(smg, t, beta, kc, P, xs, xr, xsv);

    # println(smg);

    # Solve (SP2), obtain objective and values of relevant variables
    optimize!(smg);
    sstatus = termination_status(smg);

    if sstatus != MOI.OPTIMAL
        println("Slave model solve status : ", sstatus);
        return;
    end
    objVal = JuMP.objective_value(smg);

    kcv = JuMP.value.(kc); yv = JuMP.value.(y); betav = JuMP.value.(beta);
    xv = JuMP.value.(x); xsv = JuMP.value.(xs); xrv = JuMP.value.(xr);

    # println(kmv)

    yvOutput = zeros(Int64, N);
    for i = 1:N
        yvOutput[i] = trunc(Int, round(yv[i]));
    end

    kmv = kmv - yvOutput;

    return objVal, kmv, yv, xv, xsv, kcv, betav
end
