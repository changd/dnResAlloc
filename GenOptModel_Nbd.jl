using JuMP
using Distributed
#  nprocs()
#  addprocs(Sys.CPU_THREADS);
#  println(Sys.CPU_THREADS)

@everywhere using Gurobi, Combinatorics
@everywhere using DistributedArrays

@everywhere using JuMP, DataFrames, LinearAlgebra, Random, Statistics, CSV
pwd()

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
    global Ns = 1;
    global T = 1;
    global CB = 1;
    global B = 1;
    global G = 1;
    global Gm = 1;
    global costGen = 1;
    global M = 10;
    global verbose = false;
    global sce = 1;
    global jmax = 1000;
    global dualConstantCollect = 1;
    global dualVarCollect = 1;
    global siteDevCollect = 1;
    global derAllocCollect = 1;
    global dualVar = 1;
    global objSlaveCollect = 1;
    global costToGo = 1;
    global derReallocSoln = 1;
    global derReallocSolnBD = 1;
    global rcCollect = 1;
end

@everywhere function powerset(a, min::Integer=0, max::Integer=length(a))
    itrs = [combinations(a, k) for k = min:max]
    min < 1 && append!(itrs, eltype(a)[])
    IterTools.chain(itrs...)
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

@everywhere function setGlobalParameters(optMobile)
    # println("Reached 1");
    global vmin, vmax, vmmin, vmmax, vgmin, vcmin, WAC, WLC, WVR, WSD, ov, uv
    global xrratio, rc, Cloadc, resistance, reactance, srmax
    global infty, tol, epsilon, I, O, fmin, fmax, WLC, WSD, nodes, WMG, WEG
    global path, commonPath, R, X, betamin, vgmax, vcmax, v0nom, v0dis
    global pcmax, qcmax, LLCmax, pgmax, qgmax, betamax, delta, pdmax, qdmax, Smax, pgnom, qgnom, vref, mp, mq, SITE, Nres, Nsite, WSITE, WSITE_2, WR

    global myinf, leaves, SDI, noSDI, MGL, noMGL, par, SuccMat, RES, RESf, RESm, EG, semax, DG, UDG, LOADS, CHILD
    global LTC, noLTC, N, onev

    ov = zeros(Float64,N); uv = ones(Float64,N); v0dis = 0.001; epsilon = 0.1^5; infty = 10^5+1; tol = 0.1^3; myinf = 0.1^7;
    v0nom = 1; I = zeros(N,N) + UniformScaling(1); O = zeros(Float64,N,N); vmin = 0.95uv; vmax = 1.05uv; WVR = 10; Smax = 2uv;
    vmmmax = 2uv; vref = 1.00;
    WMG = zeros(Float64,N); WEG = zeros(Float64,N); WLC = zeros(Float64,N); WSD = zeros(Float64,N); WSITE = zeros(Float64,N);
    WSITE_2 = zeros(Float64,N); WR = zeros(Float64,N,N,G);
    pcmax = zeros(Float64,N); qcmax = zeros(Float64,N); pgmax = zeros(Float64,N); qgmax = zeros(Float64,N);

    nodes = collect(1:N); EG = copy(nodes); LOADS = copy(nodes); LTC = zeros(Int64,0); DG = copy(nodes); SDI = nodes[nodes.%2 .== 1]; noSDI = setdiff(nodes,SDI); MGL = zeros(Int64,0);
    par = collect(0:N-1);
    if N%3 == 0
        MGL = [1; Int(N/3)+1; 2Int(N/3)+1];
    end
    if N%2 == 0
        EG = nodes[nodes.%2 .== 1];
    end
    # EG = union(EG, MGL);


    Random.seed!(716);
    # println(nodes[randperm(N)], " ", round(Int64,N/2));
    # DG = (nodes[randperm(N)])[1:round(Int64,N/2)];
    DG = sort((nodes[randperm(N)])[1:round(Int64,N/2)]);

    #println("DG ", DG);

    if N == 3
        par[3] = 1;
        xrratio = 2; rc = 0.1; srmax = 0.2; Cloadc = 100;
        DG = copy(nodes); LOADS = copy(nodes);

        RES = copy(nodes); SITE = copy(nodes);

        pcmax = 1srmax * uv; qcmax = pcmax / 3;
        pgmax = 0.8srmax * uv; qgmax = pgmax / 3;
        WLC[LOADS] = 300; WSD[LOADS] = 1000; WMG[MGL] = 400; WEG[EG] = 200;
        WSITE[SITE] = 0;
        println("reached 1 ");

        vmmin = vmin; vmmax = vmax; vgmin = 0.9uv; vgmax = 1.1uv; vcmin = 0.9uv; vcmax = 1.1uv;
        betamax = 0.2uv; betamin = 1 - betamax;
        pdmax = srmax * uv; qdmax = pdmax/3;
    elseif N == 6
        par[5] = 2;
        xrratio = 2; rc = 0.03; srmax = 0.1; Cloadc = 100;
        DG = copy(nodes);
        pcmax = 1.5srmax * uv; qcmax = pcmax / 3;
        pgmax = 0.5srmax * uv; qgmax = pgmax / 3;
        SDI = copy(nodes); noSDI =[];
        WLC[LOADS] = 100; WSD[LOADS] = 1000;
        MGL = copy(nodes);
        WMG[MGL] = 400; WEG[EG] = 200;
        vmmin = vmin; vmmax = vmax; vgmin = 0.95uv; vgmax = 1.1uv; vcmin = 0.9uv; vcmax = 1.1uv;
        LLCmax = sum(WSD); betamax = 0.2uv; betamin = 1 - betamax;
        pdmax = srmax * uv; qdmax = pdmax/3;
    elseif N == 12
        Random.seed!(123113);
        #SITE = sort((nodes[randperm(N)])[1:round(Int64,N/3)]);
        SITE = [1, 4, 9, 12];
        WSITE[SITE] = 100*ones(size(SITE));
        # SITE = copy(nodes)
        betamax = 0.5uv; betamin = 1 .- betamax;

        if !diversification
            par[9] = 4;
            xrratio = 2; rc = 0.01; srmax = 1/N; Cloadc = 100;
            srmax = 6/N; # for sequential vs online
            # DG = copy(nodes);
            LOADS = setdiff(nodes,DG)
            LOADS_CRIT = [3, 6]
            LOADS_NCRIT = setdiff(LOADS, LOADS_CRIT);
            pcmax[LOADS] = 1.25srmax * ones(size(LOADS)); qcmax = pcmax / 3;
            pgmax[DG] = srmax * ones(size(DG)); qgmax = pgmax / 3;
            #WLC[LOADS] = 100;
            WLC[LOADS_CRIT] = 500 * ones(size(LOADS_CRIT)); WLC[LOADS_NCRIT] = 100 * ones(size(LOADS_NCRIT));
            if opt[4] > 0
                WSD[LOADS] = opt[4]  * ones(size(LOADS));
            elseif opt[4] == 0
                WSD[LOADS_CRIT] = 5000  * ones(size(LOADS_CRIT)); WSD[LOADS_NCRIT] = 1000  * ones(size(LOADS_NCRIT));
            end
            # MGL = copy(nodes);
            WMG[MGL] = 200 * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
            SDI = copy(nodes); noSDI = [];
            # LTC = [5];
            vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
            pdmax = 2srmax * uv; qdmax = pdmax/3;
        else
            par[9] = 4;
            xrratio = 2; rc = 0.01; srmax = 1/N; Cloadc = 100;
            srmax = 3/N; # for sequential vs online
            # DG = copy(nodes);
            # LOADS = setdiff(nodes,DG)
            DG = copy(nodes); LOADS = copy(nodes);
            pcmax[LOADS] = 2.2srmax; qcmax = pcmax / 3;
            pgmax[DG] = srmax; qgmax = pgmax / 3;
            WLC[LOADS] = 100; WSD[LOADS] = 1000;
            # MGL = copy(nodes);
            WMG[MGL] = 200; WEG[EG] = 200;
            SDI = copy(nodes); noSDI = [];
            # LTC = [5];
            vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
            pdmax = 2srmax * uv; qdmax = pdmax/3;
        end
    elseif N == 24
        par[13] = 2; par[20] = 6; par[17] = 3;
        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100;
        # DG = copy(nodes);
        LOADS = setdiff(nodes, DG);
        pcmax[LOADS] = 1.25srmax; qcmax = pcmax / 3;
        pgmax[DG] = srmax; qgmax = pgmax / 3;
        WLC[LOADS] = 100; WSD[LOADS] = 1000; MGL = [5,6,10];
        WMG[MGL] = 400; WEG[EG] = 200;
        betamax = 0.2uv; betamin = 1 - betamax;
        pdmax = 3srmax * uv; qdmax = pdmax/3;
        vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
    elseif N == 36

        par[5] = 3; par[6] = 2; par[10] = 7; par[13] = 10; par[15] = 13; par[16] = 2; par[20] = 18; par[21] = 16; par[24] = 22; par[25] = 22; par[27] = 25; par[31] = 29; par[32] = 28; par[36] = 34;
        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100;

        SITE = [1,5,7,8,11,12,20,22,24,29,33,35];
        #SITE = sort((nodes[randperm(N)])[1:round(Int64,N/2)]);
        # DG = copy(nodes);
        LOADS = setdiff(nodes, DG);
        LOADS_CRIT = [4,10,20,27]; LOADS_NCRIT = setdiff(LOADS, LOADS_CRIT);

        pcmax[LOADS] = 1.25*srmax * ones(size(LOADS)); qcmax = pcmax / 3;
        pgmax[DG] = srmax* ones(size(LOADS)); qgmax = pgmax / 3;
        WLC[LOADS] = 100 * ones(size(LOADS)); MGL = [5,6,10,13,15,25,32];
        WLC[LOADS_CRIT] = 500 * ones(size(LOADS_CRIT)); WLC[LOADS_NCRIT] = 100 * ones(size(LOADS_NCRIT));
        if opt[4] > 0
            WSD[LOADS] = opt[4] * ones(size(LOADS));
        elseif opt[4] == 0
            WSD[LOADS_CRIT] = 5000 * ones(size(LOADS_CRIT)); WSD[LOADS_NCRIT] = 1000 * ones(size(LOADS_NCRIT));
        end

        WMG[MGL] = 400  * ones(size(MGL)); WEG[EG] = 200 * ones(size(EG));
        WSITE[SITE] = 100*ones(size(SITE));
        betamax = 0.5uv; betamin = ones(size(betamax)) - betamax;
        pdmax = 3srmax * uv; qdmax = pdmax/3;
        vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
    elseif N == 69
        par[28] = 3; par[36] = 3; par[47] = 4; par[45] = 8; par[53] = 9; par[66] = 11; par[68] = 12;
        xrratio = 2; rc = 0.01; srmax = 6/N; Cloadc = 100; gammac = 1; lcfmin = 0.8ones(N,1);
        LOADS = setdiff(nodes, DG);
        SITE = [2,4,6,7,12,13,14,15,16,17,18,20,21,22,27,30,31,33,34,35,36,46,47,48,54,55,56,58,59,60,64,65,67,69];
        # SITE = sort((nodes[randperm(N)])[1:round(Int64,N/2)]);

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
    elseif N == 124
        df = CSV.read("N123.csv");
        resistance = df[:R]; reactance = df[:X]; P = (df[:P])[1:85]; Q = (df[:Q])[1:85];
        loadNodes = (df[:Node])[1:85];
        loadNodes = map(loadNodes->parse(Int64,loadNodes),loadNodes)
        originalNodes = zeros(1:450);
        nodesInDoc = df[:nodeNoInDoc]; nodesInJulia = df[:nodeNoInJulia];
        to = df[:to]; from = df[:from];
        for i = 1:124
            j = nodesInDoc[i]; corr = nodesInJulia[i]; originalNodes[j] = corr;
        end
        nNodes = 124;
        N = nNodes; nodes = collect(1:nNodes); par = zeros(Int64,nNodes);

        for i = 1:122
            fromNode = from[i]; toNode = to[i];
            ofromNode = originalNodes[fromNode];
            otoNode = originalNodes[toNode];
            par[otoNode] = ofromNode;
        end

        pcmax = zeros(Float64,N); qcmax = zeros(Float64,N);
        pcmax[originalNodes[loadNodes]] = P;
        qcmax[originalNodes[loadNodes]] = Q;

        xrratio = 2; rc = 0.1; srmax = 1/N; Cloadc = 100; gammac = 1; lcfmin = 0.8ones(N,1);
        # pcmax = 2.5srmax * uv; qcmax = pcmax / 3;
        pgmax = 0.3pcmax; qgmax = 0.3qgmax;
        WLC[LOADS] = 100; WSD[LOADS] = 1000;
        MGL = [28,36,47,45,53,66];
        LTC = [13,33,73];
        WMG[MGL] = 400; WEG[EG] = 200;
        betamax = 0.2uv; betamin = 1 - betamax;
        pdmax = 2srmax * uv; qdmax = pdmax/3; RES = copy(MGL);
    end

    # RES = copy(MGL);
    Random.seed!(167); ldg = round(Int64,length(DG)/2);

    # RES = sort((DG[randperm(ldg)])[1:round(Int64,ldg/2)]);
    # RES = collect(1:G);
    RES = collect(1:G); #RESm = collect(1:Gm);
    resIdx = trunc(Int, ceil(G/2));
    # RESf = collect(1:G); RESm = [];
    if optMobile == 1
        RESf = collect(1:G); RESm = [];
    elseif optMobile == 2
        RESf = 1:resIdx; RESm = resIdx+1:G;
    end
    #println("RES ", RES);

    EG = copy(MGL);
    #println("MGL ", MGL);
    #println(sum(pcmax))
    #println("*******************************************************")
    onev = ones(Float64,length(RES),1);

    Nres = length(RES); Nsite = length(SITE);

    noMGL = setdiff(nodes,MGL); noLTC = setdiff(nodes,LTC);
    LLCmax = sum(WSD); # vmmin = vmin; vmmax = vmax; vgmin = 0.92uv; vgmax = 1.05uv; vcmin = 0.9uv; vcmax = 1.1uv;
    semax = zeros(Float64, N);
    WAC = zeros(Float64,N); WAC[1] = 0.1WLC[1];
    semax[EG] = 0.5(sum(pcmax))/length(EG)  * ones(size(EG));

    pgnom = 0pgmax; qgnom = 0qgmax; mq = 0.1;

    UDG = resourceResponse ? setdiff(DG,RES) : DG;
    UDG = diversification ? DG : UDG;
    # semax[MGL] = 3semax[MGL];

    noMGL = setdiff(nodes, MGL);
    # println("Reached 2");
    # bilevel(N);
    leaves = getLeaves(par);
    CHILD = getChildrenMatrix(par);
    SuccMat = getSuccessorMatrix();

    r = rc * ones(Float64,N); x = xrratio * copy(r);

    if N != 124
        resistance = copy(r); reactance = copy(x);
    end
    path = getPath(par);
    commonPath, R, X = getCommonPathVariables(N,par,r,x);

    costGen = ones(Int64,N);
    pgmax[SITE] = 0.8sum(pcmax) / G * ones(size(SITE));
    qgmax = pgmax/3; mq = 0.1;

    #println("reached 2 ");
    # println("R :", R);
    # println("X :", X);
    # println("size of R ", size(R));
    # N
end

@everywhere function getDualRepair(kmv, repairIneqs, repairEqs, noFailures)
    repairIneqLine = getRepairIneqBar(repairIneqs, noFailures);
    repairEqbar, repairEqPeriod, repairEqSubstation = getRepairEqbar(repairEqs, noFailures, 1);
    repairIneqbar = zeros(Float64, noFailures);
    for i = 1:sum(kmv)
        repairIneqbar[i] = JuMP.dual(repairIneqs[i]);
    end

    repairIneqbar, repairEqbar, repairEqPeriod, repairEqSubstation
end

@everywhere function  getDualCostRepairIneq(idxFail, yPre);

    repairIneqbsBenders = zeros(AffExpr, length(idxFail));
    for i = 1:length(idxFail)
        repairIneqbsBenders[i] = 1 - sum(yPre[idxFail[i],:]);
    end

    repairIneqbsBenders
end

@everywhere function getDualResourceAlloc(siteDev, mpDerAlloc, mpDerRealloc, siteDer, derOnce, derSum, xsvMP, xvMP);
    siteDevV = zeros(Float64, length(SITE));
    mpDerAllocV = zeros(Float64, length(SITE), length(RESf));
    mpDerReallocV = zeros(Float64, length(SITE), length(RESm));
    siteDerV = zeros(Float64, length(SITE), length(RES));
    derOnceV = zeros(Float64, length(RES));
    derSumV = JuMP.dual(derSum);

    for i = 1:length(SITE)
        siteDevV[i] = JuMP.dual(siteDev[i]);
        for j = 1:length(RES)
            siteDerV[i,j] = JuMP.dual(siteDer[i,j]);
        end
        for j = 1:length(RESf)
            mpDerAllocV[i,j] = JuMP.dual(mpDerAlloc[i,j]);
        end
        for j = 1:length(RESm)
            mpDerReallocV[i,j] = JuMP.dual(mpDerRealloc[i,j]);
        end
    end

    for i = 1:length(RES)
        derOnceV[i] = JuMP.dual(derOnce[i]);
    end

    siteDevV, mpDerAllocV, mpDerReallocV, siteDerV, derOnceV, derSumV
end

@everywhere function getDualPlusRC(noFailures, y, yv, Ts)
    Ts = 3;
    idxFailPre = findall(km0[:,sce] .== 1);
    idxFail = zeros(Int64, length(idxFailPre));
    for i = 1:length(idxFailPre)
        idxFail[i] = idxFailPre[i][1];
    end

    repairIneqbsBenders = zeros(AffExpr, length(idxFail));
    for i = 1:length(idxFail)
        repairIneqbsBenders[i] = 1 - sum(y1[idxFail[i],:]);
    end

    fixed1v =  zeros(Float64, length(SITE), length(setInc));
    fixed2v = zeros(Float64, length(SITE), length(RES), length(setInc));
    # fixed3v = zeros(Float64, length(SITE), length(SITE), length(RES), length(setInc));

    for i = 1:length(SITE)
        for j = 1:length(setInc)
            fixed1v[i,j] = JuMP.dual(fixed1[i,j]);
            for k = 1:length(RES)
                fixed2v[i,k,j] = JuMP.dual(fixed2[i,k,j]);
            end
        end
    end

    # fixed1v_b = repeat(xs, 1, length(setInc)); fixed2v_b = repeat(x, 1, 1, length(setInc));
    fixed1v_b = repeat(xsvMP, 1, length(setInc)); fixed2v_b = repeat(xvMP, 1, 1, length(setInc));

    dualSolution = sum(ineqbar .* Ineqbs) + sum(eqbar .* Eqbs);
    dualSolution += sum(WLC[LOADS])*Ts + sum(repairEqbs .* repairEqbar);

    # dualSolution += sum(fixed1v .* fixed1v_b[SITE,:]) + sum(fixed2v .* fixed2v_b[SITE,:,:]);
    # dualSolution += sum(fixed1v .* fixed1v_b) + sum(fixed2v .* fixed2v_b);

    # dualSolution += sum(failureIneqbar .* failureIneqbsBenders) + sum(repairIneqbar .* repairIneqbsBenders);
    dualSolution += sum(failureIneqbar .* failureIneqbs) + sum(repairIneqbar .* repairIneqbs);

    dualSolution += sum(genIneqbar .* genIneqb) + sum(droopIneqbar .* droopIneqb);

    rc_kc = getReducedCostsLoads(basicIneqbar, discIneqbar, T1);

    dualSolution, rc_kc
end

@everywhere function getConnectivitySet(kmv)

    ac = zeros(Float64, length(LOADS), length(SITE))
    for j = 1:length(LOADS)
        for k = 1:length(SITE)
            path = getPathPair([LOADS[j]; SITE[k]], par);
            kl = getPathConnectivity(path, kmv, 2);
            sumKl = trunc(Int, sum(kl));
            ac[j,k] = 1 - minimum([1; sumKl]);
        end
    end

    return ac
end

@everywhere function getPathConnectivity(path, km0, opt)
    edges = zeros(Int64, length(path)-1);

    if opt == 1
        kl = zeros(AffExpr, length(path)-1);
    elseif opt == 2
        kl = zeros(Int64, length(path)-1);
    end

    for iter = 1:length(path)-1
        edges[iter] = maximum([path[iter]; path[iter+1]]);
        kl[iter] = km0[edges[iter]];
    end

    return kl
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

@everywhere function getPeriodSubproblem(kmv, param, xsvMP, xvMP, optSP, yvPre, bcSum, dualSolnInp, dualSolnX1, dualSolnX2, kcInp)

    # Parameters / options
    G = param[1]; Yk = param[2]; ts = param[3];
    LP = optSP[1]; mobile = optSP[2]; BC = optSP[3]; bckw = optSP[4];
    derRealloc = optSP[5]; sce = optSP[6]; iter = optSP[7]; derReallocNext = optSP[8];

    # Get indices of all failed lines
    nFailed = trunc(Int,sum(kmv)); idxFailPre = findall(kmv .== 1);
    idxFail = zeros(Int64, length(idxFailPre));
    for i = 1:length(idxFailPre)
        idxFail[i] = idxFailPre[i][1];
    end

    # SET UP SUBPROBLEM
    # ---------------------------

    sm2 = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0, MIPGap = 1e-10));
    @variable(sm2, theta);

    # Get constraints/variables for resource allocation
    sm2, xs, x, xr = getMobileVariables(sm2, LP);
    sm2, siteDev, mpDerAlloc, mpDerRealloc, siteDer, derOnce, derSum = getMobileConstraints(sm2, xs, x, xr, xsvMP, xvMP, derRealloc);

    # Get remaining variables
    sm2, p, q, pg, qg, beta, P, Q, v, v0, vpar, t = getSubproblemVariables(sm2);

    y = zeros(AffExpr,N);
    if LP == 0
        y[idxFail] = @variable(sm2, yv[1:nFailed], binary = true);
        @variable(sm2, kcvar[1:length(LOADS)], binary = true); # kc
    elseif LP == 1
        y[idxFail] = @variable(sm2, yv[1:nFailed]);
        @variable(sm2, kcvar[1:length(LOADS)]); # kc
        @constraint(sm2, y[idxFail] .>= 0); #@constraint(sm3, y[idxFail, 1:T1] .<= 1);
        @constraint(sm2, kcvar .>= 0); #@constraint(sm3, kcvar .<= 1);
    end
    kcv = zeros(AffExpr,N); kcv[LOADS,:] = kcvar;

    # Get remaining constraints
    basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs = addGenPlaceConstraintsSAA(sm2, ts, p, q, pg, qg, beta, P, Q, v, v0, t, vpar, kcv, kmv, y, yvPre, costGen, G, Yk, 1, x, 1, 1);

    # If backwards pass, add Benders cut(s) to subproblem
    if bckw == 1
        # Dual costs corresponding to repair variables
        failureIneqbsBenders = addBendersCutFailures(km0[:,sce], ts, y, yvPre);
        repairIneqbsBenders = addBendersCutRepairs(idxFail, y, yvPre, ts);

        # Dual variables/costs corresponding to resource allocation
        derAllocInp = dualSolnX2[:,1:length(RESf)]; derReallocInp = dualSolnX2[:,1+length(RESf):end];
        siteDevInp = dualSolnX1[1:length(SITE)];
        dualSolnRepairIneqPre = dualSolnInp[1:N]; dualSolnFailure = dualSolnInp[N+3:end];
        dualSolnRepairIneq = dualSolnRepairIneqPre[idxFail];

        # Form Benders cut
        bendersCut = bcSum + sum(dualSolnRepairIneq.*repairIneqbsBenders) + sum(dualSolnFailure.*failureIneqbsBenders);
        if derReallocNext == 1
            bendersCut += sum(siteDevInp .* xs[SITE]) + sum(derAllocInp .* x[SITE,RESf]);
        else
            bendersCut += sum(siteDevInp .* xs[SITE]) + sum(derAllocInp .* x[SITE,RESf]) - sum(derReallocInp .* x[SITE,RESm]);
        end
        @constraint(sm2, theta >= 0); @constraint(sm2, thetalb, theta >= bendersCut);

        if LP == 1
            # println("Benders cut: ", bendersCut)
        end
    elseif bckw == 0
        @constraint(sm2, thetalb, theta >= 0);
    end

    bendersCutCollect = zeros(AffExpr, iter-1);

    # Add benders cuts from previous iterations
    for i = 1:iter-1
        failureIneqbsBendersPre = addBendersCutFailures(km0[:,sce], ts, y, yCollect[sce,:,1:ts-1,i]);
        repairIneqbsBendersPre = addBendersCutRepairs(idxFail, y, yCollect[sce,:,1:ts-1,i], ts);

        dualSolnRepairIneqPre = dualVarCollect[sce,1:N,ts+1,i];
        dualSolnRepairIneqPre = dualSolnRepairIneqPre[idxFail];
        dualSolnFailurePre = dualVarCollect[sce,N+3:end,ts+1,i];

        bendersCut = dualConstantCollect[sce,ts+1,i] + sum(dualSolnRepairIneqPre.*repairIneqbsBendersPre) + sum(dualSolnFailurePre.*failureIneqbsBendersPre);

        siteDevInpPre = siteDevCollect[sce,1:length(SITE),ts+1,i];
        derAllocInpPre = derAllocCollect[sce,:,1:length(RESf),ts+1,i];
        derReallocInpPre = derAllocCollect[sce,:,1+length(RESf):end,ts+1,i];
        if derReallocNext == 1
            bendersCut += sum(siteDevInpPre .* xs[SITE]) + sum(derAllocInpPre .* x[SITE,RESf]);
        else
            bendersCut += sum(siteDevInpPre .* xs[SITE]) + sum(derAllocInpPre .* x[SITE,RESf]) - sum(derReallocInpPre .* x[SITE,RESm]);
        end
        bendersCutCollect[i] = bendersCut;

        if LP == 1
            # println("Benders cut: ", bendersCutCollect[i])
        end
    end

    if iter > 1
        @constraint(sm2, thetalbPre[i = 1:iter-1], theta >= bendersCutCollect[i]);
    end

    # Add objective
    sm2 = addObjectiveSlaveModel(sm2, t, beta, kcv, P, Ns, 1, theta);

    # -------------------------------------

    # Solve problem, get solution
    optimize!(sm2); sstatus = termination_status(sm2);
    if sstatus != MOI.OPTIMAL
        println("Slave model solve status : ", sstatus);
        return;
    end
    objSlave = JuMP.objective_value(sm2); #println("Objective: ", objSlave)

    kcv = JuMP.value.(kcv); yv = JuMP.value.(y); betav = JuMP.value.(beta);
    xv = JuMP.value.(x); xsv = JuMP.value.(xs);
    thetav = JuMP.value.(theta); #println("Theta: ", thetav)
    # println("Repair: ", findall(yv[idxFail] .== 1))
    # println("Placement: ", xv[SITE,RES])

    if bckw == 0
        objSlave -= thetav;
    end
    # println("Theta: ", thetav)
    # println("Objective: ", objSlave)

    totalCost, totalCostNorm = printResultsSP(1, ov, uv, kcv, betav, 1);# : nothing;

    # If solving LP, get components for the Benders cuts
    # Constant term, relevant dual variables/dual costs, reduced costs, contribution from Benders cuts
    dualTheta = []; dualTotal = 0;
    if LP == 1
        # Get all dual variables / dual cost vectors
        noFailures = sum(kmv); y1 = zeros(Int64,N); yv1 = zeros(Int64,N)
        ineqbar, Ineqbs, failureIneqbar, basicIneqbar, discIneqbar, droopIneqbar, genIneqbar, eqbar = getDualVariables(basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, genIneq, Eqs, repairIneqs);
        repairIneqbar, repairEqbar, repairEqPeriod, repairEqSubstation = getDualRepair(kmv, repairIneqs, repairEqs, noFailures);
        repairIneqbsBenders = getDualCostRepairIneq(idxFail, y);
        siteDevV, mpDerAllocV, mpDerReallocV, siteDerV, derOnceV, derSumV = getDualResourceAlloc(siteDev, mpDerAlloc, mpDerRealloc, siteDer, derOnce, derSum, xsvMP, xvMP);

        # Get constant term (not function of repairs or resource alloc discrete variables)
        dualConstant = sum(ineqbar .* Ineqbs) + sum(eqbar .* Eqbs) + sum(genIneqbar .* genIneqb) + sum(droopIneqbar .* droopIneqb) + sum(WLC[LOADS]) + sum(repairEqbs .* repairEqbar);
        dualConstant += sum(derOnceV) + G*derSumV;

        dualTotal = dualConstant + sum(failureIneqbar .* failureIneqbs) + sum(repairIneqbar .* repairIneqbs);
        if derRealloc == 1
            dualTotal += sum(siteDevV .* xsvMP[SITE]) + sum(mpDerAllocV .* xvMP[SITE,RESf]);
        else
            dualTotal += sum(siteDevV .* xsvMP[SITE]) + sum(mpDerAllocV .* xvMP[SITE,RESf]) - sum(mpDerReallocV .* xvMP[SITE,RESm]);
        end

        derReallocInp = dualSolnX2[:,1+length(RESf):end];

        # Add reduced cost contribution from load shedding
        rc_kc = getReducedCostsLoads(basicIneqbar, discIneqbar);

        if bckw == 1
            rcCollect[sce,:,ts,iter] = rc_kc;
        end

        # if ts == 1
        #     dualConstant += sum(rc_kc .* kcInp[LOADS]);
        # end
        # println("Objective plus RC: ", objSlave + sum(rc_kc .* kcInp[LOADS]))

        # dualSolution += sum(rc_kc .* kcv[LOADS]);
        thetaBar = JuMP.dual(thetalb);

        # Add contributions from Benders cuts, if backwards pass
        if bckw == 1
            repairIneqbsBdCut = addBendersCutThetaRepairs(kmv, idxFail, yvPre, ts);
            failureIneqbsBdCut = addBendersCutThetaFailures(kmv, idxFail, yvPre, ts);

            dualTheta = thetaBar*(bcSum + sum(repairIneqbsBdCut.*dualSolnRepairIneq) + sum(failureIneqbsBdCut.*dualSolnFailure));
            dualConstant += dualTheta;

            for i = 1:iter-1
                repairIneqbsBdCutPre = addBendersCutThetaRepairs(kmv, idxFail, yCollect[sce,:,1:ts-1,i], ts);
                failureIneqbsBdCutPre = addBendersCutThetaFailures(kmv, idxFail, yCollect[sce,:,1:ts-1,i], ts);

                dualSolnRepairIneqPre = dualVarCollect[sce,1:N,ts+1,i];
                dualSolnRepairIneqPre = dualSolnRepairIneqPre[idxFail];
                dualSolnFailurePre = dualVarCollect[sce,N+3:end,ts+1,i];

                dualTheta = dualConstantCollect[sce,ts+1,i] + sum(repairIneqbsBdCutPre.*dualSolnRepairIneqPre) + sum(failureIneqbsBdCutPre.*dualSolnFailurePre);
                thetaBarPre = JuMP.dual(thetalbPre[i]);
                dualConstant += thetaBarPre*dualTheta;
            end
        end

        repairBar = zeros(AffExpr,N+2);
        repairBar[idxFail] = repairIneqbar;
        repairBar[N+1:end] = repairEqbar;

        dualSolnX1 = [siteDevV; derOnceV; derSumV];
        dualSolnX2 = [mpDerAllocV mpDerReallocV];
    else
        dualConstant = []; dualSolnX1 = []; dualSolnX2 = [];
        repairBar = []; failureIneqbar = []; rc_kc = [];
    end

    dualSoln = [repairBar; failureIneqbar];

    kcvOutput = zeros(Float64, N); yvOutput = zeros(Int64, N);
    xsvOutput = zeros(Int64,N); xvOutput = zeros(Int64,N,length(RES));
    for i = 1:N
        xsvOutput[i] = trunc(Int, round(xsv[i]));
        kcvOutput[i] = round(kcv[i]);
        yvOutput[i] = trunc(Int, round(yv[i]));
        for j = 1:length(RES)
            xvOutput[i,j] = trunc(Int, round(xv[i,j]));
        end
    end

    objSlave, yvOutput, xsvOutput, xvOutput, dualConstant, dualSoln, dualSolnX1, dualSolnX2, dualTotal, kcvOutput, rc_kc, betav
end

@everywhere function getSubproblemVariables(sm)
    @variable(sm, p[1:N]); @variable(sm, q[1:N]);
    @variable(sm, pgvar[1:length(SITE),1:length(RES)]);
    @variable(sm, qgvar[1:length(SITE),1:length(RES)]);
    @variable(sm, betavar[1:length(LOADS)]);
    @variable(sm, v[1:N]); @variable(sm, v0);
    @variable(sm, P[1:N]); @variable(sm, Q[1:N]);
    @variable(sm, tvar[1:N]);

    beta = zeros(AffExpr,N); t = zeros(AffExpr,N);
    pg = zeros(AffExpr,N,Nres); qg = zeros(AffExpr,N,Nres);
    beta[LOADS] = betavar; vpar = zeros(AffExpr,N);
    t = tvar;
    pg[SITE,:] = pgvar; qg[SITE,:] = qgvar;

    for i = 1:N
        vpar[i] = par[i] == 0 ? v0 : v[par[i]];
    end

    sm, p, q, pg, qg, beta, P, Q, v, v0, vpar, t
end

@everywhere function getMobileVariables(sm, LP)
if LP == 0
    @variable(sm, xsvar[1:length(SITE)], Bin);
    @variable(sm, xvar[1:length(SITE),1:length(RES)], Bin);
    @variable(sm, xrvar[1:length(SITE),1:length(SITE),1:length(RES)], Bin);
else
    @variable(sm, xsvar[1:length(SITE)]);
    @variable(sm, xvar[1:length(SITE),1:length(RES)]);
    @variable(sm, xrvar[1:length(SITE),1:length(SITE),1:length(RES)]);
end

    xs  = zeros(AffExpr,N); xs[SITE] = xsvar;
    x  = zeros(AffExpr,N,Nres); x[SITE,RES] = xvar;
    xr = zeros(AffExpr,N,N,Nres); xr[SITE,SITE,RES] = xrvar;

    sm, xs, x, xr
end

@everywhere function getMobileConstraints(sm, xs, x, xr, xsvMP, xvMP, opt)

    # Ts = length(setInc) + length(setExc);

    if opt == 0
        @constraint(sm, siteDev[i = 1:length(SITE)], xs[SITE[i]] == xsvMP[SITE[i]]);
        @constraint(sm, mpDerRealloc[i = 1:length(SITE), j = 1:length(RESm)], xvMP[SITE[i], RESm[j]] == x[SITE[i],RESm[j]]);
    elseif opt == 1
        @constraint(sm, siteDev[i = 1:length(SITE)], xs[SITE[i]] >= xsvMP[SITE[i]]);
        @constraint(sm, mpDerRealloc[i = 1:length(SITE), j = 1:length(RESm)], sum(xr[SITE[i],:,RESm[j]]) == x[SITE[i],RESm[j]]);
    end

    @constraint(sm, mpDerAlloc[i = 1:length(SITE), j = 1:length(RESf)], x[SITE[i], RESf[j]] == xvMP[SITE[i],RESf[j]]);

    @constraint(sm, siteDer[i = 1:length(SITE), j = 1:length(RES)], x[SITE[i], RES[j]] <= xs[SITE[i]]);
    @constraint(sm, derOnce[i = 1:length(RES)], sum(x[:, RES[i]]) <= 1);

    @constraint(sm, derSum, sum(x[SITE, RES]) <= G);

    sm, siteDev, mpDerAlloc, mpDerRealloc, siteDer, derOnce, derSum
end

@everywhere function addObjectiveSlaveModel(sm, t, beta, kcval, P, Ns, Ts, theta)
    @objective(sm, Min, theta + sum(sum(WAC) * P[1,j] + WVR * sum(t[:,j])/N + sum(WLC[LOADS].*(ones(Float64,length(LOADS),1)-beta[LOADS,j]))  + sum((WSD-WLC).*kcval[:,j]) for j=1:Ts));

    sm
end

@everywhere function addGenPlaceConstraintsSAA(sm, ts, p, q, pg, qg, beta, P, Q, v, v0, t, vpar, kcv, kmv, y, yv, costGen, G, Yk, Ns, xv, optRepair, Ts)
    Ineqs=[];
    Ineqbs=[];
    Eqs=[];
    Eqbs=[];

    sm, basicIneq, basicIneqb = addBasicInequalities(sm, beta, kcv, v, t, );
    sm, discIneq, discIneqb = addDisconnectivityConstraints(sm, kcv, v);
    sm, droopIneq, droopIneqb = addDroopInequalities(sm, v, qg, xv, optRepair);
    sm, genIneq, genIneqb = addGenInequalities(sm, pg, qg, xv);
    sm, basicEq, basicEqb = addBasicEqualities(sm, v, vpar, p, q, pg, qg, beta, P, Q);
    sm, failureIneqs, failureIneqbs = addFailureInequalities(sm, ts, P, Q, vpar, v, kmv, y, yv);

    sm, repairIneqs, repairIneqbs, repairEqs, repairEqbs = addRepairConstraints(sm, ts, kmv, y, yv, Yk, optRepair);

    Eqs = basicEq;
    Eqbs = basicEqb;

    # Ineqs, Ineqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs
    basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, droopIneqb, genIneq, genIneqb, Eqs, Eqbs, repairIneqs, repairIneqbs, repairEqs, repairEqbs
end

@everywhere function addBasicInequalities(sm, beta, kcval, v, t)
    @constraint(sm, betalb[i = 1:length(LOADS)], beta[LOADS[i]] >= (1-kcval[LOADS[i]]) * betamin[LOADS[i]]);
    @constraint(sm, betaub[i = 1:length(LOADS)], beta[LOADS[i]] <= (1-kcval[LOADS[i]]));

    @constraint(sm, lovrMin[i=1:N], t[i] + v[i] >= v0nom * uv[i]);
    @constraint(sm, lovrMax[i=1:N], t[i] - v[i] >= -v0nom * uv[i]);

    basicIneqs = [betalb; betaub; lovrMin; lovrMax];

    basicIneqbs = [betamin[LOADS]; ones(Float64, length(LOADS))];
    basicIneqbs = [basicIneqbs; v0nom * uv; -v0nom * uv];

    sm, basicIneqs, basicIneqbs
end

@everywhere function addDisconnectivityConstraints(sm, kcval, v)
    @constraint(sm, loadLow[i=1:length(LOADS)], v[LOADS[i]]  >= vcmin[LOADS[i]] - kcval[LOADS[i]]);
    @constraint(sm, loadUp[i=1:length(LOADS)], -v[LOADS[i]]  >=  -vcmax[LOADS[i]] - kcval[LOADS[i]]);

    discIneqs = [loadLow; loadUp];

    discIneqbs = [vcmin[LOADS]; -vcmax[LOADS]];

    sm, discIneqs, discIneqbs
end

@everywhere function addDroopInequalities(sm, v, qg, xv, optRepair)
    droopIneqs = [];
    droopIneqbs = [];

    if optRepair == 1
        @constraint(sm, voltMGUB[i = 1:length(SITE), j = 1:length(RES)], v[SITE[i]] + mq * qg[SITE[i],RES[j]] <= vref + 2 * (1 - xv[SITE[i],RES[j]]) + mq * qgnom[RES[j]]);
        @constraint(sm, voltMGLB[i = 1:length(SITE), j = 1:length(RES)], v[SITE[i]] + mq * qg[SITE[i],RES[j]] >= vref - 2 * (1 - xv[SITE[i],RES[j]]) + mq * qgnom[RES[j]]);

        droopIneqs = [voltMGUB; voltMGLB];

        bBaseLB = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) + 2 * (ones(Float64, length(SITE), length(RES)));#
        bBaseUB = mq * repeat(transpose(qgnom[RES]),length(SITE),1) + vref * ones(Float64,length(SITE),length(RES)) - 2 * (ones(Float64, length(SITE), length(RES)));# - xv[SITE,RES,i]);
        droopIneqbs = [bBaseLB; bBaseUB];
    end

    sm, droopIneqs, droopIneqbs
end

@everywhere function addGenInequalities(sm, pg, qg, xv)
    x = zeros(AffExpr,N,length(RES));
    genPlaceIdx = zeros(Int64,G);
    genIdxSim = sum(xv, dims=2);

    @constraint(sm, pglb[i = 1:length(SITE), j = 1:length(RES)], pg[SITE[i],RES[j]] >= 0);
    @constraint(sm, qglb[i = 1:length(SITE), j = 1:length(RES)], qg[SITE[i],RES[j]] >= 0);
    @constraint(sm, pgub[i = 1:length(SITE), j = 1:length(RES)], pg[SITE[i],RES[j]] <= xv[SITE[i],RES[j]] * pgmax[SITE[i]]);
    @constraint(sm, qgub[i = 1:length(SITE), j = 1:length(RES)], qg[SITE[i],RES[j]] <= xv[SITE[i],RES[j]] * qgmax[SITE[i]]);
    genIneqs=[pglb; qglb; pgub; qgub];

    ovGen = repeat(transpose(ov[RES]),length(SITE));
    pgBase = repeat(pgmax[SITE],1,length(RES));
    qgBase = repeat(qgmax[SITE],1,length(RES));

    zeroB = repeat(ovGen,2,1);
    pgB = zeros(Float64, length(SITE), length(RES));
    qgB = zeros(Float64, length(SITE), length(RES));
    genIneqbs = [zeroB; pgB; qgB];

    sm, genIneqs, genIneqbs
end

@everywhere function addBasicEqualities(sm, v, vpar, p, q, pg, qg, beta, P, Q)
    basicEq =[]; basicEqb = [];

    @constraint(sm, realFlow[i in 1:N], P[i] == sum(CHILD[i,:] .* P) + p[i]);
    @constraint(sm, reacFlow[i in 1:N], Q[i] == sum(CHILD[i,:] .* Q) + q[i]);
    @constraint(sm, realCons[i in 1:N], p[i] - beta[i]*pcmax[i] + sum(pg[i,:]) == 0);
    @constraint(sm, reacCons[i in 1:N], q[i] - beta[i]*qcmax[i] + sum(qg[i,:]) == 0);

    basicEq = [realFlow; reacFlow; realCons; reacCons];
    basicEqb = [ov; ov; ov; ov];

    if length(LTC) > 0
        @constraint(sm, voltDropLTC[i in LTC], v[i] - LTCSetting * vpar[i] == 0);
        basicEq = [basicEq; voltDropLTC];
        basicEqb = [basicEqb; ov[LTC]];
    end

    sm, basicEq, basicEqb
end

@everywhere function addFailureInequalities(sm, ts, P, Q, vpar, v, kmval, y, yv)


    failureIneqs = []; failureIneqbs = [];

    @constraint(sm, pdislb[i = 1:N], P[i] >= -M*(1-kmval[i] + y[i] + sum(yv[:,1:ts-1],dims=2)[i]));
    @constraint(sm, qdislb[i = 1:N], Q[i] >= -M*(1-kmval[i] + y[i] + sum(yv[:,1:ts-1],dims=2)[i]));
    @constraint(sm, pdisub[i = 1:N], P[i] <= M*(1-kmval[i] + y[i] + sum(yv[:,1:ts-1],dims=2)[i]));
    @constraint(sm, qdisub[i = 1:N], Q[i] <= M*(1-kmval[i] + y[i] + sum(yv[:,1:ts-1],dims=2)[i]));

    failureIneqs = [pdislb; qdislb; pdisub; qdisub];

    bBase = M*(ones(Float64,N) - kmval + sum(yv[:,1:ts-1],dims=2));
    failureIneqbs = [-bBase; -bBase; bBase; bBase];


    @constraint(sm, vlb[i=1:length(noLTC)], vpar[noLTC[i]] - v[noLTC[i]] - resistance[noLTC[i]]*P[noLTC[i]] - reactance[noLTC[i]]*Q[noLTC[i]] >= -M * (kmval[noLTC[i]] - y[noLTC[i]] - sum(yv[noLTC[i],1:ts-1])));
    @constraint(sm, vub[i=1:length(noLTC)], vpar[noLTC[i]] - v[noLTC[i]] - resistance[noLTC[i]]*P[noLTC[i]] - reactance[noLTC[i]]*Q[noLTC[i]] <= M * (kmval[noLTC[i]] - y[i] - sum(yv[i,1:ts-1])));

    failureIneqs = [failureIneqs; vlb; vub];
    bBase = M * (kmval[noLTC] - sum(yv[:,1:ts-1], dims=2));
    failureIneqbs = [failureIneqbs; -bBase; bBase];

    sm, failureIneqs, failureIneqbs
end

@everywhere function addRepairConstraints(sm, ts, kmv, y, yv, Yk, optStage)
#
    repairEq = []; repairEqb = []; repairIneq = []; repairIneqb = [];

    repairStep = @constraint(sm, sum(y) == Yk);  #crew budget
    repairEq = [repairEq; repairStep];
    repairEqb = [repairEqb; Yk];

    for e = 1:N
        if kmv[e] == 1
            repairEdge = @constraint(sm, sum(y[e]) + sum(yv[e,1:ts-1]) <= 1);
            repairIneq = [repairIneq; repairEdge];
            repairIneqb = [repairIneqb; kmv[e] - sum(yv[e,1:ts-1])];
        end
    end

    if optStage == 1
        repairSub = @constraint(sm, y[1] == 0);
        repairEq = [repairEq; repairSub];
        repairEqb = [repairEqb; 0];
    elseif optStage == 2
        repairSub = @constraint(sm, y[1] == 1);
        repairEq = [repairEq; repairSub];
        repairEqb = [repairEqb; 1];
    end

    sm, repairIneq, repairIneqb, repairEq, repairEqb
end


@everywhere function printResultsSP(Ns, ov, uv, kcv, betav, Ts)

    totalCostNorm = zeros(Float64,Ts); totalCost = zeros(Float64,Ts);

    kc0 = ov; beta0 = uv; initialCost = (WSD-WLC)'*kc0 + WLC'*(ones(Float64,size(beta0))-beta0);

    for ts = 1:Ts
        costAtTs = (WSD-WLC)' * kcv[:,ts] + WLC' * (ones(Float64,size(betav[:,ts]))-betav[:,ts]);
        costAtTsMean = mean(costAtTs);
        totalCost[ts] = costAtTsMean;
        # ts == 1 ? totalCost = vcat(totalCost, costAtTsMean) : nothing;
    end
    # println("Total cost: ", totalCost);
    totalCostNorm = 100*(ones(Float64,size(totalCost)) - totalCost/sum(WSD));
    # println("System performance for scenario ", sceIdx, ": ", totalCostNorm[sce,:])

    totalCost, totalCostNorm
end

@everywhere function getDualVariables(basicIneq, basicIneqb, discIneq, discIneqb, failureIneqs, failureIneqbs, droopIneq, genIneq, Eqs, repairIneqs)

    ineqbar, Ineqbs = getIneqBar(basicIneq, discIneq, failureIneqs, basicIneqb, discIneqb, failureIneqbs);
    failureIneqbar = getFailureIneqBar(failureIneqs);
    basicIneqbar, discIneqbar = getBasicDiscIneqBar(basicIneq, discIneq);
    droopIneqbar = getDroopIneqBar(droopIneq);
    genIneqbar = getGenineqBar(genIneq);
    eqbar = getEqBar(Eqs);

    ineqbar, Ineqbs, failureIneqbar, basicIneqbar, discIneqbar, droopIneqbar, genIneqbar, eqbar
end

@everywhere function getIneqBar(basicIneq, discIneq, failureIneqs, basicIneqb, discIneqb, failureIneqbs);

    Ineqs = [basicIneq; discIneq];
    Ineqbs = [basicIneqb; discIneqb];
    # Ineqs = [basicIneq; discIneq; failureIneqs];
    # Ineqbs = [basicIneqb; discIneqb; failureIneqbs];
    ineqbar = zeros(Float64, length(Ineqs));

    for i = 1:length(Ineqs)
        ineqbar[i] = JuMP.dual(Ineqs[i]);
    end

    ineqbar, Ineqbs
end

@everywhere function getFailureIneqBar(failureIneqs)

    failureIneqbar = zeros(Float64, N*6);
    for i = 1:N*6
        failureIneqbar[i] = JuMP.dual(failureIneqs[i]);
    end

    failureIneqbar
end

@everywhere function getBasicDiscIneqBar(basicIneq, discIneq)

    basicIneqbar = zeros(Float64, N*2 + length(LOADS)*2);
    discIneqbar = zeros(Float64, length(LOADS)*2);
    for i = 1:length(LOADS)*2
        discIneqbar[i] = JuMP.dual(discIneq[i]);
    end

    for i = 1:N*2+length(LOADS)*2
        basicIneqbar[i] = JuMP.dual(basicIneq[i]);
    end

    basicIneqbar, discIneqbar
end

@everywhere function getDroopIneqBar(droopIneq)

    droopIneqbar = zeros(Float64, size(droopIneq));
    sizeDroopIneq = size(droopIneq);
    for i = 1:sizeDroopIneq[1]
        for j = 1:sizeDroopIneq[2]
            droopIneqbar[i,j] = JuMP.dual(droopIneq[i,j]);
        end
    end

    droopIneqbar
end

@everywhere function getGenineqBar(genIneq)

    genIneqbar = zeros(Float64, size(genIneq));
    sizeGenIneq = size(genIneq);
    for i = 1:sizeGenIneq[1]
        for j = 1:sizeGenIneq[2]
            genIneqbar[i,j] = JuMP.dual(genIneq[i,j]);
        end
    end

    genIneqbar
end

@everywhere function getEqBar(Eqs)

    eqbar = zeros(Float64, size(Eqs));
    sizeEqs = size(Eqs);
    for i = 1:sizeEqs[1]
        eqbar[i] = JuMP.dual(Eqs[i]);
    end

    eqbar
end

@everywhere function getRepairIneqBar(repairIneqs, noFailures)

    repairIneqbar = zeros(Float64, size(repairIneqs));
    sizeRepairIneqs = size(repairIneqs);
    for i = 1:sizeRepairIneqs[1]
        repairIneqbar[i] = JuMP.dual(repairIneqs[i]);
    end
    repairIneqLine = repairIneqbar;

    repairIneqLine
end

@everywhere function getRepairEqbar(repairEqs, noFailures, optStage)

    noFailures = trunc(Int(noFailures));
    repairEqbar = zeros(Float64, size(repairEqs));
    sizeRepairEqs = size(repairEqs);
    for i = 1:sizeRepairEqs[1]
        repairEqbar[i] = JuMP.dual(repairEqs[i]);
    end

    if optStage == 1
        repairEqPeriod = repairEqbar;
        repairEqSubstation = 0;
    elseif optStage == 2
        repairEqPeriod = repairEqbar[1:noFailures];
        repairEqSubstation = repairEqbar[length(repairEqbar)];
    end

    repairEqbar, repairEqPeriod, repairEqSubstation
end

@everywhere function addBendersCutFailures(kmval, ts, y, yv)
    bBaseBenders = M*(ones(AffExpr,N) - kmval);
    for i = 1:N
        bBaseBenders[i] = bBaseBenders[i] + M*y[i] + M*sum(yv[i,:]);# + M*sum(yv[i,ts+1:T]);
    end
    failureIneqbsBenders = [-bBaseBenders; -bBaseBenders; bBaseBenders; bBaseBenders];

    bBaseBenders = zeros(AffExpr, N);
    for i = 1:N
        bBaseBenders[i] = M * (kmval[noLTC[i]] - y[i] - sum(yv[i,:]));# - sum(yv[i,ts+1:T]));
    end

    failureIneqbsBenders = [failureIneqbsBenders; -bBaseBenders; bBaseBenders];

    failureIneqbsBenders
end

@everywhere function addBendersCutRepairs(idxFail, y, yv, ts)

    noFailures = length(idxFail);
    repairEqbsBenders = [sum(y); y[1]];
    repairIneqbsBenders = zeros(AffExpr, noFailures);
    for i = 1:noFailures
        repairIneqbsBenders[i] = 1 - y[idxFail[i]] - sum(yv[idxFail[i],:]);# +  + sum(yv[idxFail[i],ts+1:T]);
    end

    repairIneqbsBenders
end

@everywhere function addBendersCutThetaRepairs(kmv, idxFail, yvPre, ts);

    repairIneqbsBdCut = zeros(Float64, length(idxFail));
    for i = 1:length(idxFail)
        # repairIneqbsBdCut[i] = 1 - (kmv[idxFail[i]] - sum(yvPre[idxFail[i],:]));
        repairIneqbsBdCut[i] = kmv[idxFail[i]] - sum(yvPre[idxFail[i],:]);
    end

    repairIneqbsBdCut
end

@everywhere function addBendersCutThetaFailures(kmv, idxFail, yvPre, ts);

    kmvPeriod = zeros(Int64, N);
    kmvPeriod = kmv - sum(yvPre, dims = 2);

    bbase = zeros(Float64, N); bbase2 = zeros(Float64, N);
    for i = 1:N
        if kmvPeriod[i] == 0
            bbase[i] = 10;
        elseif kmvPeriod[i] == 1
            bbase2[i] = 10;
        end
    end

    failureIneqbsBdCut = [-bbase; -bbase; bbase; bbase; -bbase2; bbase2];

    failureIneqbsBdCut
end

@everywhere function getRepairEqbar(repairEqs, noFailures)

    repairEqbar = zeros(Float64, size(repairEqs));
    sizeRepairEqs = size(repairEqs);
    for i = 1:sizeRepairEqs[1]
        repairEqbar[i] = JuMP.dual(repairEqs[i]);
    end
    repairEqFirstPeriod = repairEqbar[1:length(repairEqbar)-1];
    repairEqSubstation = repairEqbar[length(repairEqbar)];

    repairEqbar, repairEqFirstPeriod, repairEqSubstation
end

@everywhere function getReducedCostsLoads(basicIneqbar, discIneqbar)

    rc_kc = WSD[LOADS]-WLC[LOADS];
    basicIneqbar_lb = basicIneqbar[1:length(LOADS)];
    basicIneqbar_ub = basicIneqbar[length(LOADS)+1:2*length(LOADS)];
    discIneqbar_lb = discIneqbar[1:length(LOADS)];
    discIneqbar_ub = discIneqbar[length(LOADS)+1:2*length(LOADS)];

    # rc_kc = rc_kc - (betamin[LOADS].*basicIneqbar_lb + basicIneqbar_ub - discIneqbar_lb + discIneqbar_ub);
    rc_kc = rc_kc - (betamin[LOADS].*basicIneqbar_lb + basicIneqbar_ub + discIneqbar_lb + discIneqbar_ub);

    rc_kc
end

@everywhere function getReducedCostsResource(fixed1v, fixed2v, siteDerV, derOnceV, derSumV);

    rc_xs = zeros(Float64, length(SITE)); rc_x = zeros(Float64, length(SITE), length(RES));

    for i = 1:length(SITE)
        rc_xs[i] = -(fixed1v[i] - sum(siteDerV[i,:]));
        for j = 1:length(RESm)
            rc_x[i,j] = -(fixed2v[i,RES[j]] + siteDerV[i,RES[j]] + derOnceV[RES[j]] + derSumV);
        end
    end

    rc_xs, rc_x
end

@everywhere function getReducedCostsRepairs(repairIneqLine, repairEqPeriod, repairEqSubstation, failureIneqbar, km0, Ts)

    noFailures = sum(km0);
    rc_yl = zeros(Float64, noFailures, Ts);
    for i = 1:noFailures
        for j = 1:Ts
            if i == 1 && j < Ts
                rc_yl[i,j] = repairIneqLine[i] + repairEqPeriod[j];
            elseif i == 1 && j == Ts
                rc_yl[i,j] = repairIneqLine[i] + repairEqPeriod[j] + repairEqSubstation;
            elseif i > 1
                rc_yl[i,j] = repairEqPeriod[j] + repairIneqLine[i];
            end
        end
    end
    # rc_yl[1,T] = rc_yl[1,T] + repairEqSubstation;

    idx = findall(km0 .== 1);

    failureIneq_pLb = failureIneqbar[1:N,:];
    failureIneq_qLb = failureIneqbar[N+1:2*N,:];
    failureIneq_pUb = failureIneqbar[2*N+1:3*N,:];
    failureIneq_qUb = failureIneqbar[3*N+1:4*N,:];
    failureIneq_vLb = failureIneqbar[4*N+1:5*N,:];
    failureIneq_vUb = failureIneqbar[5*N+1:6*N,:];
    for i = 1:noFailures
        for j = 1:Ts
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_pLb[idx[i][1],j:Ts]) + M*sum(failureIneq_pUb[idx[i][1],j:Ts]);
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_qLb[idx[i][1],j:Ts]) + M*sum(failureIneq_qUb[idx[i][1],j:Ts]);
            rc_yl[i,j] = rc_yl[i,j] - M*sum(failureIneq_vLb[idx[i][1],j:Ts]) + M*sum(failureIneq_vUb[idx[i][1],j:Ts]);
        end
    end
    rc_yl = -rc_yl;

    rc_yl
end

@everywhere function getMasterModel(G, Ns, kmv)
    mm = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0));
    @variable(mm, xvar[1:length(SITE),1:length(RES)],Bin);
    @variable(mm, xsvar[1:length(SITE)],Bin);
    @variable(mm, theta[1:Ns]);
    @variable(mm, kcm[1:Ns,1:length(LOADS)],Bin);

    x  = zeros(AffExpr,N,Nres);
    x[SITE,RES] = xvar;
    xs  = zeros(AffExpr,N); xs[SITE] = xsvar;

    @constraint(mm, sum(x[i,j] for i in SITE, j in RES) <= G);

    # 1 DG is allocated to at most 1 site
    @constraint(mm, [j in RES], sum(x[i,j] for i in SITE) <= 1);

    # if a site is operated, at least 1 DG is allocated
    @constraint(mm, [i in SITE], xs[i] <= sum(x[i,j] for j in RES));

    # if a DG is operated at a node, a site is operated
    @constraint(mm, [i in SITE], sum(x[i,j] for j in RES) <= Nres * xs[i]);

    # No mobile DGs allocated
    @constraint(mm, [j in RESm], sum(x[i,j] for i in SITE) == 0);

    @constraint(mm, [i in 1:Ns], theta[i] >= 0);

    @objective(mm, Min, sum(WSITE[SITE].*xsvar) + sum(theta)/Ns);

    mm, x, xs, kcm, theta
end

@everywhere function getResultsMaster(x, xs)
    xv = JuMP.value.(x);
    xsv = JuMP.value.(xs);

    genIdxSim = sum(xv, dims = 2);
    idx = findall(genIdxSim -> genIdxSim .> 0,genIdxSim);

    sizeXv = size(xv);
    for i = 1:length(xsv)
        xsv[i] = trunc(Int, round(xsv[i]));
        for j = 1:sizeXv[2]
            xv[i,j] = trunc(Int, round(xv[i,j]));
        end
    end

    xv, xsv, genIdxSim
end

@everywhere function getBendersCascadeSceIter(km0, B, G, CB, opt, sce, iter, xs, x, xv, xsv)

    start = time();

    # Get number of repairs at each iteration
    nFailed = sum(km0[:,sce]); Yk = zeros(Int64,T); Yk[1] = 0; Yk[T] = 1;
    for i = 2:T-1
        Yk[i] = minimum([CB; nFailed-1]);
        nFailed -= Yk[i];
    end

    # FORWARD PASS
    println("-------------------------------------------------------------")
    println("FORWARD PASS")

    objSlave = zeros(Float64, T); yv = zeros(Int64,N,T);
    xsvPeriod = zeros(Int64, N, T); xvPeriod = zeros(Int64, N, G, T);
    dual = zeros(Float64, T); dualVar = zeros(Float64, 7*N+2, T);
    siteDev = zeros(Float64, length(SITE) + length(RES) + 1, T); derAlloc = zeros(Float64, length(SITE), length(RES), T);
    dualTotal = zeros(Float64, T); kcvCollect = zeros(Float64, N, T);

    derRealloc = zeros(Int64, T);
    reallocPeriod = trunc(Int, floor(T/2)); derRealloc[reallocPeriod+1] = 1;
    # println("Reallocation period: ", derRealloc)

    for i = 1:T-1
        println("Period ", i)
        start = time();

        # Get MILP solution (objective, repairs, resource allocations)
        param = [G; Yk[i]; i];
        opt = [0; 0; 0; 0; derRealloc[i]; sce; iter; derRealloc[i+1]];

        if i > 1
            xvInput = xvPeriod[:,:,i-1]; xsvInput = xsvPeriod[:,i-1];
        else
            xvInput = xv; xsvInput = xsv;
        end

        objSlave[i], yv[:,i], xsvPeriod[:,i], xvPeriod[:,:,i], u1, u2, u3, u4, u5, kcv, u6, u7 = getPeriodSubproblem(km0[:,sce], param,xsvInput, xvInput, opt, yv, [], [], [], [], []);

        objSlaveCollect[i,sce,iter] = objSlave[i]; kcvCollect[:,i] = kcv;
        # println("Repairs: ", findall(yv[:,i] .== 1))
        # println("Placements: ", xvPeriod[SITE,RES,i])

        # Get LP solution (dual variables, dual cost vector components, etc. to form Benders cut on backwards pass)
        if derRealloc[i] == 1
            derReallocSoln[:,:,sce,iter] = xvPeriod[:,:,i];
        end

        if i > 1
            yvInput = yv[:,1:i-1];
            xvInput = xvPeriod[:,:,i-1]; xsvInput = xsvPeriod[:,i-1];
        else
            yvInput = zeros(Int64,N,1); xvInput = xv; xsvInput = xsv;
        end

        opt = [1; 0; 0; 0; trunc(Int,derRealloc[i]); sce; iter; derRealloc[i+1]];
        ctg, u2, u3, u4, dual[i], dualVar[:,i], siteDev[:,i], derAlloc[:,:,i], dualTotal[i], kcv, u5, u6 = getPeriodSubproblem(km0[:,sce], param, xsvInput, xvInput, opt, yvInput, [], [], [], [], kcv);

        if i == T-1
            costToGo[i,sce,iter] = ctg;
        end

        timeElapsed = time() - start;
        println(timeElapsed)

#         println("---------------------")
    end

    dualForward = dual;

    # BACKWARD PASS
    println("-------------------------------------------------------------")
    println("BACKWARD PASS")

    dualConstants = zeros(Float64, T); dualConstants[T-1] = dual[T-1]; rc_kcOut = 0;
    kcvOutput = [];
    for i = 1:T-2
        # Get MILP solution with Benders cuts (record resource allocations w/ Benders cuts)
        println("Period ", T-1-i)
        start = time();

        opt = [0; 0; 0; 1; derRealloc[T-1-i]; sce; iter; derRealloc[T-i]];
        param = [G; Yk[T-1-i]; T-1-i];

        if T-1-i > 1
            xvInput = xvPeriod[:,:,T-2-i]; xsvInput = xsvPeriod[:,T-2-i];
        else
            xvInput = xv; xsvInput = xsv;
        end

        u1, u2, u3, xvPeriodBckw, u5, u6, u7, u8, u9, kcv, u10, u11 = getPeriodSubproblem(km0[:,sce], param, xsvInput, xvInput, opt, yv[:,1:T-2-i], dual[T-i], dualVar[:,T-i], siteDev[:,T-i], derAlloc[:,:,T-i], []);
#         println("---------------------")

        if derRealloc[T-1-i] == 1
            derReallocSolnBD[:,:,sce,iter] = xvPeriodBckw;
        end

        # Get LP solution with Benders cuts (along with updated dual variables, dual cost vectors, reduced costs etc.)
        opt = [1; 0; 0; 1; derRealloc[T-1-i]; sce; iter; derRealloc[T-i]];
        dualTotal[T-1-i], u2, u3, u4, dualConstants[T-1-i], dualVar[:,T-1-i], siteDev[:,T-1-i], derAlloc[:,:,T-1-i], dt, kcv, rc_kc, u5 = getPeriodSubproblem(km0[:,sce], param, xsvInput, xvInput, opt, yv[:,1:T-2-i], dual[T-i], dualVar[:,T-i], siteDev[:,T-i], derAlloc[:,:,T-i], kcv);
        dual[T-1-i] = dualConstants[T-1-i];

        costToGo[T-1-i,sce,iter] = dualTotal[T-1-i];
        kcvOutput = kcv; rc_kcOut = rc_kc;
#         println("---------------------")

        timeElapsed = time() - start;
        println(timeElapsed)
    end
    println("-------------------------------------------------------------")

    # Collect relevant outputs
    dualConstantCollect[sce,:,iter] = dual;
    dualVarCollect[sce,:,:,iter] = dualVar;
    siteDevCollect[sce,:,:,iter] = siteDev;
    derAllocCollect[sce,:,:,:,iter] = derAlloc;
    yCollect[sce,:,:,iter] = yv;

    # Get total objective
    objSlave = sum(objSlave);

    # Form Benders cut constant term for MP
    bendersCutLS = dualTotal[1] + sum(kcvOutput[LOADS].*rc_kcOut);
    bendersCutLS -= (sum(xsvPeriod[SITE,1] .* siteDev[1:length(SITE),1]) + sum(xvPeriod[SITE,RESf,1].*derAlloc[1:length(SITE), 1:length(RESf), 1]));

    # Get relevant dual variables for MP Benders cut
    siteDev = siteDev[1:length(SITE),1]; derAlloc = derAlloc[1:length(SITE),1:length(RES),1];

    objSlave, bendersCutLS, siteDev, derAlloc
end

@everywhere function addObjBDcut(tupleA, tupleB)
    (objSlaveA, bendersCutLSA, siteDevSumA, derAllocSumA) = tupleA;
    (objSlaveB, bendersCutLSB, siteDevSumB, derAllocSumB) = tupleB;

    ([objSlaveA; objSlaveB], [bendersCutLSA; bendersCutLSB], [siteDevSumA; siteDevSumB], [derAllocSumA; derAllocSumB])
end

@everywhere function getBendersCascadeIteration(mm, x, xs, kcm, theta, kmv, G, opt, iter)

    isBendersCutAdded = false; isMasterModelNotSolvable = false;

    if verbose
        println(mm);
    end

    optimize!(mm);
    mstatus = termination_status(mm);
    objMaster = JuMP.objective_value(mm);
    xv, xsv, genIdxSim = getResultsMaster(x, xs);

    if verbose
        printResultsMaster(xv, xsv, genIdxSim);
    end

    bdCut = 0; objvSum = 0;
    (objvSum, bdCut, siteDevSum, derAllocSum) = @distributed addObjBDcut for sce =  1:Ns
        objSlave, bendersCutLS, siteDev, derAlloc = getBendersCascadeSceIter(km0, B, G, CB, opt, sce, iter, xs, x, xv, xsv);
        (objSlave, bendersCutLS, siteDev, derAlloc)
    end

    epsilon = 0.1;
    for sce = 1:Ns
        idx1 = length(SITE)*(sce-1)+1; idx2 = length(SITE)*sce;
        bdCutAdd = bdCut[sce] + sum(xs[SITE] .* siteDevSum[idx1:idx2]) + sum(x[SITE,RESf] .* derAllocSum[idx1:idx2,RESf]) - sum(x[SITE,RESm] .* derAllocSum[idx1:idx2,RESm]);
        bdCutAdd += sum(kcm[sce,:] .* rcCollect[sce,:,1,iter]) + epsilon;
        # println("MP Benders cut: ", bdCutAdd)
        @constraint(mm, theta[sce] >= bdCutAdd);
        if iter == 1
            ac = getConnectivitySet(km0[:,sce]);
            @constraint(mm, [i=1:length(LOADS)], kcm[sce,i] >= 1 - sum(xs[SITE] .* ac[i,:]));
        end
    end
    isBendersCutAdded = true;

    objLB = objMaster; objUB = mean(objvSum) + sum(WSITE[SITE].*xsv[SITE]);
    println("Placement: ", xv[SITE,RES]);

    mm, isBendersCutAdded, isMasterModelNotSolvable, objLB, objUB, xv, xsv
end

@everywhere function getBendersMethodCascade(kmv, G, opt)
    printSummary = true; nCurrentIter = 0;
    optSP = opt[6]; optMobile = opt[7];

    mm, x, xs, kcm, theta = getMasterModel(G, Ns, kmv);

    println("-------------------------------------------------------------")
    println("Entering loop");
    ObjUB = Inf; ObjLB = -Inf; nCurrentIter = 0;
    LBs = zeros(Float64,jmax); UBs = zeros(Float64,jmax);
    obj = zeros(Float64,jmax);
    xv = zeros(Int64, jmax, N, G); xsv = zeros(Int64, jmax, N);
    addGreedy = zeros(Float64,jmax);
    for j = 1:jmax
        nCurrentIter += 1; println("Iteration: ", nCurrentIter)

        mm, isBendersCutAdded, isMasterModelNotSolvable, LBs[j], obj[j], xv[j,:,:], xsv[j,:] = getBendersCascadeIteration(mm, x, xs, kcm, theta, kmv, G, opt, j);

        if j > 1 && sum(xv[j,:,:] .== xv[j-1,:,:]) == N*Nres
            println("ADD GREEDY CONSTRAINT")
            idxPlace = findall(xv[j,:,:] .== 1);
            @constraint(mm, sum(x[idxPlace]) <= sum(xv[j,:,:])-1);
            addGreedy[j] = 1;
        end

        UBs[j] = minimum(obj[1:nCurrentIter]);
        println("Lower bound: ", LBs[j]); println("Upper bound: ", UBs[j]);
        println("-------------------------------------------------------------")
        println("-------------------------------------------------------------")

        if isMasterModelNotSolvable
            LBs[j] = -Inf; UBs[j] = Inf;
            println("Exit loop")
            break
        end

        if (UBs[j]-LBs[j])/LBs[j] < 0.01
            println("Exit loop")
            break
        end
    end

    idx = argmax(-obj[1:nCurrentIter]);
    xv = xv[1:nCurrentIter,:,:]; xsv = xsv[1:nCurrentIter,:];
    objValSim = obj[1:nCurrentIter];

#     println("-------------------------------------------------------------")
#     println("-------------------")

    LBs = LBs[1:nCurrentIter]; UBs = UBs[1:nCurrentIter];

    xvOpt = xv[idx,:,:]; xsvOpt = xsv[idx,:];
    sysPerf = zeros(Float64,Ns,T+2); objVal = zeros(Float64, Ns);
    for i = 1:Ns
        objVal[i], sysPerf[i,:] = getSolutionRES_MILP_Greedy(km0[:,i], G, CB, xvOpt, xsvOpt, i);

        # println("System performance: ", sysPerf[i,:]);
    end

    sysPerf = mean(sysPerf, dims = 1);
    objValMean = mean(objVal);

    nCurrentIter, xsvOpt, xvOpt, objValSim, LBs, UBs, sysPerf, idx, objValMean
end

@everywhere function getSolutionRES_MILP_Greedy(kmv, G, Y, xvOpt, xsvOpt, sce)

    # Get number of repairs at each iteration
    nFailed = sum(km0[:,sce]); Yk = zeros(Int64,T); Yk[1] = 0; Yk[T] = 1;
    for i = 2:T-1
        Yk[i] = minimum([CB; nFailed-1]);
        nFailed -= Yk[i];
    end

    kmv0 = kmv;
    objGreedy = zeros(Float64, T);
    kcv = zeros(Float64, N, 1, T); betav = zeros(Float64, N, 1, T);
    yv = zeros(Float64, N, T);
    xv = zeros(Float64, N, Nres, T); xsv = zeros(Float64, N, T);
    kcv = zeros(Float64, N, T); betav = zeros(Float64, N, T);

    xvIter = xvOpt; xsvIter = xsvOpt;
    derRealloc = zeros(Int64, T);
    # println("-------------------------")
    for i = 1:T-1
        param = [G; Yk[i]; i];
        opt = [0; 0; 0; 0; derRealloc[i]; sce; 1; derRealloc[i+1]];
        objGreedy[i], kmv0, yv[:,i], xvOut, xsvOut, kcv[:, i], betav[:, i] = getSystemPerformanceGreedy(kmv0, G, Y, i, xvIter, xsvIter);

        # objGreedy[i], yv[:,i], xsvOut, xvOut, u1, u2, u3, u4, u5, kcv[:,i], u6, betav[:,i] = getPeriodSubproblem(km0[:,sce], param, xsvIter, xvIter, opt, yv, [], [], [], [], []);

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
    println("System performance: ", sysPerf)

    objVal = sum(objGreedy);

    objVal, sysPerf
end

@everywhere function getSystemPerformanceGreedy(kmv, G, Y, tstep, xv, xsv)

    smg = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=0, MIPGap = 1e-10));

    # Variables
    smg, p, q, pg, qg, beta, P, Q, v, v0, vpar, t = getSlaveVariables(smg, 1, 1);

    y = zeros(AffExpr,N,1,1); nFailed = sum(kmv);
    idxRepair = findall(kmv .== 1);
    y[idxRepair,1,1] = @variable(smg, [1:nFailed,1,1], Bin);

    @variable(smg, kcvar[1:length(LOADS),1,1], Bin);
    kc = zeros(AffExpr,N,1,1); kc[LOADS,:,:] = kcvar;

    smg, xs, x, xr = getMobileVariables(smg, 0);

    # Constraints
    smg, genIneq, genIneqb = addGenInequalitiesGreedy(smg, pg, qg, x, 1);
    smg, droopIneq, droopIneqb = addDroopInequalitiesGreedy(smg, v, qg, 1:1:T, x, 1);
    smg, basicIneq, basicIneqb = addBasicInequalitiesGreedy(smg, beta, kc, v, t, 1);
    smg, discIneq, discIneqb = addDisconnectivityConstraintsGreedy(smg, kc, v, 1);
    smg, failureIneqs, failureIneqbs = addFailureInequalitiesGreedy(smg, P, Q, vpar, v, kmv, y, 1);
    smg, basicEq, basicEqb = addBasicEqualitiesGreedy(smg, v, vpar, p, q, pg, qg, beta, P, Q, 1);

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

@everywhere function printResultsSP(ov, uv, kcv, betav)

    kc0 = ov; beta0 = uv; initialCost = (WSD-WLC)'*kc0 + WLC'*(ones(Float64,size(beta0))-beta0);
    totalCost = [initialCost; initialCost];

    for ts = 1:T
        costAtTs = (WSD-WLC)' * kcv[:,ts] + WLC' * (ones(Float64,size(betav[:,ts]))-betav[:,ts]);
        if ts == T
            costAtTs = 0;
        end
        totalCost = [totalCost; costAtTs];

    end
    sysPerf = 100*(ones(Float64,size(totalCost)) - totalCost/sum(WSD));

    sysPerf
end

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


@everywhere function getMobileVariablesGreedy(sm, Ns, LP)
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

@everywhere function addGenInequalitiesGreedy(sm, pg, qg, x, T)

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
@everywhere function addDroopInequalitiesGreedy(sm, v, qg, ts, x, T)

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
@everywhere function addBasicInequalitiesGreedy(sm, beta, kcval, v, t, T)
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
@everywhere function addDisconnectivityConstraintsGreedy(sm, kcval, v, T)
    @constraint(sm, loadLow[i=1:length(LOADS),j=1:T], v[LOADS[i],1,j]  >= vcmin[LOADS[i]] - kcval[LOADS[i],1,j]);
    @constraint(sm, loadUp[i=1:length(LOADS),j=1:T], -v[LOADS[i],1,j]  >=  -vcmax[LOADS[i]] - kcval[LOADS[i],1,j]);

    discIneqs = [loadLow; loadUp];

    discIneqbs = [repeat(vcmin[LOADS],1,T); repeat(-vcmax[LOADS],1,T)];

    sm, discIneqs, discIneqbs
end

@everywhere function addFailureInequalitiesGreedy(sm, P, Q, vpar, v, kmval, y, T)
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
@everywhere function addBasicEqualitiesGreedy(sm, v, vpar, p, q, pg, qg, beta, P, Q, T)
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

@everywhere function addObjectiveSlaveModelGreedy(smg, t, beta, kcval, P, xs, xr, xsv)

    costP = sum(sum(WAC) * P[1,1,1]);
    costT = sum(WVR * sum(t[:,1,1])/N);
    costLC = sum(WLC[LOADS] .* (ones(Float64,length(LOADS),1,1)-beta[ LOADS,1,1]) );
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
