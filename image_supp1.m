function [mEst, hEst, BB, H, M] = image_supp1(hEstt,conv_wx,vec,mat,CCT,CC,K,L,N,maxInOutIter)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  function Ksupp = image_supp(hEstt,conv_wx,vec,mat,CCT,CC,K,L,N,maxInOutIter)
    %
    %   Written by Mohammad Tofighi <tofighi@psu.edu>
    %   Last Updated:  August 10, 2017

    %   Tries to find image support
    %   Uses the method of multipliers to solve the algorithm
    %
    %   minimize ||X||_* + \lambda ||X||_{1,2}
    %   subject to yhat = Ahat(X)

L1 = L(1);
L2 = L(2);
K1 = K(1);
K2 = K(2);
w = zeros(L1,L2);
w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = abs(reshape(hEstt, K));
w_vec = vec(w);
Ll = L1*L2;
Indw = zeros(Ll,1);
Indw = abs(w_vec)>0;
j = 1;
Kk = numel(hEstt);%nnz(Indw);
B = sparse(Ll,Kk);
h = zeros(Kk,1);
for i = 1:Ll
    if(Indw(i) == 1)
        B(i,j) = Indw(i);
        h(j) =w_vec(i);
        j = j+1;
    end
end
BB = @(x) mat(B*x);
BBT = @(x) B'*vec(x);
%==============

% Initialize Z and H
maxrank = 1;
ZInit = 1e-2*randn(N,maxrank);%ones(N,maxrank);%mb;
HInit = 1e-2*randn(Kk,maxrank);%ones(K,maxrank);%blur_kernel(:);%h;
[M,H] = blindDeconvolve_implicit_2D_V0(vec(conv_wx),CC,BB,maxrank,CCT,BBT,ZInit,HInit,maxInOutIter);%1=>4

[UM,SM,VM] = svd(M,'econ');
[UH,SH,VH] = svd(H,'econ');
[U2,S2,V2] = svd(SM*VM'*VH*SH);

%% Estimates of m and h and recovery errors

mEst=sqrt(S2(1,1))*UM*U2(:,1);
hEst=sqrt(S2(1,1))*UH*V2(:,1);