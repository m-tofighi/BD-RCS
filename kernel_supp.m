function [Ksupp, Ind, Im, H, M] = kernel_supp(mat,vec,conv_wx,CC,BB,maxrank,...
    CCT,BBT,ZInit,HInit,lambda,Indw,Ind,maxOutIter,num_out_iter,threshold_ratio,...
    threshold_ratio_image,L,K,blur_kernel,waveLevel,bookkeeping,Csize,C,wavelet,sparsity)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  function Ksupp = kernel_supp(mat,vec,conv_wx,CC,BB,maxrank,CCT,BBT,
    %  ZInit,HInit,lambda,Indw,maxOutIter,num_out_iter,threshold_ratio,L,K,blur_kernel)
    %
    %   Written by Mohammad Tofighi <tofighi@psu.edu>
    %   Last Updated:  August 10, 2017

    %   Tries to find kenrel support
    %   Uses the method of multipliers to solve the algorithm
    %
    %   minimize ||X||_* + \lambda ||X||_{2,1}
    %   subject to yhat = Ahat(X)

L1 = L(1);
L2 = L(2);
K1 = K(1);
K2 = K(2);
for iter = 1: num_out_iter
    iter
    [M,H] = blindDeconvolve_implicit_2D(vec(conv_wx),CC,BB,maxrank,CCT,BBT,ZInit,HInit,lambda,maxOutIter,Csize,sparsity);%1=>4
    support_h = abs(H(:,1))>threshold_ratio*mean(abs(H(:,1)));
    support_m = abs(M(:,1))>threshold_ratio_image*mean(abs(M(:,1)));
    
    [UM,SM,VM] = svd(M,'econ');
    [UH,SH,VH] = svd(H,'econ');
    [U2,S2,V2] = svd(SM*VM'*VH*SH);
    mEst=sqrt(S2(1,1))*UM*U2(:,1);
    
    hEstt = zeros(K1*K2,1);
    Indw_map1 = reshape(Indw, [L1, L2]);
    hEstt(Indw_map1(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2))=H(:,1);
    
    if sparsity == 1 %update B
        ZInit = M;
        HInit = abs(H(support_h~=0,:));
        Indw(Indw) = support_h;
        j = 1;
        Kk = nnz(Indw);
        Ll = L1*L2;
        B = sparse(Ll,Kk);
        for i = 1:Ll
            if(Indw(i) == 1)
                B(i,j) = Indw(i);
                j = j+1;
            end
        end
        BB = @(x) mat(B*x);
        BBT = @(x) B'*vec(x);
        figure(2),subplot(121), imagesc(abs(reshape(hEstt, size(blur_kernel)))), ...
            title(sprintf('Estimated PSF; iter: %d', iter)), colormap(gray)
        subplot(122), imagesc(blur_kernel), title('Original PSF'), colormap(gray)
        pause(0.001)
    elseif sparsity == 2 %update C
        HInit = hEstt(:);
        alpha = mEst';
        Ind(Ind) = support_m;
        j = 1;
        Kk = nnz(Ind);
        N = sum(support_m);
        C = sparse(length(Ind),N);
        %mbb = zeros(size(Ind));
        mbb = alpha(support_m);
        for i = 1:length(Ind)
            if(Ind(i) == 1)
                C(i,j) = Ind(i);
                %mbb(j) = alpha(i);
                j = j+1;
            end
        end
        ZInit = mbb';
        CC = @(x) waverec2(C*x,bookkeeping,wavelet);
        CCT = @(x) (C'*(wavedec2(x,waveLevel,wavelet))');
        figure(2)
        map = jet;
        wavelet_coefficients = C*mbb';
        colormap(map); rv = length(map);
        subplot(122),plotwavelet2(wavelet_coefficients,bookkeeping,waveLevel,wavelet,rv,'square');
        pause(0.001)
        title(['Iteration no: ',num2str(iter)]);
    end

    lambda = (1/sqrt(Kk))*lambda;
end

Ksupp = hEstt;
Im = mEst(support_m);
