function [mval,g]=subproblem_cost(x,C1,C2,C1T,C2T,maxrank,meas_fft,y,sigma,siglen,n1,n2,lambda,w)
    % This helper function computes the value and gradient of the augmented
    % Lagrangian  
    %   Written by Mohammad Tofighi <tofighi@psu.edu>
    %   Last Updated:  August 10, 2017

    p1 = sqrt(siglen);
    p2 = p1;

    Z = reshape(x(1:(n1*maxrank)),n1,maxrank);
    H = reshape(x((n1*maxrank) + (1:(n2*maxrank))),n2,maxrank);
    mat = @(x) reshape(x,p1,p2);
    vec = @(x) x(:);
    % compute equation error
    dev = zeros(siglen,1);
    for i = 1:maxrank
        dev = dev+vec(fft2(C1(Z(:,i))).*fft2(C2(H(:,i))))/siglen;
    end    
    dev = dev - meas_fft;
    % compute the cost of the extended Lagrangian penalty function
    mval = norm(Z,'fro')^2+norm(H,'fro')^2 - 2*real(y'*dev) + sigma*norm(dev,'fro')^2 + lambda^2*sum(sum((H*Z').^2, 2)./w); 
%     mval = 2*real(y'*dev) + sigma*norm(dev,'fro')^2 + lambda^2*sum(sum((H*Z').^2, 2)./w); 

    % compute the gradient
    yhat = y - sigma*dev;
    temp1 = zeros(siglen,maxrank);
    temp2 = zeros(siglen,maxrank);
    for i = 1:maxrank 
        temp1(:,i) = vec(fft2(C2(H(:,i))));
        temp2(:,i) = vec(ifft2(C1(Z(:,i))));
    end
    temp3 = conj((yhat)*ones(1,maxrank)).*temp1;
    temp4 = (yhat)*ones(1,maxrank).*temp2;
    temp5 = zeros(n1,maxrank);
    temp6 = zeros(n2,maxrank);
    for i = 1:maxrank
        temp5(:,i) =  C1T(fft2(mat(temp3(:,i))));
        temp6(:,i) = C2T(ifft2(mat(temp4(:,i))));
    end
    adjoint_times_H = temp5/siglen;
    adjoint_times_Z = temp6*siglen;

    w_inv = 1./w;
    GradZ = 2*(Z - adjoint_times_H + lambda^2*Z*H'*bsxfun(@times, w_inv, H));
    GradH = 2*(H - adjoint_times_Z + lambda^2*bsxfun(@times, w_inv, H*Z'*Z));

    g = real([GradZ(:);GradH(:)]);
