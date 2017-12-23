function [Z,H] = blindDeconvolve_implicit_2D_V0(conv_zh,C1,C2,maxrank,C1T,C2T,ZInit,HInit,maxInOutIter)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  Solving problem 6 in paper
    %  function [Z,H] = blindDeconvolve(conv_zh,C1,C2,maxrank,C1T,C2T,ZInit,HInit,maxInOutIter)
    %
    %   Written by Mohammad Tofighi <tofighi@psu.edu>
    %   Last Updated:  August 10, 2017
    %   Modified from BlindDeconvolve file written by Ali Ahmed <alikhan@gatech.edu>.

    %  Tries to find matrices Z and H such that
    %       Z is n1 x maxrank
    %       H is n2 x maxrank
    %
    %       conv_zh = sum_k circular_conv(Z(:,k),H(:,k))
    %
    %   Uses the method of multipliers to solve the algorithm
    %
    %   minimize ||X||_* + \lambda ||X||_{1,2}
    %   subject to yhat = Ahat(X)
    %
    %   References:
    %
    %   [1] Samuel Burer and Renato D. C. Monteiro.  "A nonlinear programming
    %   algorithm for solving semidefinite programs via low-rank
    %   factorization." Mathematical Programming (Series B), Vol. 95,
    %   2003. pp 329-357.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The parameters are
    %
    %   maxOutIter = maximum number of outer iterations
    %
    %   rmseTol = the root mean square of the errors must drop below
    %   rmseTol before termination
    %
    %   sigmaInit = starting value for the Augmented Lagrangian parameter
    %
    %   LR1 = the feasibility must drop below LR1 times the previous
    %   feasibility before updating the Lagrange multiplier.
    %   Otherwise we update sigma.
    %
    %   LR2 = the amount we update sigma by if feasibility is not improved
    %
    %   progTol = like LR1, but much smaller.  If feasibility is not
    %   improved by progtol for numbaditers iterations, then we decide
    %   that further progress cannot be made and quit
    %
    %   numbaditers = the number of iterations where trivial progress
    %   is made improving the feasibility of L and R before algorithm
    %   termination.


pars.maxOutIter = maxInOutIter;%25
pars.rmseTol = 1e-8;
pars.sigmaInit = 1e4;
pars.LR1 = 0.25;
pars.LR2 = 10;
pars.progTol = 1e-3;
pars.numbaditers = 6;

% options for minFunc
options = [];
options.display = 'none';
options.maxFunEvals = 50000;
options.Method = 'lbfgs';


%% Derived constants:
siglen = length(conv_zh);
p = siglen;
p1 = sqrt(p);
p2 = sqrt(p);
mat = @(x) reshape(x,p1,p2);
vec = @(x) x(:);
n1 = length(C1T(mat(conv_zh))); % There must be a better way to do this...
n2 = length(C2T(mat(conv_zh)));


% we actually run the equality constraint in the frequency domain:
meas_fft = vec(fft2(mat(conv_zh)))/siglen;

% If starting points are specified, use them. otherwise use the default.
% A better default might speed things up.  Not sure.
if nargin>6 & ~isempty(ZInit),
    Z = ZInit;
else
    Z = 1e-2*randn(n1,maxrank);
end

if nargin>7 & ~isempty(HInit),
    H = HInit;
else
    H = 1e-2*randn(n2,maxrank);
end

% y is the Lagrange multiplier
y = zeros(siglen,1);
% sigma is the penalty parameter
sigma = pars.sigmaInit;

% compute initial infeasibility
dev = zeros(p,1);
for i = 1:maxrank
    dev = dev+vec(fft2(mat(C1(Z(:,i)))).*fft2(mat(C2(H(:,i)))));
end
dev = vec(ifft2(mat(dev))) - conv_zh;
vOld = norm(dev,'fro')^2;

v = vOld;
badcnt = 0;
T0 = clock;

fprintf('|      |          |          | iter  | tot   |\n');
fprintf('| iter |  rmse    |  sigma   | time  | time  |\n');
for j=1:46, fprintf('-'); end
fprintf('\n');

iterCount = 0;

for outIter=1:pars.maxOutIter,
    
    T1 = clock;
    
    % minimize the Augmented Lagrangian using BFGS
    [x,mval] = minFunc(@subproblem_costV0,[Z(:);H(:)],options,C1,C2,C1T,C2T,maxrank,meas_fft,y,sigma,siglen,n1,n2);
    Z = reshape(x(1:(n1*maxrank)),n1,maxrank);
    H = reshape(x((n1*maxrank) + (1:(n2*maxrank))),n2,maxrank);
    
    % compute the equation error
    dev = zeros(p,1);
    for i = 1:maxrank
        dev = dev+vec(fft2(C1(Z(:,i))).*fft2(C2(H(:,i))))/siglen;
    end
    dev = dev - meas_fft;
    vLast = v;
    v = norm(dev)^2; % v is sum of the squares of the errors
    
    % if unable to improve feasibility for several iterations, quit.
    if abs(vLast-v)/vLast<pars.progTol,
        badcnt = badcnt+1;
        if badcnt>pars.numbaditers,
            fprintf('\nunable to make progress. terminating\n');
            break;
        end
    else
        badcnt = 0;
    end
    
    % print diagnostics
    fprintf('| %2d   | %.2e | %.2e |  %3.0f  |  %3.0f  |\n',...
        outIter,sqrt(v/siglen),sigma,etime(clock,T1),etime(clock,T0));
    
    % if solution is feasible to specified tolerance, we're done
    if sqrt(v/siglen)<pars.rmseTol,
        break;
        % if normed feasibility is greatly reduced, update Lagrange multipliers
    elseif v < pars.LR1*vOld
        y = y - sigma*dev;
        vOld = v;
        % if normed feasibility is not reduced, increase penalty term
    else
        sigma = pars.LR2*sigma;
    end
    
end
fprintf('elapsed time: %.0f seconds\n',etime(clock,T0));

