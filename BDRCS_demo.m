%% Blind Image Deblurring Using Row-Column Sparse Representations
%% Mohammad Tofighi

clear
close all
clc
%% Path
addpath(fullfile('minFunc'));
addpath(fullfile('minFunc_2012'));
addpath(fullfile('minFunc','compiled'));
addpath(fullfile('minFunc','mex'));

%% Load Data
rgb = double(imread('Im2.jpg'));
rgb = imresize(rgb, 0.25);
rgb = padarray(rgb, [5 5], 'symmetric');
x = mean(rgb,3);
x = double(x)/norm(x,'fro');
imagesc(x); title('Original shapes image'),colormap(gray),colorbar;

OrigImage = mat2gray(x);
OrigImage = im2double(OrigImage);

L1 = size(x,1);
L2 = size(x,2);
L = L1*L2;
%% Useful functions
mat = @(x) reshape(x,L1,L2);
vec = @(x) x(:);

%% Blur Kernel
blur_kernel = fspecial('motion',20,30);
% blur_kernel = fspecial('gaussian',15,1.5);
[K1 K2] = size(blur_kernel);
blur_kernel = blur_kernel/norm(blur_kernel,'fro');
w = zeros(L1,L2);
w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = blur_kernel; % K1 and K2 are odd; change if K1 and K2 even

%% Computing matrix B; see blind deconvolution using convex programming paper for notations
w_vec = vec(w);
Indw = abs(w_vec)>0;
j = 1;
K = nnz(Indw);
B = sparse(L,K);
h = zeros(K,1);
for i = 1:L
    if(Indw(i) == 1)
        B(i,j) = Indw(i);
        h(j) =w_vec(i);
        j = j+1;
    end
end

%% Define function BB
BB = @(x) mat(B*x); %function handle to multiply vector x by matrix B
BBT = @(x) B'*vec(x); %function handle to multiply vectorized x by matrix B
w1 = BB(h);   % w = Bh

%% 2D convolution
conv_wx = ifft2(fft2(x).*fft2(BB(h)));
conv_wx_image = fftshift(mat(conv_wx));

BlurImage = mat2gray(conv_wx_image);
BlurImage = im2double(BlurImage);

%% Compute and display wavelet coefficients of the original and blurred image
waveLevel = 6;
[alpha_conv,bookkeeping] = wavedec2(conv_wx_image,waveLevel,'db1');

[alpha_x,bookkeeping] = wavedec2(x,waveLevel,'db1');

%% C selected by wavelet coeffs of blurred\original\both image
alpha = alpha_x;
Ind = zeros(1,length(alpha));
Ind_alpha_conv = abs(alpha_conv)>0.005*max(abs(alpha_conv));%0.001~0.005
Ind_alpha_x = abs(alpha_x)>0.0005*max(abs(alpha_x));%0.0005

Ind_alpha_x = zeros(1,length(alpha)); % Do this if you want to kill support info. from original image
% Ind_alpha_conv = zeros(1,length(alpha)); % Do this if you want to kill support info. from blurred image
Ind = ((Ind_alpha_conv>0)|(Ind_alpha_x>0)); % Taking union of both supports

fprintf('Number of non-zeros in x estimated from the blurred image: %.3d\n', sum(Ind_alpha_conv));
fprintf('Number of non-zeros in x estimated from the original image: %.3d\n', sum(Ind_alpha_x));
fprintf('Union of the non-zero support from original and blurred image: %.3d\n', sum(Ind));
%% Compute matrix C; see blind deconvolution paper for notations
j = 1;
N = sum(Ind);
C = sparse(size(alpha,2),N);
for i = 1:size(alpha,2)
    if(Ind(i) == 1)
        C(i,j) = Ind(i);
        m(j) = alpha(i);
        mb(j) = alpha_conv(i);
        j = j+1;
    end
end
m = m';
Csize = size(C,1);

%% Define function CC
[c,bookkeeping] = wavedec2(conv_wx_image,waveLevel,'db1');
CC = @(x) waverec2(C*x,bookkeeping,'db1');
CCT = @(x) (C'*(wavedec2(x,waveLevel,'db1'))');

%% Approximated convolved image
x_hat = waverec2(C*m,bookkeeping,'db1');
fprintf('Original image vs Wavelet approximated image: %.3e\n', norm(x-x_hat,'fro')/norm(x,'fro'));

%% Convex Program for deconvolution
%=============================
w = zeros(L1,L2);
w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = ones(size(blur_kernel));
w_vec = vec(w);
Indw = zeros(L,1);
Indw = abs(w_vec)>0;
j = 1;
K = sum(Indw);
B = sparse(L,K);
h = zeros(K,1);
for i = 1:L
    if(Indw(i) == 1)
        B(i,j) = Indw(i);
        h(j) =w_vec(i);
        j = j+1;
    end
end
BB = @(x) mat(B*x); %function handle to multiply vector x by matrix B
BBT = @(x) B'*vec(x); %function handle to multiply vectorized x by matrix B

%=============================
RefineIter = 1;
% Initialize Z and H
maxrank = 4;
ZInit = [mb', zeros(length(mb'),maxrank-1)];
HInit = [ones(K,1)/K, zeros(K,maxrank-1)];
num_out_iter = 4;%outer iteration
threshold_ratio = 0.5;
threshold_ratio_image = threshold_ratio*0.01;
lambda = 0.01;
maxInIter = 10;%row sparsity inducer
LL = size(x);
KK = size(blur_kernel);
gg = 1;
wavelet = 'db1';
for ip = 1:RefineIter
    ip
    sparsity = 1;
    figure(1)
    [hEstt, ~, ~] = kernel_supp(mat,vec,conv_wx,CC,BB,maxrank,CCT,BBT,ZInit,HInit,...
        lambda,Indw,Ind,maxInIter,num_out_iter,threshold_ratio,threshold_ratio_image,LL,KK,blur_kernel,waveLevel,...
        bookkeeping,Csize,C,wavelet,sparsity);
    hEstt = abs(hEstt);
    
    w = zeros(L1,L2);
    w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2) = reshape(hEstt,KK);
    w_vec = vec(w);
    Indw1 = zeros(L,1);
    Indw1 = abs(w_vec)>0;
    K = numel(blur_kernel);
    B = sparse(L,K);
    lk = 1;
    IndB1 = L1/2-(K1+1)/2+2:L1/2+(K1+1)/2;
    IndB2 = L2/2-(K2+1)/2+2:L2/2+(K2+1)/2;
    for jk = 1:size(blur_kernel,2)
        for ik = 1:size(blur_kernel,1)
            IndB(lk) = sub2ind(size(w),IndB1(ik),IndB2(jk));
            lk = lk+1;
        end
    end
    for hk = 1:K
        B(IndB(hk),hk) = Indw1(IndB(hk));
    end
    BB = @(x) mat(B*x); %function handle to multiply vector x by matrix B
    BBT = @(x) B'*vec(x); %function handle to multiply vectorized x by matrix B
    
    if ip == 1
        N = nnz(Ind);
        C = sparse(Csize,N);
        j = 1;
        for i = 1:size(alpha,2)
            if(Ind(i) == 1)
                C(i,j) = Ind(i);
                mb(j) = alpha_conv(i);
                j = j+1;
            end
        end
        ZInit = mb';
        CC = @(x) waverec2(C*x,bookkeeping,'db1');
        CCT = @(x) (C'*(wavedec2(x,waveLevel,'db1'))');
    end
    
    HInit = hEstt(:);
    ZInit = mb';

    %% Define function CC
    CC = @(x) waverec2(C*x,bookkeeping,'db1');
    CCT = @(x) (C'*(wavedec2(x,waveLevel,'db1'))');
end

%% Estimates of x and w
wEst = abs(BB(hEstt));
figure;
imagesc(wEst), colormap(gray),colorbar, title('Estimated Kernel');

figure, subplot(121), imagesc(abs(HInit))
subplot(122), imagesc(blur_kernel(:))

figure, subplot(121), imagesc(abs(wEst))
subplot(122), imagesc(w1)

wFinal = wEst(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2,L2/2-(K2+1)/2+2:L2/2+(K2+1)/2);
X_final = wFinal(:)*ZInit';
X = blur_kernel(:)*m';
figure, subplot(121), imagesc(X_final(:, 1:200)); title('reconstructed')
subplot(122), imagesc(X(:, 1:200)); title('correct')
