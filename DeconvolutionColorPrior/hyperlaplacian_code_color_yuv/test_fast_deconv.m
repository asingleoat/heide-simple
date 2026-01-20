% Test the fast deconvolution  method presented in the paper
% D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
% Priors", Proceedings of NIPS 2009.

clear all;
close all;

% read in and normalize input data - if data is not normalized, change
% lambda parameter accordingly
image_filename = 'houses_big.jpg'; 
I = imread(image_filename);
I = imresize(I, 0.15); %Sample down for faster deconvolution
I = double(I);
I = I ./ max(I(:));

%Apply inverse gamma
y = I .^ (2.2);

% load 2 pre-defined kernels
%{
load kernels.mat;

% which kernel to use
K_blur = cell(0);
kernel = kernel1;
ks = floor((size(kernel, 1) - 1)/2);

K_blur_disp = zeros(size(kernel, 1), size(kernel, 2), size(I, 3));
for ch = 1:size(I,3)
    K_blur{ch} = kernel;
    K_blur_disp(:,:,ch) = kernel;
end
K_blur_disp = K_blur_disp ./ max(K_blur_disp(:));
%}


%Define square sized blur kernel
blur_size = 9; %has to be an odd number !
K_blur = cell(0); %Blur for all channels

%Loaded kernel
inc_blur_exp = 1.7;
max_blur_size = round(blur_size * (inc_blur_exp)^(size(I,3) - 1));
max_blur_size = max_blur_size + (1 - mod(max_blur_size,2));
max_radius = floor(max_blur_size / 2);
K_blur_disp = zeros(max_blur_size, max_blur_size, size(I,3));
for ch = 1:size(I,3)
    %K_blur{ch} = imread('kernels/planoconvex_lens_kernel.png');
    %K_blur{ch} = imread('kernels/fading.png');
    K_blur{ch} = imread('kernels/snake.png');
    
    %Curr blur size
    curr_blur_size = round(blur_size* (inc_blur_exp)^(ch - 1));
    curr_blur_size = curr_blur_size + (1 - mod(curr_blur_size,2));
    
    K_blur{ch} = imresize( K_blur{ch}, [curr_blur_size, curr_blur_size] , 'bicubic');
    K_blur{ch} = img_to_norm_grayscale(K_blur{ch});
    K_blur{ch} = K_blur{ch} ./ sum(K_blur{ch}(:)); %Finally normalize kernel to sum of 1
    
    %Store kernel for display
    curr_blur_radius = floor(curr_blur_size / 2);
    K_blur_disp( 1 + max_radius + (-curr_blur_radius:curr_blur_radius),1 + max_radius + (-curr_blur_radius:curr_blur_radius), ch) = ...
        (K_blur{ch} ./ max(K_blur{ch}(:)));
end
ks = max_radius;

% parameter values; other values such as the continuation regime of the
% parameter beta should be changed in fast_deconv.m
lambda = 2e3;
alpha = 0.65;
rho_yuv = [0.1, 1.0, 1.0];
w_rgb = [1/4, 1/2, 1/4];
theta = max_blur_size^2 * (1e-3)^2 * 1e3

% convolve with kernel1 and add noise
yorig = y;
for ch = 1:size(y,3)
    y(:,:,ch) = imfilter(yorig(:,:,ch), K_blur{ch}, 'conv', 'replicate');
end
y = y + 0.001*randn(size(y));
y = max( min( y, 1 ), 0);
%y = double(uint8(y .* 255))./255;

% Check if Eero Simoncell's function exists
if (exist('pointOp') ~= 3) 
  fprintf('WARNING: Will use slower interp1 for LUT interpolations. For speed please see comments at the top of fast_deconv.m\n'); 
end;

clear persistent;

figure()
subplot(1,4,1), imagesc( K_blur_disp ), axis image, title('Kernel');
subplot(1,4,2), imagesc(yorig), axis image, title('Orig');
subplot(1,4,3), imagesc(y), axis image, title('Blurred');
pause(0.5)

%SNR before
snr_blur = snr(y, 0, yorig);

tic;
[x] = fast_deconv(y, K_blur, lambda, rho_yuv, w_rgb, theta, alpha);
timetaken = toc;

%SNR after
snr_recon = snr(x, 0, yorig);

%Apply gamma
x = min(max(x, 0), 1);
x = x .^ (1.0/2.2);

yorig = min(max(yorig, 0), 1);
yorig = yorig .^ (1.0/2.2);

y = min(max(y, 0), 1);
y = y .^ (1.0/2.2);

figure; imagesc([yorig y x]), axis image; 
tt = sprintf('Original  Blurred (SNR %.2f) Reconstructed (SNR %.2f)', ...
             snr_blur, snr_recon);
title(tt);

fprintf('Time taken for image of size %dx%d is %.3f\n', size(I,2), size(I,1), ...
        timetaken);

