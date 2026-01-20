function [yout] = fast_deconv_yuv(yin, k, lambda, rho_yuv, w_rgb, theta, alpha, yout0)
%
% Reimplementation of:
% C. J. Schuler, M. Hirsch, S. Harmeling and B. Scholkopf: 
% "Non-stationary Correction of Optical Aberrations", Proceedings of ICCV 2011.
% based on code from:
% D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
% Priors", Proceedings of NIPS 2009.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Prepare y
% edgetaper to better handle circular boundary conditions
%Compute max blur radius
ks = max(size(k{1}, 1), size(k{1}, 2));
for ch = 2:size(yin,3)
    ks = max(ks, max(size(k{ch}, 1), size(k{ch}, 2)));
end
ks = max(0,  floor(ks / 2) );

%Pad
yin_tmp = yin;
yin = zeros(size(yin_tmp,1) + 2*ks, size(yin_tmp,2) + 2*ks, size(yin_tmp,3));
for ch = 1:size(yin,3)
    yin(:,:,ch) = padarray(yin_tmp(:,:,ch), [1 1]*ks, 'replicate', 'both');
end

%Edgetaper
for a=1:4
  for ch = 1:size(yin,3)
        yin(:,:,ch) = edgetaper(yin(:,:,ch), k{ch});
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actual minimization

% continuation parameters
beta = 1;
beta_rate = 2*sqrt(2);
beta_max = 2^8;

% number of inner iterations per outer iteration
mit_inn = 1;

[m n c] = size(yin); 
% initialize with input or passed in initialization
if (nargin >= 8)
  yout = yout0;
else
  yout = yin; 
end;

% make sure k is a odd-sized
for ch = 1:c
    if ((mod(size(k{ch}, 1), 2) ~= 1) | (mod(size(k{ch}, 2), 2) ~= 1))
      fprintf('Error - blur kernel k must be odd-sized.\n');
      return;
    end;
end

% compute constant quantities
% see Eqn. (3) of paper
%K'B
KtB = zeros(size(yin));
for ch = 1:c
    KtB(:,:,ch) = imconv(yin(:,:,ch), fliplr(flipud(k{ch})), 'same');
end

%Preallocated stuff
youtx = zeros(size(yin));
youty = zeros(size(yin));
Wxtx = zeros( size(yout) );
Wyty = zeros( size(yout) );
Wx = zeros( size(yout) );
Wy = zeros( size(yout) );
youtk = zeros( size(yout) );

%C is RGB_to_YUV conversion matrix
RGB_to_YUV = [ 0.299,   0.587,     0.114; ...
              -0.14713, -0.28886,  0.436; ...
               0.615,   -0.51499, -0.10001];
C = RGB_to_YUV;

%Convert yout to YUV
yout_yuv = img_mult(yout, C);   

% x and y gradients of yout_yuv
dxf=[1 -1];
dyf=[1; -1];

%in YUV now
for ch = 1:c
    youtx_tmp = imconv(yout_yuv(:,:,ch), dxf, 'full');
    youtx(:,:,ch) = youtx_tmp(:, 1:end - 1);

    youty_tmp = imconv(yout_yuv(:,:,ch), dyf, 'full');
    youty(:,:,ch) = youty_tmp(1:end - 1, :);
end

% store some of the statistics
costfun = [];
Outiter = 0;

%% Main loop
while beta < beta_max
    Outiter = Outiter + 1; 
    fprintf('Outer iteration %d; beta %.3g\n',Outiter, beta);

    for Inniter = 1:mit_inn
      
      if (1)
        %%% Compute cost function - uncomment to see the original
        % minimization function costs at every iteration
        youtk(:) = 0;
        for ch = 1:c
            youtk(:,:,ch) = imconv( yout(:,:,ch), k{ch}, 'same' );
        end
        
        % likelihood term
        cost = 0;
        for ch = 1:c
            lh = sum(sum((youtk(:,:,ch) - yin(:,:,ch)).^2 ));
            l2rgb = sum(sum( yout(:,:,ch).^2 ));
            youtx_ch = youtx(:,:,ch);
            youty_ch = youty(:,:,ch);
            cost = cost + (lambda/2) * lh +  rho_yuv(ch) * ( sum(abs(youtx_ch(:)).^alpha) + sum(abs(youty_ch(:)).^alpha) ) + theta * w_rgb(ch) * l2rgb;
        end
        fprintf('Iteration %d; cost %.3g\n', (Outiter - 1) * mit_inn + Inniter, cost);
        
        costfun = [costfun, cost];
      end;
      %
      % w-subproblem: eqn (5) of paper
      %  
      Wx(:) = 0;
      Wy(:) = 0;
      for ch = 1:c
          Wx(:,:,ch) = solve_image(youtx(:,:,ch), beta / rho_yuv(ch) , alpha); 
          Wy(:,:,ch) = solve_image(youty(:,:,ch), beta / rho_yuv(ch) , alpha);
      end
                   
      % 
      %   x-subproblem: eqn (3) of paper
      % 
      % The transpose of x and y gradients; if other gradient filters
      % (such as higher-order filters) are to be used, then add them
      % below the comment above as well
      
      %Compute RHS
      
      %Filter terms:
      Wxtx(:) = 0;
      Wyty(:) = 0;
      for ch = 1:c
          Wxtx_tmp = imconv(Wx(:,:,ch), fliplr(flipud(dxf)), 'full');
          Wxtx(:,:,ch) = Wxtx_tmp(:, 2:end);

          Wyty_tmp = imconv(Wy(:,:,ch), fliplr(flipud(dyf)), 'full');
          Wyty(:,:,ch) = Wyty_tmp(2:end, :);
      end
      
      %Apply C'
      Wxtx = img_mult(Wxtx, C');  
      Wyty = img_mult(Wyty, C');   
      
      Wxx = Wxtx + Wyty;
      
      %Add all up to get rhs of linear system
      rhs = Wxx + lambda/beta * KtB;
        
      %Solve with CG
      yout = solve_cg_subproblem(lambda, beta, dxf, dyf, k, rhs, C, w_rgb, theta, yout );
      
      % update the gradient terms with new solution
      %Convert yout to YUV
      yout_yuv = img_mult(yout, C);   

      for ch = 1:c
          youtx_tmp = imconv(yout_yuv(:,:,ch), dxf, 'full');
          youtx(:,:,ch) = youtx_tmp(:, 1:end - 1);

          youty_tmp = imconv(yout_yuv(:,:,ch), dyf, 'full');
          youty(:,:,ch) = youty_tmp(1:end - 1, :);
      end

    end %inner
    beta = beta*beta_rate;
end %Outer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Remove padding in y
yout = yout(ks+1:end-ks, ks+1:end-ks,:);

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CG solver for quadratic subproblem
function x = solve_cg_subproblem(lambda, beta, dxf, dyf, k, b, C, w_rgb, theta, x_0 )
    %Solves Ax = b with
    % A = (C' * dxf' * dxf * C + C' * dyf' * dyf * C + lambda/beta * K'* K )
    
    fprintf('CG: ')
    
	%Initialize x
    x = x_0;

    %Compute residual
    
    [m,n,c] = size(b);
    x_k_t_k = zeros(size(b));
    for ch = 1:c
        x_k_t_k(:,:,ch) = imconv( imconv(x(:,:,ch), k{ch}, 'same'), fliplr(flipud(k{ch})), 'same' );
    end
    
    %In yuv
    %Apply C
    x_yuv = img_mult(x, C);   
          
    x_dx_t_dx = zeros(size(b));
    x_dy_t_dy = zeros(size(b));
    
    for ch = 1:c
        
        tmp = imconv(x_yuv(:,:,ch), dxf, 'full');
        tmp = imconv(tmp(:, 1:end - 1), fliplr(flipud(dxf)), 'full');
        tmp = tmp(:, 2:end);
        x_dx_t_dx(:,:,ch) = tmp;
        
        tmp = imconv(x_yuv(:,:,ch), dyf, 'full');
        tmp = imconv(tmp(1:end - 1, :), fliplr(flipud(dyf)), 'full');
        tmp = tmp(2:end, :);
        x_dy_t_dy(:,:,ch) = tmp;
    end
    
    %Apply C'
    x_dx_t_dx = img_mult(x_dx_t_dx, C');  
    x_dy_t_dy = img_mult(x_dy_t_dy, C'); 
    
    %Compute robust matrix term
    rb_x = x;
    for ch = 1:c
       rb_x(:,:,ch) = theta/beta * 2 * w_rgb(ch) * rb_x(:,:,ch);
    end
    
    Ax = x_dx_t_dx + x_dy_t_dy + lambda/beta *  x_k_t_k + rb_x;
    
    r = b - Ax;

    %Do cg iterations
    cg_tol = 1e-5;
    cg_iter = min(50, size(x(:),1));
    %cg_fig = figure();
    
    %Preallocated stuff
    p_k_t_k = zeros(size(b));
    p_dx_t_dx = zeros(size(b));
    p_dy_t_dy = zeros(size(b));
    
    for iter = 1:cg_iter  
        rho = (r(:)'*r(:));

        if ( iter > 1 ),                       % direction vector
            beta = rho / rho_1;
            p = r + beta*p;
        else
            p = r;
        end
         
        %Compute Ap         
        for ch = 1:c
            p_k_t_k(:,:,ch) = imconv( imconv(p(:,:,ch), k{ch}, 'same'), fliplr(flipud(k{ch})), 'same' );
        end
         
        %Apply C
        p_yuv = img_mult(p, C);   
        
        for ch = 1:c
            tmp = imconv(p_yuv(:,:,ch), dxf, 'full');
            tmp = imconv(tmp(:, 1:end - 1), fliplr(flipud(dxf)), 'full');
            tmp = tmp(:, 2:end);
            p_dx_t_dx(:,:,ch) = tmp;

            tmp = imconv(p_yuv(:,:,ch), dyf, 'full');
            tmp = imconv(tmp(1:end - 1, :), fliplr(flipud(dyf)), 'full');
            tmp = tmp(2:end, :);
            p_dy_t_dy(:,:,ch) = tmp;
        end
        
        %Apply C'
        p_dx_t_dx = img_mult(p_dx_t_dx, C');  
        p_dy_t_dy = img_mult(p_dy_t_dy, C'); 
        
        %Compute robust matrix term
        rb_p = p;
        for ch = 1:c
            rb_p(:,:,ch) = theta/beta * 2 * w_rgb(ch) * rb_p(:,:,ch);
        end

        Ap = p_dx_t_dx + p_dy_t_dy + lambda/beta *  p_k_t_k + rb_p;
         
        %Cg update
        q = Ap;
        alpha = rho / (p(:)'*q(:) );
        x = x + alpha * p;                    % update approximation vector

        r = r - alpha*q;                      % compute residual

        rho_1 = rho;
         
        %figure(cg_fig);
        %imshow( x ./ max(abs(x(:))) ), title(sprintf('Cg iter %d',iter));
        %pause(0.1)
        
        fprintf('.')
         
        % Check for convergence
        if norm(r(:)) <= cg_tol
           break;
        end
    end
    
    fprintf('\n')
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imconv
function F_filt = imconv(F,K,output)

%Convolution with boundary condition
%General: F_filt = imfilter(F, K, 'full', 'conv', 'replicate');

%Speedup for small two entry kernels (full)
if size(K,1) == 1 && size(K,2) == 2 && strcmp('full', output)
    F_filt = K(1,2)* F(:,[1 1:end],:) + K(1,1)*F(:,[1:end end],:);
elseif size(K,1) == 2 && size(K,2) == 1 && strcmp('full', output)
    F_filt = K(2,1)* F([1 1:end],:,:) + K(1,1)*F([1:end end],:,:);
else
    %Speed up with fft if kernel large:
    if (max(size(K)>25))
        
        %Get pad and blur radius
        pad = 2 * size(K,1);
        blur_radius = floor(size(K,1)/2);
        
        %Boundary conditions
        F_pad = boundary_transform_deblurring(F, 'add', pad, true); %No tapering for speedup
        
        %Initialize Kernel
        K_pad = zeros(size(F_pad));
        K_pad(1:size(K,1), 1:size(K,2)) = K;

        %Save K in padded array
        K_pad = circshift(K_pad, [-floor(size(K,1)/2), -floor(size(K,2)/2)]);

        %Compute fft
        F_filt = real(ifft2(fft2(F_pad) .* fft2(K_pad)));
        
        %Remove boundary
        if strcmp('same', output)
            F_filt = boundary_transform_deblurring(F_filt, 'rem', pad); 
        else
            F_filt = boundary_transform_deblurring(F_filt, 'rem', pad - blur_radius);  
        end
    else
        %General model
        F_filt = imfilter(F, K, output, 'conv', 'replicate');
    end
end

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Img_mult
function Cx = img_mult(x,C)

	%Expects a 3x3 image multiplication matrix
	if( ~isequal(size(C), [3,3]) )
		error('Expected 3x3 matrix in image mult.\n')
    end

    %Vectorize
    x_vec = [reshape( x(:,:,1), [], 1) , reshape( x(:,:,2), [], 1) , reshape( x(:,:,3), [], 1) ];
    Cx_vec = C * x_vec';
    Cx_vec = Cx_vec';

    Cx = cat(3, reshape( Cx_vec(:,1), size(x, 1), size(x, 2)) ,...
                reshape( Cx_vec(:,2), size(x, 1), size(x, 2)) ,...
                reshape( Cx_vec(:,3), size(x, 1), size(x, 2)) );
            
return;