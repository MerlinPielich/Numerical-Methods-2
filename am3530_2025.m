clear all; close all;

% Choose the image:
gif_name = 'SL_simulated.gif';
%gif_name = 'SL_measured.gif';

% Time step:
dt = 1.e-5;
nt = 40;

fm = [];
fn = double(imread(gif_name))/255;
[nx,ny] = size(fn);
if ( strcmp(gif_name,'SL_simulated.gif') )
   fm = fn;
   sigma = 0.1;
   randn('state',0);
   fn = fm + sigma*randn(nx,ny);

% Plot the model image
   figure;
   hold on;
   title('Noise-free image');
   imagesc(fm,[0 1]);
   axis('square');
   axis off;
end

% Plot the noisy image
figure;
hold on;
title('Noisy image');
imagesc(fn,[0 1]);
axis('square');
axis off;

% Picard iteration
method = 1;
figure;
title('Picard iteration');
sigma = zeros(9,1);
for i = 1:9
   fidelity = 10^(i-1);

% Add your code here. It this moment the filtered image is just a copy of the original image
   fs = fn;

   subplot(3,3,i)
   hold on;
   title(['Picard, \lambda = ', num2str(fidelity)]);
   imagesc(fs,[0 1]);
   axis('square');
   axis off;
   if (~isempty(fm) )
      sigma(i) = norm( fs - fm,'fro')/sqrt(nx*ny);
   end;
end
if (~isempty(fm) )
   figure;
   hold on;
   title('Standard deviation versus fidelity');
   plot([0:1:8],sigma);
   xlabel('Logarithm Fidelity');
   ylabel('\sigma');
end

% Explicit Euler
method = 2;
fidelity = 0;
% Add your code here. At this moment the filtered image is just a copy of the original image
fs = fn;
sum_x = ones(nt+1,1); % These you have to calculate 
err_x = ones(nt+1,1); % These you have to calcultate

% Plot the sum of the average pixel values:
t = [0:dt:(length(sum_x)-1)*dt];
figure;
plot(t,sum_x,'-x');
title('Sum of pixel values, Explicit Euler');
xlabel('Time');
ylabel('\Sigma x/N');

figure;
hold on;
title('Explicit Euler');
imagesc(fs,[0 1]);
axis('square');
axis off;
if (~isempty(fm) )
   figure;
   t = [0:dt:(length(err_x)-1)*dt];
   plot(t,err_x,'-x');
   title('Standard deviation versus time, Explicit Euler');
   xlabel('Time');
   ylabel('\sigma');
end

% Improved Euler
method = 3;
fidelity = 0;
% Add your code here. At this moment the filtered image is just a copy of the original image
fs = fn;
sum_x = ones(nt+1,1); % These you have to calculate 
err_x = ones(nt+1,1); % These you have to calcultate

% Plot the sum of the average pixel values:
t = [0:dt:(length(sum_x)-1)*dt];
figure;
plot(t,sum_x,'-x');
title('Sum of pixel values, Improved Euler');
xlabel('Time');
ylabel('\Sigma x/N');

figure;
hold on;
title('Improved Euler');
imagesc(fs,[0 1]);
axis('square');
axis off;
if (~isempty(fm) )
   figure;
   t = [0:dt:(length(err_x)-1)*dt];
   plot(t,err_x,'-x');
   title('Standard deviation versus time, Improved Euler');
   xlabel('Time');
   ylabel('\sigma');
end


