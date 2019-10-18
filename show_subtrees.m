close all
% fname_fmt = 'build-reldeb/depth_%d.vol';
% 
% for d = 0:6
%     n = 2^d;
%     fname = sprintf(fname_fmt, n);
%     f = fopen(fname);
%     vol = reshape(fread(f, n^3, 'float32'), [n,n,n]);
%     figure(); max_vol_projection(vol,true)
%     fclose(f);
% end

f1 = '/home/miguel/gitrepos/backprojector/build-release/original.chunk';
f2 = '/home/miguel/gitrepos/backprojector/build-release/comp.chunk';
r2 = 696;
f = fopen(f1);
orig = permute(reshape(fread(f, 4096*1827), [1827, 4096]), [2,1]);
fclose(f);
f = fopen(f2);
comp = permute(reshape(fread(f, 4096*r2), [r2, 4096]), [2,1]);
fclose(f);
figure();
subplot(2,1,1);
imshow(tonemap(orig));
axis equal
subplot(2,1,2);
imshow(tonemap(comp));
axis equal