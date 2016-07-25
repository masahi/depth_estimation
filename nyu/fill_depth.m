addpath('toolbox')

load file_names
files = file_names
n_files = size(files, 1)

for i=1:n_files
    disp(i)
    
    f_name = strtrim(files(i,:));
    load(f_name); 
    
    depth_filled = fill_depth_cross_bf(rgb, depth);
    save(f_name, 'rgb', 'depth', 'depth_filled')
end    
